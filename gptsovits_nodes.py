# gptsovits_nodes.py
import os
import re
import torch
import folder_paths # 导入 ComfyUI 文件夹路径管理模块
from .gptsovits_inference import get_tts_wav,dict_language_v2,change_gpt_weights,change_sovits_weights # 导入 TTS 生成函数
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

# --- Global Paths (Hardcoded as requested) ---

# cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base" # CNHubert 底模型路径
# bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large" # BERT 模型路径
# v3_bigvgan_path = "GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" # BigVGAN v3 模型路径
# gpt_v3_pretrained = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
# sovits_v3_pretrained = "GPT_SoVITS/pretrained_models/s2Gv3.pth"

dict_language_ui = {
    "中文": "all_zh", "英文": "en", "日文": "all_ja", "粤语": "all_yue", "韩文": "all_ko",
    "中英混合": "zh", "日英混合": "ja", "粤英混合": "yue", "韩英混合": "ko",
    "多语种混合": "auto", "多语种混合(粤语)": "auto_yue",
}

# 标点符号集合
splits = {"，", "。", "？", "！","：","；", "……",",", ".", "?", "!", ":",";", "…", "~", "—", } 

# --- ComfyUI Nodes ---
class Ref_Text_Loader:
    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点输入类型。
        """
        return {
        "required": {
            "ref_text": ("STRING", {"multiline": True}), # 目标文本输入 (多行文本框)
            "ref_language": (list(dict_language_ui.keys()), {"default": "中文"}),#type: ignore # 目标文本语言选择下拉菜单
        }
    }

    CATEGORY = "GPT-SoVITS" # 节点类别
    RETURN_TYPES = ("REF_FEATURES",) 
    RETURN_NAMES = ("REF_FEATURES",) 
    FUNCTION = "load_ref_text" # 节点执行函数

    def load_ref_text(self, ref_text, ref_language):
        ref_language_value = dict_language_ui[ref_language] # 获取内部目标语言标识符

        return ({"ref_text":ref_text,"ref_language":ref_language_value},) # 返回切分后文本及语言类型

class Target_Text_Preprocess:
    """
    目标文本预处理节点。
    """
    split_method_ui = {
        "自动切分": "auto",
        "整句切分": "sentence",
        "不做切分": "none",
    } # UI 文本切分方法到内部标识符的映射字典

    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点输入类型。
        """
        return {
        "required": {
            "target_text": ("STRING", {"multiline": True}), # 目标文本输入 (多行文本框)
            "target_language": (list(dict_language_ui.keys()), {"default": "中文"}),#type: ignore # 目标文本语言选择下拉菜单
            "split_method": (list(Target_Text_Preprocess.split_method_ui.keys()), {"default": "自动切分"}), # 文本切分方法选择下拉菜单
        }
    }

    CATEGORY = "GPT-SoVITS" # 节点类别
    RETURN_TYPES = ("TARGET_FEATURES",) 
    RETURN_NAMES = ("TARGET_FEATURES",) 
    FUNCTION = "preprocess_target_text" # 节点执行函数

    def preprocess_target_text(self, target_text, target_language, split_method):
        """
        预处理目标文本，根据切分方法分割文本。
        """
        split_method_value = Target_Text_Preprocess.split_method_ui[split_method] # 获取内部文本切分方法标识符
        target_language_value = dict_language_ui[target_language] # 获取内部目标语言标识符

        target_texts = self.split_text(target_text, split_method_value) # 根据切分方法分割文本

        # print("Split target texts:\n")
        # for text in target_texts:
        #     print(str(len(text))+": "+text+"\n")

        return ({"target_texts":target_texts,"target_language":target_language_value},) # 返回切分后文本及语言类型

    def split_text(self, text, split_method):
        """
        根据切分方法分割文本。

        Args:
            inp (str): 输入文本。
            split_method (str): 切分方法 ("auto", "sentence", "none").

        Returns:
            list: 切分后的文本段列表。
        """
        # 移除文本前后的换行符,并将连续的换行符替换为单个换行符
        text = text.strip("\n") 
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        if split_method == "none":
            return [text]  # 不切分
        elif split_method == "auto":
            return self.auto_split(text)  # 自动切分
        elif split_method == "sentence":
            return self.sentence_split(text)  # 按句子切分
        else:
            return [text]  # 默认不切分

    def sentence_split(self, text):
        """
        按句子切分：先根据回车换行切分，再根据标点符号切分。

        Args:
            inp (str): 输入文本。

        Returns:
            list: 切分后的文本段列表。
        """
        paragraphs = text.splitlines()  # 根据换行符切分段落
        sentences = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:  # 忽略空段落
                sentences.extend(self.split_by_punctuation(paragraph))
        return sentences
    
    def split_by_punctuation(self, text):
        """
        根据标点符号切分文本，处理数字中的小数点，并将标点符号附加到前一段文字。
        """
        parts = re.split(r'((?<!\d)\.(?!\d)|[。？！：；……]+|[?!:;…]+)', text)

        result = []
        current_part = ""

        for part in parts:
            if part is None or part == "":  # 过滤 None 和空字符串
                continue

            if re.match(r'(?<!\d)\.(?!\d)|[。？！：；……]+|[?!:;…]+', part):  # 检查是否是标点符号
                if current_part:  # 如果有前文，则将标点附加到前文
                    result.append((current_part + part).strip())
                    current_part = ""  # 重置 current_part
                else: #如果没有前文，说明标点在开头，一般是异常情况
                    result.append(part.strip()) #这种也保留下来
            else:
                current_part = part # 累积非标点符号部分

        if current_part:  # 处理最后剩余的非标点部分（如果有）
            result.append(current_part.strip())

        return result

    def auto_split(self, text):
        """
        自动切分：先根据回车换行切分，再根据标点符号切分，最后检查长度并按需切分。

        Args:
            inp (str): 输入文本。

        Returns:
            list: 切分后的文本段列表。
        """
        sentences = self.sentence_split(text)  # 先按句子切分
        final_splits = []
        for sentence in sentences:
            if len(sentence) > 50:
                final_splits.extend(self.split_long_sentence(sentence))
            else:
                final_splits.append(sentence)
        return final_splits

    def split_long_sentence(self, sentence):
        """
        递归地将句子切分成长度不超过50的子句。

        Args:
            sentence: 要切分的句子。
            splits: 用于切分句子的标点符号字符串。

        Returns:
            一个列表，包含切分后的子句。
        """

        if len(sentence) <= 50:
            return [sentence]

        best_split_pos = -1
        min_diff = float('inf')
        mid_point = len(sentence) // 2

        split_candidates = []
        # Find punctuation positions
        for i, char in enumerate(sentence):
            if char in splits:
                # 处理英文句号在数字中间的情况
                if char == '.' and 0 < i < len(sentence) - 1 and sentence[i-1].isdigit() and sentence[i+1].isdigit():
                    continue
                split_candidates.append(i)

        # Find best split position
        for i in split_candidates:
            diff = abs(i - mid_point)
            if diff <= min_diff:
                min_diff = diff
                best_split_pos = i

        if best_split_pos != -1 :
            part1 = sentence[:best_split_pos + 1].strip()
            part2 = sentence[best_split_pos + 1:].strip()

            # 递归调用split_sentence处理两个子句
            return self.split_long_sentence(part1) + self.split_long_sentence(part2)
        else:  # 如果没有找到合适的标点符号
             return [sentence]

class GPTSoVITS_Inference:
    """
    GPT 推理节点。
    """
    CATEGORY = "GPT-SoVITS" # 节点类别
    RETURN_TYPES = ("AUDIO",) # 节点输出类型
    RETURN_NAMES = ("AUDIO",) # 节点输出名称
    FUNCTION = "run_gpt_inference" # 节点执行函数

    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点输入类型。
        """
        # Directly use folder_paths.models_dir
        models_dir = folder_paths.models_dir # 获取 ComfyUI 模型目录      
        sovits_v3_dir = os.path.join(models_dir, "SoVITS_weights_v3") # SoVITS v3 模型权重目录
        gpt_v3_dir = os.path.join(models_dir, "GPT_weights_v3") # GPT v3 模型权重目录
        sovits_v3_files = [f for f in os.listdir(sovits_v3_dir) if f.endswith(".pth")] if os.path.exists(sovits_v3_dir) else [] # 获取 SoVITS v3 模型权重文件列表
        gpt_v3_files = [f for f in os.listdir(gpt_v3_dir) if f.endswith(".ckpt")] if os.path.exists(gpt_v3_dir) else [] # 获取 GPT v3 模型权重文件列表    
        sovits_choices =["s2Gv3.pth"] + sorted(sovits_v3_files) # 排序 SoVITS 模型选项
        gpt_choices = ["s1v3.ckpt"] + sorted(gpt_v3_files) # 排序 GPT 模型选项    

        return {
            "required": {
                "ref_audio": ("AUDIO", {"forceInput": True,}),
                "ref_features": ("REF_FEATURES", {"forceInput": True,}),
                "target_features": ("TARGET_FEATURES", {"forceInput": True,}),

                "gpt_model": (gpt_choices, {"default": gpt_choices[0] if gpt_choices else ""}), # GPT 模型选择下拉菜单
                "sovits_model": (sovits_choices, {"default": sovits_choices[0] if sovits_choices else ""}), # SoVITS 模型选择下拉菜单

                "seed": ("INT", {"default": 42}),

                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),# temperature 参数输入
                "top_k": ("INT", {"default": 15, "min": 1, "max": 100}),# top_k 参数输入
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),# top_p 参数输入
                "sample_steps": ("INT", {"default": 32, "min": 4, "max": 32, "step": 4}),# 采样步数输入
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),# 语速参数输入
                "pause_second": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.1}),
                "enable_upsampled": ("BOOLEAN", {"default": False, "label": "Toggle Switch"} ), # 是否启用超分
            },
        }
   
    # target_text, target_language
    def run_gpt_inference(self, ref_audio, ref_features, target_features, gpt_model, sovits_model, seed, temperature, top_k, top_p,sample_steps,speed,pause_second,enable_upsampled):
        models_dir = folder_paths.models_dir # 获取 ComfyUI 模型目录
        pretrained_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS/pretrained_models")
        if sovits_model == "s2Gv3.pth":
            sovits_model_path = os.path.join(pretrained_dir,sovits_model)
        else:
            sovits_model_path = os.path.join(models_dir, "SoVITS_weights_v3", sovits_model) # 构建 SoVITS 模型路径
        if gpt_model == "s1v3.ckpt":
            gpt_model_path = os.path.join(pretrained_dir,gpt_model)
        else:
            gpt_model_path = os.path.join(models_dir, "GPT_weights_v3", gpt_model) # 构建 GPT 模型路径

        ref_text = ref_features["ref_text"]
        ref_language = ref_features["ref_language"]

        target_text = target_features["target_texts"]
        target_language = target_features["target_language"]
        
        change_gpt_weights(gpt_model_path)
        change_sovits_weights(sovits_model_path)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        sr,audio = get_tts_wav(ref_audio, ref_text, ref_language, target_text, target_language, top_k, top_p, temperature, speed,sample_steps,enable_upsampled,pause_second)
        return ({"sample_rate": sr, "waveform": torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)},)
