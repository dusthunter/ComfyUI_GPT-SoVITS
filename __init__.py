import os
import sys

# 获取当前文件 (__init__.py) 所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# 将 GPT_SoVITS 和 tools 目录添加到 Python 的搜索路径
gpt_sovits_dir = os.path.join(current_dir, "GPT_SoVITS")
sys.path.insert(0, gpt_sovits_dir)

print("sys.path: ", sys.path)

from .gptsovits_nodes import *

# ComfyUI 节点类映射字典
NODE_CLASS_MAPPINGS = {
    "GPTSoVITS Ref Text Loader": Ref_Text_Loader, # 参考音频预处理节点
    "GPTSoVITS Target Text Preprocess": Target_Text_Preprocess, # 目标文本预处理节点
    "GPTSoVITS Inference": GPTSoVITS_Inference, # GPT-SoVITS 推理节点
} 

# ComfyUI 节点名称映射字典
NODE_DISPLAY_NAME_MAPPINGS = {
    "Ref_Text_Loader": "Reference Text Loader (v3)",
    "Target_Text_Preprocess": "Target Text Preprocess (v3)",
    "GPTSoVITS_Inference": "GPTSoVITS Inference Inference (v3)",
}

# __ALL__ = [
#     "NODE_CLASS_MAPPINGS",
#     "NODE_DISPLAY_NAME_MAPPINGS",
#     "WEB_DIRECTORY",
# ]