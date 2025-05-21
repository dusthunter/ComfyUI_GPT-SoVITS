'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import logging
import torchaudio,warnings
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, re, sys
import torch
from GPT_SoVITS.text.LangSegmenter import LangSegmenter

version=model_version=os.environ.get("version","v2")
path_sovits_v3="GPT_SoVITS/pretrained_models/s2Gv3.pth"
pretrained_sovits_name=["GPT_SoVITS/pretrained_models/s2G488k.pth", "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",path_sovits_v3]
pretrained_gpt_name=["GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt","GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "GPT_SoVITS/pretrained_models/s1v3.ckpt"]

parent_directory = os.path.dirname(os.path.abspath(__file__))
cnhubert_base_path = os.path.join(parent_directory,'GPT_SoVITS/pretrained_models/chinese-hubert-base')
bert_path = os.path.join(parent_directory,"GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")

path_sovits_v3=os.path.join(parent_directory,"GPT_SoVITS/pretrained_models/s2Gv3.pth")
is_exist_s2gv3=os.path.exists(path_sovits_v3)

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
# is_half=False
punctuation = set(['!', '?', '…', ',', '.', '-'," "])

from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from GPT_SoVITS.feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path

from GPT_SoVITS.module.models import SynthesizerTrn,SynthesizerTrnV3
import numpy as np

from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from time import time as ttime
from tools.my_utils import load_audio
from tools.i18n.i18n import I18nAuto, scan_language_list
from peft import LoraConfig, PeftModel, get_peft_model

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dict_language_v1 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("粤语"): "all_yue",#全部按中文识别
    i18n("韩文"): "all_ko",#全部按韩文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("粤英混合"): "yue",#按粤英混合识别####不变
    i18n("韩英混合"): "ko",#按韩英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",#多语种启动切分识别语种
}
dict_language = dict_language_v1 if version =='v1' else dict_language_v2

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

# Global variables to store currently loaded model paths
current_sovits_path = None
current_gpt_path = None

resample_transform_dict={}
def resample(audio_tensor, sr0):
    global resample_transform_dict
    if sr0 not in resample_transform_dict:
        resample_transform_dict[sr0] = torchaudio.transforms.Resample(
            sr0, 24000
        ).to(device)
    return resample_transform_dict[sr0](audio_tensor)

###todo:put them to process_ckpt and modify my_save func (save sovits weights), gpt save weights use my_save in process_ckpt
#symbol_version-model_version-if_lora_v3
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast,load_sovits_new
def change_sovits_weights(sovits_path): # ,prompt_language=None,text_language=None
    global vq_model, hps, version, model_version, dict_language, if_lora_v3, current_sovits_path
    # Check if the model is already loaded
    if sovits_path == current_sovits_path:
        print(f"SoVITS model {sovits_path} already loaded. Skipping.")
        return

    print(f"Loading SoVITS model from: {sovits_path}")
    version, model_version, if_lora_v3=get_sovits_version_from_path_fast(sovits_path)
    # print(sovits_path,version, model_version, if_lora_v3)
    if if_lora_v3==True and is_exist_s2gv3==False:
        info= "GPT_SoVITS/pretrained_models/s2Gv3.pth" + i18n("SoVITS V3 底模缺失，无法加载相应 LoRA 权重")
        print(info)
        raise FileExistsError(info)
    dict_language = dict_language_v1 if version =='v1' else dict_language_v2

    # Patch GPT_SoVITS.utils.HParams to be pickable
    # because it conflicts with utils.py in ComfyUI
    import GPT_SoVITS.utils as utils
    comfyui_utils = sys.modules['utils']
    sys.modules['utils'] = utils
    dict_s2 = load_sovits_new(sovits_path)
    # Restore patch for utils
    sys.modules['utils'] = comfyui_utils

    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if 'enc_p.text_embedding.weight'not in dict_s2['weight']:
        hps.model.version = "v2"#v3model,v2sybomls
    elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version=hps.model.version
    # print("sovits版本:",hps.model.version)
    if model_version!="v3":
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        model_version=version
    else:
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
    if ("pretrained" not in sovits_path):
        try:
            del vq_model.enc_q
        except:pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    if if_lora_v3==False:
        print("loading sovits_%s"%model_version,vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else:
        print("loading sovits_v3pretrained_G", vq_model.load_state_dict(load_sovits_new(path_sovits_v3)["weight"], strict=False))
        lora_rank=dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print("loading sovits_v3_lora%s"%(lora_rank))
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        # torch.save(vq_model.state_dict(),"merge_win.pth")
        vq_model.eval()
    # Update the current path
    current_sovits_path = sovits_path
    print(f"SoVITS model {sovits_path} loaded successfully.")

def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config, current_gpt_path
    # Check if the model is already loaded
    if gpt_path == current_gpt_path:
        print(f"GPT model {gpt_path} already loaded. Skipping.")
        return

    print(f"Loading GPT model from: {gpt_path}")
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    # Update the current path
    current_gpt_path = gpt_path
    print(f"GPT model {gpt_path} loaded successfully.")

os.environ["HF_ENDPOINT"]          = "https://hf-mirror.com"
import torch,soundfile
now_dir = os.getcwd()
import soundfile

def init_bigvgan():
    global bigvgan_model
    from GPT_SoVITS.BigVGAN import bigvgan
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(os.path.join(parent_directory,"GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x"), use_cuda_kernel=False)  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)

if model_version!="v3":bigvgan_model=None
else:init_bigvgan()


def get_spepc(hps, waveform,sample_rate):
    wav24k = waveform.squeeze(0).squeeze(0)
    if sample_rate!=hps.data.sampling_rate:
            wav24k=resample(wav24k,sample_rate)
    audio = wav24k.to(device) 
    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def clean_text_inf(text, language, version):
    language = language.replace("all_","")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

from GPT_SoVITS.text import chinese
def get_phones_and_bert(text,language,version,final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "all_zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "all_yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)

    return phones,bert.to(dtype),norm_text

from GPT_SoVITS.module.mel_processing import spectrogram_torch,mel_spectrogram_torch
spec_min = -12
spec_max = 2
def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1
def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min
mel_fn=lambda x: mel_spectrogram_torch(x, **{
    "n_fft": 1024,
    "win_size": 1024,
    "hop_size": 256,
    "num_mels": 100,
    "sampling_rate": 24000,
    "fmin": 0,
    "fmax": None,
    "center": False
})

def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

sr_model=None
def audio_sr(audio,sr):
    global sr_model
    if sr_model==None:
        from tools.audio_sr import AP_BWE
        try:
            sr_model=AP_BWE(device,DictToAttrRecursive)
        except FileNotFoundError:
            print(i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
            return audio.cpu().detach().numpy(),sr
    return sr_model(audio,sr)

def get_tts_wav(ref_audio, prompt_text, prompt_language, texts, text_language, top_k=20, top_p=0.6, temperature=0.6, speed=1,sample_steps=8,if_sr=False,pause_second=0.3):
    ref_waveform = ref_audio["waveform"]
    ref_sample_rate = ref_audio["sample_rate"]

    t = []
    t0 = ttime()

    prompt_text = prompt_text.strip("\n").strip()
    if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
    print(i18n("实际输入的参考文本:"), prompt_text)

    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half == True else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half == True:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)

    with torch.no_grad():
        wav16k = ref_waveform.squeeze(0).squeeze(0)
        if ref_sample_rate != 16000:
            wav16k = torchaudio.transforms.Resample(orig_freq=ref_sample_rate, new_freq=16000)(wav16k)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            print(i18n("参考音频在3~10秒范围外，请更换！"))
            raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
        if is_half == True:
            wav16k = wav16k.half().to(device)
        else:
            wav16k = wav16k.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1-t0)

    # --- Pre-compute reference features and prompt embeddings ---
    with torch.no_grad():
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
        phoneme_ids0=torch.LongTensor(phones1).to(device).unsqueeze(0)

        refer = get_spepc(hps, ref_waveform,ref_sample_rate).to(device).to(dtype)
        fea_ref_pre, ge_pre = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)

        ref_wav=ref_waveform.squeeze(0).to(device).float()
        if (ref_wav.shape[0] == 2):
            ref_wav = ref_wav.mean(0).unsqueeze(0)
        if ref_sample_rate!=24000:
            ref_wav=resample(ref_wav,ref_sample_rate)
        mel2_pre = mel_fn(ref_wav)
        mel2_pre = norm_spec(mel2_pre)
        T_min_ref = min(mel2_pre.shape[2], fea_ref_pre.shape[2])
        mel2_pre_sliced = mel2_pre[:, :, :T_min_ref]
        fea_ref_pre_sliced = fea_ref_pre[:, :, :T_min_ref]
    # --- Pre-computation done ---


    audio_opt = []
    for i_text,text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        text = text.strip()
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language, version)
        print(i18n("前端处理后的文本(每句):"), norm_text2)

        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()

        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt, # Reusing pre-computed prompt
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

        t3 = ttime()

        # --- Reusing pre-computed reference features ---
        phoneme_ids1=torch.LongTensor(phones2).to(device).unsqueeze(0)
        fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge_pre,speed) # Using pre-computed ge_pre

        mel2 = mel2_pre_sliced.clone() # Reusing pre-computed mel2_pre_sliced
        fea_ref = fea_ref_pre_sliced.clone() # Reusing pre-computed fea_ref_pre_sliced
        T_min = T_min_ref # Reusing pre-computed T_min_ref

        chunk_len = 934 - T_min
        cfm_resss = []
        idx = 0
        while (1):
            fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
            if (fea_todo_chunk.shape[-1] == 0): break
            idx += chunk_len
            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
            cfm_res = vq_model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0)
            cfm_res = cfm_res[:, :, mel2.shape[2]:]
            mel2 = cfm_res[:, :, -T_min:]
            fea_ref = fea_todo_chunk[:, :, -T_min:]
            cfm_resss.append(cfm_res)
        cmf_res = torch.cat(cfm_resss, 2)
        cmf_res = denorm_spec(cmf_res)
        if bigvgan_model==None:init_bigvgan()
        with torch.inference_mode():
            wav_gen = bigvgan_model(cmf_res)
            audio=wav_gen[0][0]#.cpu().detach().numpy()
        max_audio=torch.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:audio=audio/max_audio
        audio_opt.append(audio)
        if (norm_text2[-1] == "."): #只在句号时添加停顿
            audio_opt.append(zero_wav_torch)#zero_wav
        else:
            audio_opt.append(zero_wav_torch[:zero_wav_torch.shape[0] // 2])
        t4 = ttime()
        t.extend([t2 - t1,t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
    audio_opt=torch.cat(audio_opt, 0)#np.concatenate
    sr=hps.data.sampling_rate if model_version!="v3"else 24000
    if if_sr==True and sr==24000:
        print(i18n("音频超分中"))
        audio_opt,sr=audio_sr(audio_opt.unsqueeze(0),sr)
        max_audio=np.abs(audio_opt).max()
        if max_audio > 1: audio_opt /= max_audio
    else:
        audio_opt=audio_opt.cpu().detach().numpy()

    return sr, (audio_opt * 32767).astype(np.int16)
