# ComfyUI-GPT-SoVITS

这是一个用于[ComfyUI](https://github.com/comfyanonymous/ComfyUI)的GPT-SoVITS节点实现。支持将GPT-SoVITS的功能集成到ComfyUI工作流程中。

## 功能特点

- 支持SoVITS v3(支持LoRA)
- 支持音频超分辨率处理
- 支持自动切分（先按标点，再按句长）以提高合成准确率

## 预训练模型目录，自定义节点目录下GPT_SoVITS/pretrained_models
```bash
pretrained_models
├── chinese-hubert-base
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── chinese-roberta-wwm-ext-large
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer.json
├── fast_langdetect
│   └── lid.176.bin
├── gsv-v2final-pretrained
│   ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
│   ├── s2D2333k.pth
│   └── s2G2333k.pth
├── models--nvidia--bigvgan_v2_24khz_100band_256x
│   ├── bigvgan_generator.pt
│   └── config.json
├── s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
├── s1v3.ckpt
├── s2D488k.pth
├── s2G488k.pth
└── s2Gv3.pth

6 directories, 17 files
```
## 自己微调训练模型目录，请放到ComfyUI/models下，GPT_weights_v3和SoVITS_weights_v3目录

## 安装

1. 首先安装ComfyUI

2. 克隆本仓库到ComfyUI的custom_nodes目录:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dusthunter/ComfyUI-GPT-SoVITS
```

3. 安装依赖:

```bash
cd ComfyUI-GPT-SoVITS
pip install -r requirements.txt
```

4. 模型下载，


## 使用方法

1. 加载模型权重:
- GPT模型(.ckpt文件)
- SoVITS模型(.pth文件)

2. 准备参考音频(.wav或.mp3)

3. 设置生成参数:
- 参考文本
- 目标文本
- 语言设置
- 参数调整(top_k、top_p、temperature等)

4. 在ComfyUI中运行工作流

5. 预设工作流：workflow/GPT-SoVITS-v3.json


## 致谢

这是一个GPT-SoVITS的非官方ComfyUI实现:
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## 开源协议

本项目采用沿用GPT-SoVITS授权模式，采用 MIT License 许可证授权 - 详情请参阅相关许可证文件。
