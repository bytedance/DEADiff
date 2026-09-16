# DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations ([CVPR 2024 Highlight](https://cvpr.thecvf.com/virtual/2024/poster/29773))

<div align="center">

 <a href='https://openaccess.thecvf.com/content/CVPR2024/html/Qi_DEADiff_An_Efficient_Stylization_Diffusion_Model_with_Disentangled_Representations_CVPR_2024_paper.html'><img src='https://img.shields.io/badge/Paper-CVPR%202024-blue.svg' alt='Paper: CVPR 2024'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://arxiv.org/abs/2403.06951'><img src='https://img.shields.io/badge/arXiv-2403.06951-b31b1b.svg' alt='arXiv: 2403.06951'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://tianhao-qi.github.io/DEADiff/'><img src='https://img.shields.io/badge/Project-Page-Green' alt='Project Page'></a>


_**[Tianhao Qi*](https://github.com/Tianhao-Qi/), [Shancheng Fang](https://csse.szu.edu.cn/en/pages/user/index?id=1362), [Yanze Wu✝](https://tothebeginning.github.io/), [Hongtao Xie✉](https://imcc.ustc.edu.cn/_upload/tpl/0d/13/3347/template3347/xiehongtao.html), [Jiawei Liu](https://scholar.google.com/citations?user=X21Fz-EAAAAJ&hl=en&authuser=1), <br>[Lang Chen](https://scholar.google.com/citations?user=h5xex20AAAAJ&hl=zh-CN), [Qian He](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&authuser=1&user=9rWWCgUAAAAJ), [Yongdong Zhang](https://scholar.google.com.hk/citations?user=hxGs4ukAAAAJ&hl=zh-CN)**_
<br><br>
(*Works done during the internship at ByteDance, ✝Project Lead, ✉Corresponding author)

From University of Science and Technology of China and ByteDance.

</div>


## 🔆 Introduction

**TL;DR:** We propose DEADiff, a generic method facilitating the synthesis of novel images that embody the style of a given reference image and adhere to text prompts.  <br>


### ⭐⭐ Stylized Text-to-Image Generation.

<div align="center">
<img src=docs/showcase_img.png>
<p>Stylized text-to-image results. Resolution: 512 x 512. (Compressed)</p>
</div>

### ⭐⭐ Style Transfer.

<div align="center">
<img src=docs/showcase_controlnet.png>
<p>Style transfer results with 
  <a href="https://github.com/lllyasviel/ControlNet.git" target="_blank">ControlNet</a>.
</p>
</div>


## 📝 Changelog

- **`[2025.07.06]`**: 🔥🔥 Release the ControlNet inference code for Canny-guided and depth-guided style transfer.
- **`[2024.04.03]`**: 🔥🔥 Release the inference code and pretrained checkpoint.
- **`[2024.03.05]`**: 🔥🔥 Release the project page.


## ⏳ TODO
- [x] Release the inference code.
- [ ] Release training data.


## ⚙️ Setup

```bash
conda create -n deadiff python=3.9.2
conda activate deadiff
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/salesforce/LAVIS.git@20230801-blip-diffusion-edit
pip install -r requirements.txt
pip install -e .
```

## 💫 Inference

1) Download the pretrained model from [Hugging Face](https://huggingface.co/qth/DEADiff/tree/main) and put it under ./pretrained/.
2) Run the commands in terminal.
```python3
python3 scripts/app.py
```
The Gradio app allows you to transfer style from the reference image. Just try it for more details.

Prompt: "A curly-haired boy"
![p](https://github.com/Tianhao-Qi/DEADiff_code_private/assets/37017794/bc0ebbf5-9bc9-4397-a0f6-dc291527571d)

Prompt: "A robot"
![p](https://github.com/Tianhao-Qi/DEADiff_code_private/assets/37017794/4b7bb264-aabb-42ae-bdc3-c20ebae5c0e6)

Prompt: "A motorcycle"
![p](https://github.com/Tianhao-Qi/DEADiff_code_private/assets/37017794/f23f8c4f-b72e-463c-9855-9767941e4932)

### ➕ Style Transfer with ControlNet

We support **style transfer with structural control** by combining DEADiff with [ControlNet](https://github.com/lllyasviel/ControlNet). This enables users to guide the spatial layout (e.g., edges or depth maps) of the generated images, while transferring the visual style from a reference image.

To perform style transfer with ControlNet, please download the following pretrained models:
- `control_sd15_canny.pth`: [Download](https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth) → place it under `./pretrained/`
- `control_sd15_depth.pth`: [Download](https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth) → place it under `./pretrained/`
- `dpt_hybrid-midas-501f0c75.pt` (for depth estimation): [Download](https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt) → place it under `ldm/controlnet/annotator/ckpts/`
These checkpoints are required for Canny and Depth-based ControlNet stylization modes.
Then run the following commands in terminal.
```python3
# Canny-based control
python3 scripts/app_canny_control.py
```

```python3
# Depth-based control
python3 scripts/app_depth_control.py
```

## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****

## ✈️ Citation

```bibtex
@inproceedings{qi2024deadiff,
  title={DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations},
  author={Qi, Tianhao and Fang, Shancheng and Wu, Yanze and Xie, Hongtao and Liu, Jiawei and Chen, Lang and He, Qian and Zhang, Yongdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2024},
  pages={8693--8702}
}
```

## 📭 Contact
If you have any comments or questions, feel free to contact [qth@mail.ustc.edu.cn](mailto:qth@mail.ustc.edu.cn).
