# DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations (CVPR 2024)

<div align="center">

 <a href='https://arxiv.org/abs/2403.06951'><img src='https://img.shields.io/badge/arXiv-2403.06951-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://tianhao-qi.github.io/DEADiff/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


_**[Tianhao Qi*](https://github.com/Tianhao-Qi/), [Shancheng Fang](https://tothebeginning.github.io/), [Yanze Wu✝](https://tothebeginning.github.io/), [Hongtao Xie✉](https://imcc.ustc.edu.cn/_upload/tpl/0d/13/3347/template3347/xiehongtao.html), [Jiawei Liu](https://scholar.google.com/citations?user=X21Fz-EAAAAJ&hl=en&authuser=1), <br>[Lang Chen](https://scholar.google.com/citations?user=h5xex20AAAAJ&hl=zh-CN), [Qian He](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&authuser=1&user=9rWWCgUAAAAJ), [Yongdong Zhang](https://scholar.google.com.hk/citations?user=hxGs4ukAAAAJ&hl=zh-CN)**_
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


## 📝 Changelog
- __[2024.3.29]__: 🔥🔥 Release the inference code and pretrained checkpoint.
- __[2024.3.5]__: 🔥🔥 Release the project page.


## ⏳ TODO
- [x] Release the inference code.
- [ ] Release training data.


## 🧰 Models

|Model|Resolution|Checkpoint|
|:---------|:---------|:--------|
|DEADiff|512x512|[Hugging Face]()|

It takes approximately 2.5 seconds to generate a 512×512 image using a single NVIDIA A100 (80G) GPU. A GPU with at least 12G GPU memory is required to perform the inference process.

## ⚙️ Setup

```bash
conda create -n deadiff python=3.9.2
conda activate deadiff
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/salesforce/LAVIS.git -i "https://bytedpypi.byted.org/simple"
```

## 💫 Inference

1) Download the pretrained model and put it under ./pretrained/.
2) Run the commands in terminal.
```bash
sh scripts/inference.sh
```

## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****

## 🌟 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Tianhao-Qi/DEADiff_code&type=Date)](https://star-history.com/#Tianhao-Qi/DEADiff_code&Date)

## ✈️ Citation

```bibtex
@article{qi2024deadiff,
  title={DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations},
  author={Qi, Tianhao and Fang, Shancheng and Wu, Yanze and Xie, Hongtao and Liu, Jiawei and Chen, Lang and He, Qian and Zhang, Yongdong},
  journal={arXiv preprint arXiv:2403.06951},
  year={2024}
}
```

## 📭 Contact
If your have any comments or questions, feel free to contact [qth@mail.ustc.edu.cn](qth@mail.ustc.edu.cn)