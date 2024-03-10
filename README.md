# DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations

<div align="center">

 <!-- <a href='https://arxiv.org/abs/2312.00330'><img src='https://img.shields.io/badge/arXiv-2312.00330-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -->
 <a href='https://tianhao-qi.github.io/DEADiff/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<!-- <a href='https://huggingface.co/spaces/liuhuohuo/StyleCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -->


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


<!-- ### 2. Style-Guided Text-to-Image Generation.

<div align="center">
<img src=docs/showcase_img.jpeg>
<p>Style-guided text-to-image results. Resolution: 512 x 512. (Compressed)</p>
</div> -->


## 📝 Changelog
- __[2024.3.5]__: 🔥🔥 Release the project page.


## ⏳ TODO
- [ ] Release the inference code.
- [ ] Release training data.


<!-- ## 🧰 Models

|Model|Resolution|Checkpoint|
|:---------|:---------|:--------|
|StyleCrafter|320x512|[Hugging Face](https://huggingface.co/liuhuohuo/StyleCrafter/tree/main)|


It takes approximately 5 seconds to generate a 512×512 image and 85 seconds to generate a 320×512 video with 16 frames using a single NVIDIA A100 (40G) GPU. A GPU with at least 16G GPU memory is required to perform the inference process.

## ⚙️ Setup

```bash
conda create -n stylecrafter python=3.8.5
conda activate stylecrafter
pip install -r requirements.txt
```

## 💫 Inference

1) Download all checkpoints according to the [instructions](./checkpoints/README.md)
2) Run the commands in terminal.
```bash
# style-guided text-to-image generation
sh scripts/run_infer_image.sh

# style-guided text-to-video generation
sh scripts/run_infer_video.sh
```
3) (Optional) Infernce on your own data according to the [instructions](./eval_data/README.md)



## 👨‍👩‍👧‍👦 Crafter Family
[VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter): Framework for high-quality text-to-video generation.

[ScaleCrafter](https://github.com/YingqingHe/ScaleCrafter): Tuning-free method for high-resolution image/video generation.

[TaleCrafter](https://github.com/AILab-CVC/TaleCrafter): An interactive story visualization tool that supports multiple characters.  

[LongerCrafter](https://github.com/arthur-qiu/LongerCrafter): Tuning-free method for longer high-quality video generation.  

[DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter) Animate open-domain still images to high-quality videos. -->


## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****

<!-- ## 🙏 Acknowledgements
We would like to thank [AK(@_akhaliq)](https://twitter.com/_akhaliq?lang=en) for the help of setting up online demo. -->


## 📭 Contact
If your have any comments or questions, feel free to contact [qth@mail.ustc.edu.cn](qth@mail.ustc.edu.cn)