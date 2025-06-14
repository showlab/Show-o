<div align="center">
<br>
<img src="docs/showo2.png" width="200">

[//]: # (<h3>Improved Unified Multimodal Models</h3>)

[Jinheng Xie](https://sierkinhane.github.io/)<sup>1</sup>&nbsp;
[Zhenheng Yang](https://scholar.google.com/citations?user=Ds5wwRoAAAAJ&hl=en)<sup>2</sup>&nbsp;
[Mike Zheng Shou](https://sites.google.com/view/showlab)<sup>1</sup> 

<sup>1</sup> [Show Lab](https://sites.google.com/view/showlab/home?authuser=0), National University of Singapore&nbsp; <sup>2</sup> Bytedance&nbsp;
 
[![ArXiv](https://img.shields.io/badge/Report-PDF-<COLOR>.svg)](https://github.com/showlab/Show-o/blob/main/show-o2/Show_o2.pdf) [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://github.com/showlab/Show-o/blob/main/docs/wechat_qa_3.jpg) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fshowlab%2FShow-o&count_bg=%234DC621&title_bg=%23811AD2&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
</div>

## News
* **[2025-06-12]** We release the Show-o2 models **with 1.5B and 7B LLM parameters** for multimodal understanding and generation.

## What is the new about Show-o2?
Below is an overview of **Show-o2**. We perform the unified learning of multimodal understanding and generation on the text token and **3D Causal VAE space**, which is scalable for **text, image, and video modalities**. A dual-path of spatial (-temporal) fusion is proposed to accommodate the distinct feature dependency  of multimodal understanding and generation. We employ specific heads with **autoregressive modeling and flow matching** for the overall unified learning of **multimodal understanding, image/video and mixed-modality generation.**
<img src="docs/overview.png" width="1000">
<br/>

<img src="docs/demo3.png" width="1000">

<table style="width:100%;">
  <tr>
    <td style="width:50%;">
      <img src="docs/videos/waves.gif" style="width:100%;" alt="GIF 1" />
    </td>
    <td style="width:50%;">
      <img src="docs/videos/sky.gif" style="width:100%;" alt="GIF 2" />
    </td>
  </tr>
</table>

<table style="width:100%;">
  <tr>
    <td style="width:25%;">
      <img src="docs/videos/i2v_3.gif" style="width:100%;" alt="GIF 1" />
    </td>
    <td style="width:25%;">
      <img src="docs/videos/i2v_4.gif" style="width:100%;" alt="GIF 2" />
    </td>
    <td style="width:25%;">
      <img src="docs/videos/i2v_1.gif" style="width:100%;" alt="GIF 3" />
    </td>
    <td style="width:25%;">
      <img src="docs/videos/i2v_2.gif" style="width:100%;" alt="GIF 4" />
    </td>
  </tr>
</table>


## TODO
- [X] Release the models for single image-text understanding and generation.
- [ ] Release the evaluation code.
- [ ] Release the training code.
- [ ] Release the models supporting image generation in a higher resolution (512x512 and 1024x1024) with better text rendering and mixed-modality generation.
- [ ] Release the models supporting image-to-video and text-to-video generation.

## Pre-trained Model Weigths
The Show-o2 checkpoints can be found on Hugging Face:
* [showlab/show-o2-1.5B](https://huggingface.co/showlab/show-o2-1.5B)
* [showlab/show-o2-7B](https://huggingface.co/showlab/show-o2-7B)

## Getting Started
First, set up the environment:
```
bash build_env.sh
```
Login your wandb account on your machine or server.
```
wandb login <your wandb keys>
```
Download Wan2.1 3D causal VAE model weight [here](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/Wan2.1_VAE.pth) and put it on the current directory.

Demo for **Multimodal Understanding** and you can find the results on wandb.
```
python3 inference_mmu.py config=configs/showo2_7b_demo_432x432.yaml \
                         mmu_image_path=./docs/mmu/pexels-jane-pham-727419-1571673.jpg question='Describe the image in detail.'

python3 inference_mmu.py config=configs/showo2_7b_demo_432x432.yaml \
                         mmu_image_path=./docs/mmu/pexels-fotios-photos-2923436.jpg question='请告诉我图片中写着什么？'

python3 inference_mmu.py config=configs/showo2_7b_demo_432x432.yaml \
                         mmu_image_path=./docs/mmu/pexels-taryn-elliott-4144459.jpg question='How many avocados (including the halved) are in this image? Tell me how to make an avocado milkshake in detail.'
```
<img src="docs/demo1.png" width="1000">

Demo for **Text-to-Image Generation** and you can find the results on wandb.
```
python3 inference_t2i.py config=configs/showo2_1.5b_demo_432x432.yaml \
                         batch_size=4 guidance_scale=7.5 num_inference_steps=50;

python3 inference_t2i.py config=configs/showo2_7b_demo_432x432.yaml \
                         batch_size=4 guidance_scale=7.5 num_inference_steps=50;
```
<img src="docs/demo2.png" width="1000">


### Citation
To cite the paper and model, please use the below:
```
@article{xie2025showo2,
  title={Show-o2: Improved Native Unified Multimodal Models},
  author={Xie, Jinheng and Yang, Zhenheng and Shou, Mike Zheng},
  journal={arXiv preprint},
  year={2025}
}
```
### Acknowledgments
This work is heavily based on [Show-o](https://github.com/showlab/show-o).
