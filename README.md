<div align="center">
<h1><font color=#6FA8DC>S</font><font color=#6FB150>h</font><font color=#E16766>o</font><font color=#F7B26B>w</font>-o</h1>
<h3>One Single Transformer to Unify Multimodal Understanding and Generation</h3>

[Jinheng Xie](https://sierkinhane.github.io/)<sup>1&#42;</sup>&nbsp;
[Weijia Mao](https://scholar.google.com/citations?hl=zh-CN&user=S7bGBmkyNtEC&view_op=list_works&sortby=pubdate)<sup>1&#42;</sup>&nbsp;
[Zechen Bai](https://www.baizechen.site/)<sup>1&#42;</sup>&nbsp;
[David Junhao Zhang](https://junhaozhang98.github.io/)<sup>1&#42;</sup>&nbsp;
<br>
Weihao Wang<sup>2</sup>&nbsp;
[Kevin Qinghong Lin](https://qinghonglin.github.io/)<sup>1</sup>&nbsp;
[Yuchao Gu](https://ycgu.site/)<sup>1</sup>
Zhijie Chen<sup>2</sup>&nbsp;
[Zhenheng Yang](https://scholar.google.com/citations?user=Ds5wwRoAAAAJ&hl=en)<sup>2</sup>&nbsp;
[Mike Zheng Shou](https://scholar.google.com/citations?hl=zh-CN&user=h1-3lSoAAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup> 

<sup>1</sup> Show Lab, National University of Singapore&nbsp; <sup>2</sup> Bytedance&nbsp;
 
[![arXiv](https://img.shields.io/badge/arXiv-<TBD>-<COLOR>.svg)]() [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fshowlab%2FShow-o&count_bg=%234DC621&title_bg=%23811AD2&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

<img src="docs/showo.png" width="1000">
An overview of Show-o. The input data, regardless of its modalities, is tokenized and then prompted into a formatted input sequence. Show-o processes text tokens autoregressively with causal attention and image tokens in (discrete) denoising diffusion modeling via full attention, and then generates the desired output. Specifically, Show-o is capable of handling image captioning, visual question answering, text-to-image generation, text-guided inpainting/extrapolation, and mixed modality generation.

## News
* **[2024-08-23]** We release the inference code of Show-o (**1.3B**) for multimodal understanding and generation including image captioning, visual question answering (VQA), text-to-image generation, text-guided inpaitning and extrapolation.

## TODO
- [X] Release the inference code.
- [ ] Release the training code (in the coming weeks).
- [ ] Scale up the model size (based on LLaMA3) and increase the number of training data.

## Getting Started
First, set up the environment:
```
pip3 install -r requirments.txt
```
Download model weight of a [pre-trained LLM (Phi-1.5)](https://huggingface.co/microsoft/phi-1_5):
```
git lfs install
git clone https://huggingface.co/microsoft/phi-1_5
```
Download model weights of [Show-o]() and put them to a directory in the structure below:
```
├── checkpoints/ 
|   ├── magvitv2.pth
|   ├── showo.bin
|   ├── showo_w_clip_vit.bin
|   ├── phi-1_5
```
Login your wandb account on your machine or server.
```
wandb login <your wandb keys>
```
Inference demo for **Multimodal Understanding** and you can view the results on wandb.
```
python3 inference_mmu.py config=configs/showo_demo_w_clip_vit.yaml \
mmu_image_root=./mmu_validation question='Please describe this image in detail. *** Do you think the image is unusual or not?' \
pretrained_model_path=./checkpoints/showo_w_clip_vit.bin
```
<img src="docs/github_mmu.png" width="1000">

Inference demo for **Text-to-Image Generation** and you can view the results on wandb.
```
python3 inference_t2i.py config=configs/showo_demo.yaml \
batch_size=32 validation_prompts_file=validation_prompts/showoprompts.txt \
guidance_scale=1.75 generation_timesteps=18 \
mode='t2i' pretrained_model_path=./checkpoints/showo.bin
```
<img src="docs/github_t2i.png" width="1000">

Inference demo for **Text-guided Inpainting** and you can view the results on wandb.
```
python3 inference_t2i.py config=configs/showo_demo.yaml \
batch_size=32 \
guidance_scale=1.75 generation_timesteps=16 \
pretrained_model_path=./checkpoints/showo.bin \
mode='inpainting' prompt='A blue sports car with sleek curves and tinted windows, parked on a bustling city street.' \
image_path=./inpainting_validation/bus.jpg inpainting_mask_path=./inpainting_validation/bus_mask.webp
```
<img src="docs/github_inpainting.png" width="1000">

Inference demo for **Text-guided Extrapolation** and you can view the results on wandb.
```
python3 inference_t2i.py config=configs/showo_demo.yaml \
batch_size=32 \
guidance_scale=1.75 generation_timesteps=16 \
pretrained_model_path=./checkpoints/showo.bin \
mode='extrapolation' extra_direction='left *** left *** left *** right *** right *** right' offset=0 prompt='a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees. *** a serene natural landscape featuring a clear, blue lake surrounded by lush green trees.' \
image_path=./inpainting_validation/alpine_lake.jpg
```
<img src="docs/github_extrapolation.png" width="1000">

### Citation
To cite the paper and model, please use the below:
```
@article{xie2024showo,
  title={Show-o: One Single Transformer to Unify Multimodal Understanding and Generation},
  author={Xie, Jinheng and Mao, Weijia and Bai, Zechen and Zhang, David Junhao and Wang, Weihao and Lin, Kevin Qinghong and Gu, Yuchao and Chen, Zhijie and Yang, Zhenheng and Shou, Mike Zheng},
  journal={},
  year={2024}
}
```
### Acknowledgments
This work is heavily based on [open-muse](https://github.com/huggingface/open-muse), [Phi-1.5](https://huggingface.co/microsoft/phi-1_5), [muse-maskgit-pytorch](https://github.com/lucidrains/muse-maskgit-pytorch), [maskgit](https://github.com/google-research/maskgit), [taming-transformers](https://github.com/CompVis/taming-transformers), [transformers](https://github.com/huggingface/transformers), [accelerate](https://github.com/huggingface/accelerate), [diffusers](https://github.com/huggingface/diffusers), and [webdatset](https://github.com/webdataset/webdataset). Thanks to all the authors for their great work.
