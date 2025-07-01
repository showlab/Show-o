<div align="center">
<br>
<img src="docs/showo2.png" width="200">

[//]: # (<h3>Improved Unified Multimodal Models</h3>)

[Jinheng Xie](https://sierkinhane.github.io/)<sup>1</sup>&nbsp;
[Zhenheng Yang](https://scholar.google.com/citations?user=Ds5wwRoAAAAJ&hl=en)<sup>2</sup>&nbsp;
[Mike Zheng Shou](https://sites.google.com/view/showlab)<sup>1</sup> 

<sup>1</sup> [Show Lab](https://sites.google.com/view/showlab/home?authuser=0), National University of Singapore&nbsp; <sup>2</sup> Bytedance&nbsp;
 
[![ArXiv](https://img.shields.io/badge/Arxiv-<2506.15564>-<COLOR>.svg)](https://arxiv.org/abs/2506.15564) [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://github.com/showlab/Show-o/blob/main/docs/wechat_qa_3.jpg)
</div>

## News
* **[2025-06-27]** We release the training code for multimodal understanding and generation.
* **[2025-06-25]** We thank team [OneIG-Bench](https://github.com/OneIG-Bench/OneIG-Benchmark) for evaluating Show-o2 models on their new benchmark, in which our models have achieved leading performance in terms of Alignment and Reasoning metrics. The leaderboard is maintained [here](https://oneig-bench.github.io/).

<img src="docs/OnelG-Bench.jpg" width="1000">

* **[2025-06-20]** We are including more concurrent works in our [comparative analysis tables](https://github.com/showlab/Show-o/blob/main/show-o2/docs/comparative_analysis.png). Feel free to reach out to us if we miss your works.

* **[2025-06-19]** We release the Show-o2 models **with 1.5B and 7B LLM parameters** for multimodal understanding and generation.

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
- [X] Release the evaluation code.
- [X] Release the training code.
- [X] Release the models supporting image generation in a higher resolution (512x512 and 1024x1024) with better text rendering.
- [ ] Release the models supporting mixed-modality generation.
- [ ] Release the models supporting image-to-video and text-to-video generation.

## Pre-trained Model Weigths
The Show-o2 checkpoints can be found on Hugging Face:
* [showlab/show-o2-1.5B](https://huggingface.co/showlab/show-o2-1.5B)
* [showlab/show-o2-1.5B-HQ](https://huggingface.co/showlab/show-o2-1.5B-HQ) (text-to-image generation in resolutions of 512x512 and 1024x1024 with better text rendering)
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
python3 inference_t2i.py config=configs/showo2_1.5b_demo_1024x1024.yaml \
                         batch_size=4 guidance_scale=7.5 num_inference_steps=50;
         
python3 inference_t2i.py config=configs/showo2_1.5b_demo_512x512.yaml \
                         batch_size=4 guidance_scale=7.5 num_inference_steps=50;
                                      
python3 inference_t2i.py config=configs/showo2_1.5b_demo_432x432.yaml \
                         batch_size=4 guidance_scale=7.5 num_inference_steps=50;

python3 inference_t2i.py config=configs/showo2_7b_demo_432x432.yaml \
                         batch_size=4 guidance_scale=7.5 num_inference_steps=50;
```
<img src="docs/demo2.png" width="1000">

## Evaluation
### GenEval
```
# Generate images
bash evaluation/sample_geneval.sh

# Create an independent environment for GenEval (we use PyTorch 1.10.0)
git clone https://github.com/djghosh13/geneval.git
cd geneval
./evaluation/download_models.sh 'weights';
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x;
pip3 install -v -e .;
sudo pip3 install open-clip-torch;
sudo pip3 install clip-benchmark;
pip3 install -U openmim;
mim install mmcv-full;

# Evaluate
python3 evaluation/evaluate_images.py \
    "/path/to/your/generated/images" \
    --outfile "results.jsonl" \
    --model-path "./weights";
python3 evaluation/summary_scores.py "results.jsonl";
```
### DPG-Bench
```
# Generate images
bash evaluation/sample_dpg.sh

# Create an independent environment for DPG-Bench (we use PyTorch 2.5.1)
pip3 install modelscope==1.22.2; (if encountering issues, try modelscope==1.20.0)
pip3 install librosa==0.10.1
pip3 install git+https://github.com/One-sixth/fairseq.git
pip3 install opencv-python;
pip3 install unicodedata2;
pip3 install zhconv;
pip3 install rapidfuzz;
pip3 install numpy==1.23.5;
pip3 install addict;
pip3 install datasets==2.21.0;
pip3 install simplejson;
pip3 install sortedcontainers;

# Evaluate
cd evaluation;
bash dist_eval.sh /path/to/your/generated/images image_resolution
```

### Multimodal Understanding Benchmarks

Download and install [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) following their instructions.

[Download, add, replace some codes of lmms-eval](https://drive.google.com/file/d/1R9b5S1A0yYrcH7P-iiZc9XqG3B2SFKKn/view?usp=sharing) and structure them as follows:
```
├── lmms-eval/ 
|   ├── models
|   |   ├—— showo2_utils
|   |   ├—— ...
|   ├── __init__.py
|   ├── showo2_qwen2_5.py
|   ├── ...
```
``` 
# Evaluate
python3 -m accelerate.commands.launch --main_process_port 24348 \
    --num_processes=8 \
    -m lmms_eval \
    --model showo2_qwen2_5 \
    --model_args "config_file=/path/to/configs/showo2_7b_demo_432x432.yaml"  \
    --tasks mme,gqa,seedbench,mmbench,mmmu_val,mmstar,ai2d \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix showo2_qwen2_5 \
    --output_path ./logs/
``` 

## Training
Below is an example to train Show-o2 with 1.5B LLM parameters on one node with 8 GPUs. Please refer to `accelerate` for multi-node distributed training.

(As the code is manually cleaned based on our original codes without other inspection. Feel free to contact me if you encounter any issues)
### Data Preparation
We did use our `internal packages` to load large-scale data shards. For convenience, we here simply implement a dataset classes (`./datasets`) based on torch `Dataset`. We recommend other packages like `webdataset` when loading large-scale datasets.

### Stage-1
Prepare a `.jsonl` annotation file for your image-text pairs in the format as follows and change the `path` in `./configs`:
```
{
    "image_path": "path/to/your/image1",
    "prompt": "a description of the image1"
}
{
    "image_path": "path/to/your/image2",
    "prompt": "a description of the image2"
}
```

```
bash train_showo2_1.5b_stage1.sh
```


### Stage-2
Follow [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train#about-the-llava-onevision-data) and [DenseFusion](https://github.com/baaivision/DenseFusion) to download the required images. Download the annotation files [here](https://huggingface.co/datasets/Sierkinhane/show-o2-data-annotations) and change the `image dir` and `annotation path` in `./configs`.
``` 
bash train_showo2_1.5b_stage2.sh
```

### Add additional high-quality image generation, interleaved image-text, or video data
In our experiments, we add additional these kinds of additional data in stage-1 to enhance the base show-o2 models with more comprehensive capabilities. We will provide the scripts and code soon.

### Downstream fine-tuning
More comprehensive training codes on interleaved image-text pairs will be provided soon. Here, we simply take the mixed-modality generation on [visual storytelling](https://visionandlanguage.net/VIST/index.html) dataset as an example. 

Following the instructions to download the dataset visual storytelling data [here](https://visionandlanguage.net/VIST/dataset.html) and our processed annotation [here](https://huggingface.co/datasets/Sierkinhane/show-o2-data-annotations/blob/main/vist_train_annotations.json).

We use this config `conigs/showo2_1.5b_downstream_mixed_modality_simple.yaml` and set `frozen_params` as follows for the warm-up training:
``` 
frozen_params: ['image_embedder_und', 'und_trans', 'showo', 'position_embedding']
```
**Training script**:
``` 
accelerate launch --config_file ../accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=9999 train_mixed_modality_simple.py config=configs/showo2_1.5b_downstream_mixed_modality_simple.yaml
```
As the above config did not train the base LLM parameters, the model still cannot generate text in the style of the visual storytelling data. Next, we can set all the model parameters trainable and modified the `max_train_steps` as follows and continue the training:
``` 
frozen_params: null
max_train_steps: 50000  # adjust it according to the performance
```
**Training script**:
``` 
accelerate launch --config_file ../accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=9999 train_mixed_modality_simple.py config=configs/showo2_1.5b_downstream_mixed_modality_simple.yaml
```
**Mixed-modality Inference**. The model will automatically generate interleaved texts and images.
``` 
CUDA_VISIBLE_DEVICES=0 python3 inference_mixed_modality.py config=configs/showo2_1.5b_demo_432x432_mixed_modal.yaml \
                         model_path=./show-o2-qwen2-5-1.5b-downstream-mixed-modality-432x432/checkpoint-50000/unwrapped_model/pytorch_model.bin \
                         batch_size=4 guidance_scale=5.0 num_inference_steps=50;
```

### Citation
To cite the paper and model, please use the below:
```
@article{xie2025showo2,
  title={Show-o2: Improved Native Unified Multimodal Models},
  author={Xie, Jinheng and Yang, Zhenheng and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2506.15564},
  year={2025}
}

@article{xie2024showo,
  title={Show-o: One Single Transformer to Unify Multimodal Understanding and Generation},
  author={Xie, Jinheng and Mao, Weijia and Bai, Zechen and Zhang, David Junhao and Wang, Weihao and Lin, Kevin Qinghong and Gu, Yuchao and Chen, Zhijie and Yang, Zhenheng and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2408.12528},
  year={2024}
}
```
### Acknowledgments
This work is heavily based on [Show-o](https://github.com/showlab/show-o).
