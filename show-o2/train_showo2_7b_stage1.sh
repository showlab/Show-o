export OMP_NUM_THREADS=8

# download pre-distilled semantic layers
wget https://huggingface.co/Sierkinhane/pre-distilled_semantic_layers/resolve/main/pre-distilled_semantic_layers.pt

# if you want to load the pre-trained flow_head
# an option is to make sure to put the pre-trained 1.5B show-o2 checkpoint on the directory
# and set params_not_load: ['image_embedder_gen', 'showo', 'fusion_proj', 'diff_proj', 'time_embed_proj'] in configs/showo2_7b_stage_1_a.yaml
mkdir showo2-7b-stage-1-a;
cd ./showo2-7b-stage-1-a;
cp -r ../showo2-1.5b-stage-2-c/checkpoint-final .;
mv ./checkpoint-final ./checkpoint-0/;
cd ../;

# training on a relatively large-scale dataset
accelerate launch --config_file ../accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=9999 train_stage_one.py config=configs/showo2_7b_stage_1_a.yaml;

# change the directory name if you have modified them in your config files
mkdir showo2-7b-stage-1-b;
cd ./showo2-7b-stage-1-b;
cp -r ../showo2-7b-stage-1-a/checkpoint-150000 .;
mv ./checkpoint-150000 ./checkpoint-0/;
cd ../;

# replace the image generation data with the high-quality one
accelerate launch --config_file ../accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=9999 train_stage_one.py config=configs/showo2_7b_stage_1_b.yaml;
