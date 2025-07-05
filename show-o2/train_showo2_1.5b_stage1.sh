export OMP_NUM_THREADS=8

# download pre-distilled semantic layers
wget https://huggingface.co/Sierkinhane/pre-distilled_semantic_layers/resolve/main/pre-distilled_semantic_layers.pt

# training with a relatively large-scale dataset
accelerate launch --config_file ../accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=9999 train_stage_one.py config=configs/showo2_1.5b_stage_1_a.yaml;

# change the directory name if you have modified them in your config files
mkdir show-o2-1.5b-stage-1-b;
cd ./show-o2-1.5b-stage-1-b;
cp -r ../show-o2-1.5b-stage-1-a/checkpoint-150000 .;
mv ./checkpoint-150000 ./checkpoint-0/;
cd ../;

# replace the image generation data with the high-quality one
accelerate launch --config_file ../accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=9999 train_stage_one.py config=configs/showo2_1.5b_stage_1_b.yaml;
