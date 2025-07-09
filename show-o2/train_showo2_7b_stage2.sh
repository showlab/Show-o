export OMP_NUM_THREADS=8

# we follow the training schedule of LLaVA-OneVision, which includes three sub-stages composing of different instructional tuning data.
accelerate launch --config_file ../show-o/accelerate_configs/multi_nodes_6_deepspeed_zero2/8_gpus_node_0.yaml --main_process_port=8888 train_stage_two.py config=configs/showo2_7b_stage_2_a.yaml;

# change the directory name if you have modified them in your config files
mkdir showo2-7b-stage-2-b;
cd ./showo2-7b-stage-2-b;
cp -r ../showo2-7b-stage-2-a/checkpoint-final .;
mv ./checkpoint-final ./checkpoint-0/;
cd ../;
accelerate launch --config_file ../show-o/accelerate_configs/multi_nodes_6_deepspeed_zero2/8_gpus_node_0.yaml --main_process_port=8888 train_stage_two.py config=configs/showo2_7b_stage_2_b.yaml;

# change the directory name if you have modified them in your config files
mkdir showo2-7b-stage-2-c;
cd ./showo2-7b-stage-2-c;
cp -r ../showo2-7b-stage-2-b/checkpoint-final .;
mv ./checkpoint-final ./checkpoint-0/;
cd ../;
accelerate launch --config_file ../show-o/accelerate_configs/multi_nodes_6_deepspeed_zero2/8_gpus_node_0.yaml --main_process_port=8888 train_stage_two.py config=configs/showo2_7b_stage_2_c.yaml;
