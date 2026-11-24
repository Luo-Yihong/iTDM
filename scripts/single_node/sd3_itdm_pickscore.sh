accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29502 scripts/train_sd3_itdm.py --config config/itdm.py:pickscore_sd3_8gpu
