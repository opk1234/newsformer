#!/bin/bash

python ./run_train.py \
	--model_name_or_path bert-base-uncased \
	--learning_rate 5e-5 \
	--node_config_name ./bert_base_1layer \
	--train_file ./train_data/PHEME/train.json \
	--do_train \
	--per_device_train_batch_size 2 \
	--num_train_epochs 1 \
	--dataloader_num_workers 8 \
	--save_steps 10000 \
	--output_dir ./pretrain_output \
	--dataset_script_dir ./data_scripts \
	--dataset_cache_dir ./cache \
	--limit 50000000 \
	--overwrite_output_dir \
	--tokenizer_name ./bert-base-uncased \
  --fp16 true
