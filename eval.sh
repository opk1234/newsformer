#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python ./run_cls.py \
	--model_name_or_path ./pretrain_output \
	--learning_rate 5e-5 \
	--node_config_name ./bert_base_1layer \
	--train_file ./train_data/PHEME/train.json \
	--do_train \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--dataloader_drop_last true \
	--num_train_epochs 1 \
	--dataloader_num_workers 8 \
	--save_steps 10000 \
	--save_steps 10000 \
	--output_dir ./finetune_output \
	--dataset_script_dir ./data_scripts \
	--dataset_cache_dir ./cache \
	--limit 50000000 \
	--overwrite_output_dir \
	--tokenizer_name ./pretrain_output \
  --fp16 true
