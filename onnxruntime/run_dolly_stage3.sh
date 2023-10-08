#!/bin/bash

ds_config=`mktemp --suffix ".json"`
echo the deepspeed config is put at $ds_config
cat << EOF > $ds_config
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 200000000,
    "contiguous_gradients": false,
    "cpu_offload": false,
    "memory_efficient_linear": true
  },
  "zero_allow_untested_optimizer": true,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "steps_per_print": 2000,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
EOF


num_gpus=8
torchrun --nproc_per_node $num_gpus \
    examples/onnxruntime/training/language-modeling/run_clm.py \
        --model_name_or_path databricks/dolly-v2-3b \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --do_train \
        --output_dir /tmp/test-clm --overwrite_output_dir \
        --fp16 \
        --report_to none \
        --max_steps 100 --logging_steps 1 --use_module_with_loss \
        --deepspeed $ds_config
