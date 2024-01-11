unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG

pkill python
pkill python


export EXP_ID=`date +%Y-%m-%d_%H-%M-%S`

mkdir -p $EXP_ID

echo "EXP_ID: $EXP_ID, all logs, ONNX models and diff data are saved there."

export ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER='0'

python -m torch.distributed.launch --nproc_per_node=8 \
 --use-env examples/onnxruntime/training/text-classification/run_glue.py \
 --model_name_or_path microsoft/deberta-xlarge --task_name MRPC --max_seq_length 128 \
 --learning_rate 3e-6 --do_train --output_dir /dev/shm --overwrite_output_dir --max_steps 20 \
  --logging_steps 1 --per_device_train_batch_size 4  --deepspeed aml_ds_config_zero_1.config > $EXP_ID/ortmodule_enable_embedding_sparse_optimizer_0.log 2>&1


export ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER='1'

python -m torch.distributed.launch --nproc_per_node=8 \
 --use-env examples/onnxruntime/training/text-classification/run_glue.py \
 --model_name_or_path microsoft/deberta-xlarge --task_name MRPC --max_seq_length 128 \
 --learning_rate 3e-6 --do_train --output_dir /dev/shm --overwrite_output_dir --max_steps 20 \
  --logging_steps 1 --per_device_train_batch_size 4  \
  --deepspeed aml_ds_config_zero_1.config > $EXP_ID/ortmodule_enable_embedding_sparse_optimizer_1.log 2>&1


cd $EXP_ID
python -m onnxruntime.training.utils.hooks.merge_activation_summary --pt_dir no_emb_sparsity_run_0 --ort_dir use_emb_sparsity_run_0 --output_dir diff_rank0


echo "Done. Now you can check the result NOW!"

cd /tmp/onnxruntime
nohup python spin.py &
