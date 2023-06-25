  
  ```bash
   sudo env PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH ncu --target-processes all --set full -o pengwa_prof_baseline  ./onnxruntime_test_all --gtest_filter=CudaKernelTest.BiasGeluGradDx_basic
  ```


```
sudo -E  "PATH=$PATH" "LD_LIBRARY_PATH=/bert_ort/pengwa/py38/lib/python3.8/site-packages/onnxruntime/capi/" \
   /usr/local/cuda-11.8/bin/ncu  --export ./ncu_res_%p_%i \
    --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats \
    --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats \
    --force-overwrite --target-processes all --replay-mode kernel \
    --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 \
    --cache-control all --clock-control none --apply-rules no \
    --check-exit-code yes \
    --import-source yes \
    --profile-from-start on \
    --kernel-name-base function \
    --kernel-name regex:"_GatherScatterElementsKernel" \
       torchrun --nproc_per_node=1 examples/onnxruntime/training/language-modeling/run_mlm.py  \
        --model_name_or_path microsoft/deberta-v3-base --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
        --num_train_epochs 10 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --do_train \
        --overwrite_output_dir --output_dir ./outputs/ --seed 1137 --fp16 --report_to none --optim adamw_ort_fused \
        --max_steps 2 --logging_steps 1 --use_module_with_loss

##     --kernel-name-base demangled --kernel-name regex:"FillOutputWithIndexKernel" \

```
