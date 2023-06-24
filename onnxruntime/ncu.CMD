sudo -E  "PATH=$PATH" "LD_LIBRARY_PATH=...." /opt/nvidia/nsight-compute/2022.1.1/ncu --export ./nsys_res_%p_%i \
    --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats \
    --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats \
    --force-overwrite --target-processes all --replay-mode kernel \
    --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 \
    --cache-control all --clock-control none --apply-rules no \
    --check-exit-code yes \
    --import-source yes \
    --profile-from-start off \ 
    --kernel-name-base demangled --kernel-name regex:"name"
        python toy.py # the script
