  
  
   sudo env PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH ncu --target-processes all --set full -o pengwa_prof_baseline  ./onnxruntime_test_all --gtest_filter=CudaKernelTest.BiasGeluGradDx_basic
