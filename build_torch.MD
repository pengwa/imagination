pengwa@webxt6cb30000AL:~/pytorch$ cat build_in_bash.sh

    export BUILD_SPLIT_CUDA=1
    export CUDA_HOME=/usr/local/cuda
    #export NCCL_HOME=
    #RELEASE=1 BUILD_SHARED_LIBS=1 BUILD_CAFFE2=1 BUILD_CAFFE2_OPS=1  USE_GLOO=1  USE_NCCL=1 USE_NUMPY=1 USE_OBSERVERS=1 USE_OPENMP=1 USE_DISTRIBUTED=1 USE_MPI=1 BUILD_PYTHON=1  USE_MKLDNN=0 USE_CUDA=1 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 python setup.py develop
    RELEASE=1 python setup.py install
    #DEBUG=1 python setup.py install
