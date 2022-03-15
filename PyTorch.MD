pengwa@webxt6cb30000AL:~/pytorch$ cat build_in_bash.sh

    export BUILD_SPLIT_CUDA=1
    export CUDA_HOME=/usr/local/cuda
    #export NCCL_HOME=
    #RELEASE=1 BUILD_SHARED_LIBS=1 BUILD_CAFFE2=1 BUILD_CAFFE2_OPS=1  USE_GLOO=1  USE_NCCL=1 USE_NUMPY=1 USE_OBSERVERS=1 USE_OPENMP=1 USE_DISTRIBUTED=1 USE_MPI=1 BUILD_PYTHON=1  USE_MKLDNN=0 USE_CUDA=1 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 python setup.py develop
    RELEASE=1 python setup.py install
    #DEBUG=1 python setup.py install


Launch multiple node:

mpi_entry.sh

        #!/bin/bash
        /opt/conda/bin/python3 -m torch.distributed.launch --nproc_per_node=8 --nnode=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP --master_port=$MASTER_PORT ~/cuda_bpr.py

/job/hostfile

    worker-0 slots=8
    worker-1 slots=8

nsys profile -o multi_gpu_profile_%p -t cuda,nvtx  mpirun -npernode 1 -hostfile /job/hostfile -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x MASTER_IP -x MASTER_PORT bash mpirun_entry.sh
