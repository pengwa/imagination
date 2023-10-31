wheel_path=`pip show onnxruntime-training | grep -i location | cut  -d" " -f2`

export CUDA_HOME=/usr/local/cuda-11.8
export CUDNN_HOME=/usr/local/cuda-11.8
export CUDACXX=$CUDA_HOME/bin/nvcc

export PATH=/opt/openmpi-4.0.4/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi-4.0.4/lib:$LD_LIBRARY_PATH
export MPI_CXX_INCLUDE_DIRS=/opt/openmpi-4.0.4/include

#rm $wheel_path/onnxruntime/training/ortmodule/torch_cpp_extensions -rf
rm $wheel_path/onnxruntime/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils/*.h
rm $wheel_path/onnxruntime/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils/*.cc
rm $wheel_path/onnxruntime/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils/*.py
cp orttraining/orttraining/python/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils/* $wheel_path/onnxruntime/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils/ -rf

cur=`pwd`
cd /tmp/
python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
cd $cur
