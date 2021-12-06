export CUDA_HOME=/usr/local/cuda-11.1
export CUDNN_HOME=/usr/local/cuda-11.1
export CUDACXX=$CUDA_HOME/bin/nvcc

export PATH=/opt/openmpi-4.0.4/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi-4.0.4/lib:$LD_LIBRARY_PATH
export MPI_CXX_INCLUDE_DIRS=/opt/openmpi-4.0.4/include

pip uninstall onnxruntime_training onnxruntime_training_gpu --yes
flavor=Debug
flavor=RelWithDebInfo

rm -rf /home/pengwa/dev/ort_dev/build/Linux/$flavor/dist/*.whl

# don't give --build_shared_lib
./build.sh --config $flavor --use_cuda --enable_training  --build_wheel --parallel 8 --use_mpi --skip_tests --enable_training_torch_interop --cuda_version=11.1

pip install /home/pengwa/dev/ort_dev/build/Linux/$flavor/dist/onnxruntime_training-1.8.2+torch1100a0.cu111-cp37-cp37m-linux_x86_64.whl

cd /tmp/
python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
cd /home/pengwa/dev/onnxruntime
