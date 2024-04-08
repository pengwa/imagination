
git config --global --add safe.directory '*'
chmod 777 . -R
export CUDA_HOME=/usr/local/cuda-12.1
export CUDNN_HOME=/usr/local/cuda-12.1
export CUDACXX=$CUDA_HOME/bin/nvcc
dir_of_ort=`pwd`
export PATH=/opt/openmpi-4.0.4/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi-4.0.4/lib:$LD_LIBRARY_PATH
export MPI_CXX_INCLUDE_DIRS=/opt/openmpi-4.0.4/include

wheel_path=`pip show onnxruntime-training | grep -i location | cut  -d" " -f2`

pip uninstall onnxruntime_training onnxruntime_training_gpu --yes
#rm -rf $wheel_path/onnxruntime
flavor=Debug
flavor=RelWithDebInfo



rm -rf $dir_of_ort/build/Linux/$flavor/dist/*.whl


./build.sh --config $flavor --use_cuda --build_wheel --parallel 8 --enable_training --skip_tests --cuda_version=12.1 --allow_running_as_root --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="60;70;75;80;86"

pip install $dir_of_ort/build/Linux/$flavor/dist/*.whl

cd /tmp/
python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
cd $dir_of_ort
