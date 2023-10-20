wheel_path=`pip show onnxruntime-training | grep -i location | cut  -d" " -f2`

#rm $wheel_path/onnxruntime/training/ortmodule/torch_cpp_extensions -rf
cp orttraining/orttraining/python/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils/torch_interop_utils.cc $wheel_path/onnxruntime/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils/torch_interop_utils.cc -rf

cur=`pwd`
cd /tmp/
python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
cd $cur
