
dir_of_ort=/bert_ort/pengwa/on_device
flavor=RelWithDebInfo
echo -e "update ort package"
build_path=$dir_of_ort/build/Linux/$flavor
cd $build_path
make -j40 onnxruntime_pybind11_state
wheel_path=`pip show onnxruntime-training | grep -i location | cut  -d" " -f2`
cd $wheel_path/onnxruntime/capi
so_files=`ls *.so`
sudo rm -rf $so_files
for i in $so_files

do
    sudo ln -s  $build_path/$i $i
done

echo -e "\033[0;31m update ORT by softlink done \033[0m"
