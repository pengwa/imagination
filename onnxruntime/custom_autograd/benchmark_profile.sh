#!/bin/bash 

p="/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile"
options="--force-overwrite=true -t cuda,nvtx,osrt python orttraining/orttraining/test/python/bench.py"


batches_str="512," #",1024,2048"
IFS=', ' read -r -a batches <<< "${batches_str}"

hiddens_str="2048,4096,8192,16384"
IFS=', ' read -r -a hiddens <<< "$hiddens_str"

layers_str="6,12"
IFS=', ' read -r -a layers <<< "$layers_str"

echo "start the loop"

for b in "${batches[@]}"
do
  for h in "${hiddens[@]}"
  do 
    for l in "${layers[@]}"
    do 
      name=pt_b${b}_h${h}_l${l}
      echo "handling $name"
      $p -o $name ${options} --batch ${b} --hidden ${h} --layer ${l} --tag $name 2>&1 > $name.txt
      ort_name=ort_b${b}_h${h}_l${l}
      echo "handling $ort_name"
      $p -o $ort_name ${options} --batch ${b} --hidden ${h} --layer ${l} --ort --tag $ort_name 2>&1 > $ort_name.txt
    done
  done
done

