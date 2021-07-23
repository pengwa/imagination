#!/bin/bash 


rm -rf throughput.txt
p=$1
FILES=$p/*
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  if [[ $f == *"_fp32_"* ]]; then
    grep -H  "throughput: " $f | tail -n 128 >> throughput.txt
  else
    grep -H -A 2 "is_all_finite 1" $f | grep "throughput: " | tail -n 128 >> throughput.txt
  fi
done
