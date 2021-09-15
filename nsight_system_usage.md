Use:

/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile -o 24layers_%p_ort -t cuda,nvtx python ...

Be ntoed: when use deepspeed to run multiple process training, you need remove osrt from -t, otherwise it will hang.
/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile -o 24layers_%p_ort -t cuda,nvtx,osrt python ...

Use NVTX:

https://nvtx.readthedocs.io/en/latest/basic.html


As a decorator:

@nvtx.annotate(message="my_message", color="blue")
def my_func():
    pass
As a context manager:

with nvtx.annotate(message="my_message", color="green"):
    pass
    
rng = nvtx.start_range(message=”my_message”, color=”blue”) # … do something … # 
nvtx.end_range(rng)
 


PYTHONMALLOC=malloc valgrind --tool=memcheck --leak-check=yes --track-origins=yes --log-file="mem2.log" python orttraining/orttraining/test/python/bench.py
  389  cd ..
  390  wget https://sourceware.org/pub/valgrind/valgrind-3.17.0.tar.bz2
  391  bzip2 -d valgrind-3.17.0.tar.bz2
  392  tar -xf valgrind-3.17.0.tar
  393  ls
  394  cd valgrind-3.17.0
  395  ls
  396  ./config
  397  ./configure
  398  make
  399  make install
  400  sudo make
  401  sudo make install
  402  valgrind
  403  cd ..
  404  ls
  405  cd onnxruntime/
  406  PYTHONMALLOC=malloc valgrind --tool=memcheck --leak-check=yes --track-origins=yes --log-file="mem3.log" python orttraining/orttraining/test/python/bench.py
