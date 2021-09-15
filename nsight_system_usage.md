

NSight System Usage:

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

Start/end range:

rng = nvtx.start_range(message=”my_message”, color=”blue”) # … do something … # 
nvtx.end_range(rng)
 
