

## NSight System Usage: ##

    /opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile -o 24layers_%p_ort -t cuda,nvtx python ...

Be ntoed: when use deepspeed to run multiple process training, you need remove osrt from -t, otherwise it will hang for example using this command:

    /opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile -o 24layers_%p_ort -t cuda,nvtx,osrt python ...
    
Other sample scripts from NSightSystem web site - https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-single-command-lines

    Typical case: profile a Python script that uses CUDA

        nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --delay=60 python my_dnn_script.py

    Effect: Launch a Python script and start profiling it 60 seconds after the launch, tracing CUDA, cuDNN, cuBLAS, OS runtime APIs, and NVTX as well as collecting thread
    schedule information.

## Use NVTX: ##

https://nvtx.readthedocs.io/en/latest/basic.html

As a decorator:

    @nvtx.annotate(message="my_message", color="blue")
    def my_func():
        pass
        
As a context manager:

    with nvtx.annotate(message="my_message", color="green"):
        pass

Start/end range:

    rng = nvtx.start_range(message="my_message", color="blue") # … do something … # 
    nvtx.end_range(rng)
 

Decorate all functions of a class:

    import inspect
    import nvtx

    def decorator_for_func(orig_func):
        def decorator(*args, **kwargs):
            with nvtx.annotate(message=orig_func.__name__, color="red"):
                # print("Decorating wrapper called for method %s" % orig_func.__name__)
                result = orig_func(*args, **kwargs)
                return result
        return decorator

    def decorator_for_class(cls):
        for name, method in inspect.getmembers(cls):
            if (not inspect.ismethod(method) and not inspect.isfunction(method)) or inspect.isbuiltin(method):
                continue
            # print("Decorating function %s" % name)
            setattr(cls, name, decorator_for_func(method))
        return cls

    # @decorator_for_class
    # class decorated_class:
    #      def method1(self, arg, **kwargs):
    #          print("Method 1 called with arg %s" % arg)
    #      def method2(self, arg):
    #          print("Method 2 called with arg %s" % arg)


    # d=decorated_class()
    # d.method1(1, a=10)
    # d.method2(2)
