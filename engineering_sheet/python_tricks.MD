Print callstack https://stackoverflow.com/questions/1156023/print-current-call-stack-from-a-method-in-python-code
    
    import traceback

    def f():
        g()

    def g():
        for line in traceback.format_stack():
            print(line.strip())

    f()

    # Prints:
    # File "so-stack.py", line 10, in <module>
    #     f()
    # File "so-stack.py", line 4, in f
    #     g()
    # File "so-stack.py", line 7, in g
    #     for line in traceback.format_stack():
If you really only want to print the stack to stderr, you can use:

        traceback.print_stack()
    
Or to print to stdout (useful if want to keep redirected output together), use:

        traceback.print_stack(file=sys.stdout)


Inspect Callstack:

    sudo env "PATH=$PATH" py-spy dump --pid 34754



Ignore Local Python Site-Package:
    
    user_name@51b86f944ce14a90b1c3aa0798c41135-master-0:pip show python_lib_name
    WARNING: Ignoring invalid distribution -pex (/home/user_name/.local/lib/python3.8/site-packages)
    WARNING: Ignoring invalid distribution -nnxruntime-training (/home/user_name/.local/lib/python3.8/site-packages)
    WARNING: Ignoring invalid distribution - (/home/user_name/.local/lib/python3.8/site-packages)
    Name: python_lib_name
    Version: 1.0.0
    Summary: python_lib_name library
    Home-page: http://aka.ms/python_lib_name
    Author: python_lib_name Team
    Author-email: python_lib_name@microsoft.com
    License: UNKNOWN
    Location: /home/user_name/.local/lib/python3.8/site-packages
    Requires:
    Required-by:
    user_name@51b86f944ce14a90b1c3aa0798c41135-master-0:/work/code$ export PYTHONPATH=/opt/conda/lib
    lib/     libexec/
    user_name@51b86f944ce14a90b1c3aa0798c41135-master-0:/work/code$ export PYTHONPATH=/opt/conda/lib
    lib/     libexec/
    user_name@51b86f944ce14a90b1c3aa0798c41135-master-0:/work/code$ expor^Cport PYTHONPATH=/opt/conda/lib
    user_name@51b86f944ce14a90b1c3aa0798c41135-master-0:/work/code$ export PYTHONNOUSERSITE=True
    user_name@51b86f944ce14a90b1c3aa0798c41135-master-0:/work/code$ pip show python_lib_name
    Name: python_lib_name
    Version: 1.0.0
    Summary: python_lib_name library
    Home-page: http://aka.ms/python_lib_name
    Author: python_lib_name Team
    Author-email: python_lib_name@microsoft.com
    License: UNKNOWN
    Location: /opt/conda/lib/python3.8/site-packages
    Requires:
    Required-by:
    user_name@51b86f944ce14a90b1c3aa0798c41135-master-0:/work/code$



The FunctionType constructor is used to make a deep copy of a function. (http://stackoverflow.com/a/6528148/190597)

    import types
    def copy_func(f, name=None):
        return types.FunctionType(f.func_code, f.func_globals, name or f.func_name,
            f.func_defaults, f.func_closure)
    
    def A():
        """A"""
        pass
    B = copy_func(A, "B")
    B.__doc__ = """B"""
