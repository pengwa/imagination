

Commit: b7f0e87dc2143cdf9cbc050e34ba80440074c53b
Repo: git clone https://github.com/openai/triton.git

Installation From Source:

    # https://triton-lang.org/getting-started/installation.html
    cd triton/python;
    pip install cmake; # build time dependency
    pip install -e .

You will hit error message like this:

    root@351819daeedc:/nfs/pengwa/dev/triton/python# pip install -e .
    Obtaining file:///nfs/pengwa/dev/triton/python
    Requirement already satisfied: torch in /opt/conda/lib/python3.7/site-packages (from triton==1.1.2) (1.11.0a0+git97f29bd)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.7/site-packages (from triton==1.1.2) (3.3.2)
    Requirement already satisfied: typing_extensions in /opt/conda/lib/python3.7/site-packages (from torch->triton==1.1.2) (3.10.0.2)
    Installing collected packages: triton
      Running setup.py develop for triton
        ERROR: Complete output from command /opt/conda/bin/python -c 'import setuptools, tokenize;__file__='"'"'/nfs/pengwa/dev/triton/python/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' develop --no-deps:
        ERROR: running develop
        running egg_info
        writing triton.egg-info/PKG-INFO
        writing dependency_links to triton.egg-info/dependency_links.txt
        writing requirements to triton.egg-info/requires.txt
        writing top-level names to triton.egg-info/top_level.txt
        /opt/conda/lib/python3.7/site-packages/setuptools/dist.py:720: UserWarning: Usage of dash-separated 'description-file' will not be supported in future versions. Please use the underscore name 'description_file' instead
          % (opt, underscore_opt)
        /opt/conda/lib/python3.7/site-packages/setuptools/command/easy_install.py:159: EasyInstallDeprecationWarning: easy_install command is deprecated. Use build and pip and other standards-based tools.
          EasyInstallDeprecationWarning,
        /opt/conda/lib/python3.7/site-packages/setuptools/command/install.py:37: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
          setuptools.SetuptoolsDeprecationWarning,
        package init file 'triton/_C/__init__.py' not found (or not a regular file)
        reading manifest file 'triton.egg-info/SOURCES.txt'
        reading manifest template 'MANIFEST.in'
        writing manifest file 'triton.egg-info/SOURCES.txt'
        running build_ext
        -- Adding Python module
        -- Configuring done
        CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
        Please set them or make sure they are set and tested correctly in the CMake files:
        TERMINFO_LIBRARY
            linked by target "triton" in directory /nfs/pengwa/dev/triton

        -- Generating done
        CMake Generate step failed.  Build files cannot be regenerated correctly.
        Traceback (most recent call last):
          File "<string>", line 1, in <module>
          File "/nfs/pengwa/dev/triton/python/setup.py", line 144, in <module>
            "Programming Language :: Python :: 3.6",
          File "/opt/conda/lib/python3.7/site-packages/setuptools/__init__.py", line 159, in setup
            return distutils.core.setup(**attrs)
          File "/opt/conda/lib/python3.7/distutils/core.py", line 148, in setup
            dist.run_commands()
          File "/opt/conda/lib/python3.7/distutils/dist.py", line 966, in run_commands
            self.run_command(cmd)
          File "/opt/conda/lib/python3.7/distutils/dist.py", line 985, in run_command
            cmd_obj.run()
          File "/opt/conda/lib/python3.7/distutils/dist.py", line 985, in run_command
            cmd_obj.run()
          File "/nfs/pengwa/dev/triton/python/setup.py", line 77, in run
            self.build_extension(ext)
          File "/nfs/pengwa/dev/triton/python/setup.py", line 118, in build_extension
            subprocess.check_call(["cmake", self.base_dir] + cmake_args, cwd=self.build_temp, env=env)
          File "/opt/conda/lib/python3.7/subprocess.py", line 347, in check_call
            raise CalledProcessError(retcode, cmd)
        subprocess.CalledProcessError: Command '['cmake', '/nfs/pengwa/dev/triton', '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/nfs/pengwa/dev/triton/python/triton/_C', '-DBUILD_TUTORIALS=OFF', '-DBUILD_PYTHON_MODULE=ON', '-DLLVM_INCLUDE_DIRS=/tmp/clang+llvm-11.0.1-x86_64-linux-gnu-ubuntu-16.04/include', '-DLLVM_LIBRARY_DIR=/tmp/clang+llvm-11.0.1-x86_64-linux-gnu-ubuntu-16.04/lib', '-DTRITON_LLVM_BUILD_DIR=/tmp/llvm-release', '-DPYTHON_INCLUDE_DIRS=/opt/conda/include/python3.7m;/usr/local/cuda/include', '-DCMAKE_BUILD_TYPE=Release']' returned non-zero exit status 1.
        ----------------------------------------
    ERROR: Command "/opt/conda/bin/python -c 'import setuptools, tokenize;__file__='"'"'/nfs/pengwa/dev/triton/python/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' develop --no-deps" failed with error code 1 in /nfs/pengwa/dev/triton/python/

Resolution:

CMAKE failed to find the package tinfo

    ./CMakeLists.txt:find_library(TERMINFO_LIBRARY tinfo)
    
Install libtinfo-dev by entering the following commands in the terminal:

    sudo apt update
    sudo apt install libtinfo-dev
    
Issue should be fixed.
