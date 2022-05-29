  
    sudo apt-get install google-perftools
    pip install yep
  

https://v0dro.in/blog/2020/04/21/profiling-and-benchmarking-python-programs/ 

    import yep

    yep.start("file_name.prof")
    # do something
    yep.stop()
    
    
    
Install GO

    curl -OL https://golang.org/dl/go1.16.7.linux-amd64.tar.gz

    tar -xvf go1.16.7.linux-amd64.tar.gz

    export PATH=$PATH:<PATH TO GO>/go/bin


    go get -u github.com/google/pprof
    
    
    pprof -text -lines file_name.prof
