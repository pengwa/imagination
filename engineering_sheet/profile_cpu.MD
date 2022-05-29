
    sudo PATH=$PATH:/bert_ort/pengwa/py36/bin/ perf record -g /bin/bash run_cpu.sh
    sudo perf report


    perf record --call-graph fp
    perf report --call-graph --stdio -G
    https://stackoverflow.com/questions/12160449/call-stack-in-the-perf-profiler
