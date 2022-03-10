
sudo PATH=$PATH:/bert_ort/pengwa/py36/bin/ perf record -g /bin/bash run_cpu.sh
sudo perf report
