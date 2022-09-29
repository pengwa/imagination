file=$1
step=$2
execution_step=$3

grep '"ph":"X","pid":${step},' $1 | grep -v '"name":"Summary"' | grep '"tid":${execution_step},' | sort -V > $file_step_${step}_execution_step_${execution_step}.log

#python get_peak.py --path step10_summary_sorted.log

#  {"ph":"X","pid":1,"tid":4595,"ts":-1638,"dur":1638,"name":"Summary","cname":"black","args":{"total_bytes":16384,"used_bytes":16384,"free_bytes":0,"used_percent":1,"free_percent":0,"bytes for pattern":16384}},
