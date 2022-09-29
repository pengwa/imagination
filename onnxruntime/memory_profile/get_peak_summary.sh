
file=$1
step=$2

grep '"name":"Summary"' $file > /tmp/$file"_summary.log"

grep '"ph":"X","pid":${step},' /tmp/$file"_summary.log" > /tmp/$file"_summary_step_${step}.log"

cat /tmp/$file"_summary_step_${step}.log" | sort -V > /tmp/$file"_summary_step_${step}_sorted.log"

python get_peak.py --path /tmp/$file"_summary_step_${step}_sorted.log"

#  {"ph":"X","pid":1,"tid":4595,"ts":-1638,"dur":1638,"name":"Summary","cname":"black","args":{"total_bytes":16384,"used_bytes":16384,"free_bytes":0,"used_percent":1,"free_percent":0,"bytes for pattern":16384}},
