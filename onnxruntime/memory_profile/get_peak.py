import re
import argparse
import pprint
import numpy as np
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)

# {"ph":"X","pid":10,"tid":15117,"ts":-556704358,"dur":556704358,"name":"Summary","cname":"black","args":{"total_bytes":363857920,"used_bytes":363857920,"free_bytes":0,"used_percent":1,"free_percent":0,"bytes for pattern":363857920}},

regexp="[\s\S]*{\"ph\":\"X\",\"pid\":([0-9]+),\"tid\":([0-9]+),\"ts\":[-]?[0-9]+,\"dur\":([0-9]+),\"name\":\"Summary\",\"cname\":\"black\",\"args\":{\"total_bytes\":([0-9]+),\"used_bytes\":([0-9]+),\"free_bytes\":([0-9]+),\"used_percent\":([0-9]+),\"free_percent\":([0-9]+),\"bytes for pattern\":([0-9]+)}},[\s\S]*"
#regexp="output([0-9]+).txt:([0-9]+.[0-9]+[m]?s)  ([0-9]+.[0-9]+[m]?s)             \([0-9]+ [0-9]+ [0-9]+\)       \([0-9]+ [0-9]+ [0-9]+\)        [0-9]+       [0-9]+B        [0-9]+B         -           -           -           -  Tesla V100-SXM3         [0-9]+[\s]*[0-9]+  ncclAllReduceRingLLKernel_sum_f16\(ncclColl\) \[[0-9]+\]\n"
# output67482.txt:495.013s  173.64ms             (16 1 1)       (257 1 1)        96       64B        0B         -           -           -           -  Tesla V100-SXM3         1         7  ncclAllReduceRingLLKernel_sum_f16(ncclColl) [862541]
#output66056.txt:253.257s  209.84ms             (16 1 1)       (257 1 1)        96       64B        0B         -           -           -           -  Tesla V100-SXM3         1        29  ncclAllReduceRingLLKernel_sum_f16(ncclColl) [15802638]\n

args = parser.parse_args()
fileName=args.path

process_dict = collections.OrderedDict()
with open(fileName) as f:
  for line in f:
    match = re.match(regexp, line)
    if match:
      t_id=int(match.group(2))
      total_bytes = int(match.group(4))
      process_dict[t_id] = total_bytes
    else:
      print("warning: the line is not parsed correctly:", line)

sorted_tuples = sorted(
    process_dict.items(), key=lambda item: item[1], reverse=True
)

print(f"## The sorted memory consumption for each execution step:")
print(f"(execution_step, total_bytes)")
import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(sorted_tuples)
