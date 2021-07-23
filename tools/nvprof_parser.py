import re
import argparse
import pprint
import numpy as np
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)

regexp="output([0-9]+).txt:([0-9]+.[0-9]+[m]?s)  ([0-9]+.[0-9]+[m]?s)             \([0-9]+ [0-9]+ [0-9]+\)       \([0-9]+ [0-9]+ [0-9]+\)        [0-9]+       [0-9]+B        [0-9]+B         -           -           -           -  Tesla V100-SXM3         [0-9]+[\s]*[0-9]+  ncclAllReduceRingLLKernel_sum_f16\(ncclColl\) \[[0-9]+\]\n"
# output67482.txt:495.013s  173.64ms             (16 1 1)       (257 1 1)        96       64B        0B         -           -           -           -  Tesla V100-SXM3         1         7  ncclAllReduceRingLLKernel_sum_f16(ncclColl) [862541]
#output66056.txt:253.257s  209.84ms             (16 1 1)       (257 1 1)        96       64B        0B         -           -           -           -  Tesla V100-SXM3         1        29  ncclAllReduceRingLLKernel_sum_f16(ncclColl) [15802638]\n
class KernelDesp:
  def __init__(self, start, duration):
    self.start=start
    self.duration=duration
    self.last_nccl_done=None


def parse_time(t_str):
  t = None
  if t_str.endswith("ms"):
    t=float(t_str.split("ms")[0])
  elif t_str.endswith("s"):
    #print(t_str.split("s")[0])
    t=float(t_str.split("s")[0]) * 1000
  else:
    raise ValueError("unexpected start time postfix")
  return t

args = parser.parse_args()
fileName=args.path

process_dict = collections.OrderedDict()
with open(fileName) as f:
  for line in f:
    match = re.match(regexp, line)
    if match:
      process_id=str(match.group(1))
      start_str = str(match.group(2))
      start = parse_time(start_str)
      duration_str=str(match.group(3))
      duration = parse_time(duration_str)
      if process_id not in process_dict:
        process_dict[process_id]=[]
      process_dict[process_id].append(KernelDesp(start, duration))
    else:
      print("warning: the line is not parsed correctly ", line)

n = None
for k in process_dict:
  if n is None:
    n = len(process_dict[k])
  if len(process_dict[k]) != n :
      print("unexpected count of record for process ", k, ", process count: ", len(process_dict[k]))
  #print(str(k) + '\t')

i = 1
while i < n - 1:
  #print('process id', 'allreduce for iteration ' + str(i), 'duration between allreduce end' + str(i - 1) + "~" + str(i), 'computation in iteration ' + str(i))
  computation_time = []
  allreduce_time = []
  for k in process_dict:
    kernel_desc = process_dict[k][i]
    last_allreduce_end = process_dict[k][i - 1].start + process_dict[k][i - 1].duration
    duration_between_allreduce_ends = process_dict[k][i].start + process_dict[k][i].duration - last_allreduce_end
    current_iteration_computation_time = process_dict[k][i].start - last_allreduce_end

    duration_between_allreduce_starts = process_dict[k][i].start - process_dict[k][i - 1].start 
    print(i, k, duration_between_allreduce_ends, process_dict[k][i].duration, current_iteration_computation_time, duration_between_allreduce_starts)
    computation_time.append(current_iteration_computation_time)
    allreduce_time.append(process_dict[k][i].duration)
  #ind = np.argmax(allreduce_time)
  #print('max allreduce_time: ', computation_time[ind], allreduce_time[ind], process_dict.keys()[ind])
  #ind = np.argmin(allreduce_time)
  #print('min allreduce_time: ', computation_time[ind], allreduce_time[ind], process_dict.keys()[ind])
  i += 1

print(len(process_dict))
