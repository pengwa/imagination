import re
import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)

#regexp="[0-9a-zA-Z_-\/]+run_res_accu-[0-9]+_maxseq-([0-9]+)_([0-9a-zA-Z_-]+)_(fp[0-9]+)_g([0-9]+)_b([0-9]+):Throughput: ([0-9]+.[0-9]+) Examples \/ Second"
regexp="[0-9a-zA-Z_\-\/]+run_res_accu-[0-9]+_maxseq-([0-9]+)_local-[0-9]+_world-([0-9]+)_[0-9a-zA-Z_-]+_(fp[0-9]+)_g([0-9]+)_b([0-9]+)[-:]Round [0-9]+, Step: ([0-9]+), epoch: [0-9]+, batch: [0-9]+\/[0-9]+, shard_iteration: [0-9]+\/[0-9]+, time: ([0-9]+.*[0-9]*) ms, throughput: ([0-9]+.*[0-9]*) ex\/sec"

args = parser.parse_args()
fileName=args.path

sums = {}
freqs = {}
batch_map={}
with open(fileName) as f:
  for line in f:
    match = re.match(regexp, line)
    if match:
      seq_length=str(match.group(1))
      if seq_length == "128":
        phase = 1
      else:
        assert seq_length == "512"
        phase = 2
      world_rank = match.group(2)
      key=match.group(3) + "_g" + str(match.group(4)).zfill(2) + "_phase" + str(phase) + "_b" + str(match.group(5)).zfill(3) + "_rank"+ str(world_rank).zfill(3)
      latency = match.group(7)
      if key not in sums:
        sums[key] = 0.0
        freqs[key] = 0
        batch_map[key]=match.group(5)

      sums[key] = sums[key] + float(latency) 
      freqs[key] += 1
    else:
      raise Exception("warning: the line is not parsed correctly ", line)

aggregated_throughput={}
rank_count={}
print("('fptype_gpu_phase_batch', throughput , collected_run_cnt, aggregated_key)")
for k in sorted(sums):
  if freqs[k] != 128:
    print(k, " frequencey is not 128, it is not correct!!")

  averaged_latency = float(sums[k]) / float(freqs[k]) / float(1000)
  throughput = float(batch_map[k]) / float(averaged_latency)

  aggregated_key = "_".join(k.split("_")[:-1])
  print(k, throughput, freqs[k], aggregated_key)

  if aggregated_key not in aggregated_throughput:
    aggregated_throughput[aggregated_key] = 0.0
    rank_count[aggregated_key] = 0

  aggregated_throughput[aggregated_key] = aggregated_throughput[aggregated_key] + throughput
  rank_count[aggregated_key] += 1


print("('fptype_gpu_phase_batch', aggregated_throughput , rank_count)")
for k in sorted(aggregated_throughput):
  print(k, aggregated_throughput[k], rank_count[k])
