from onnxruntime.training.ortmodule.torch_cpp_extensions import fused_ops

import torch
num_work = 2 #8
batch_size = 3 #204800
vocab_end_index = 8 #7000000

num_work = 8
batch_size = 6 * 204800
vocab_end_index = 34820728

output_half = True

num_user_embeddings_per_partition = vocab_end_index // num_work
embedding_dim = 128
# local_vocab_embedding = torch.arange(num_user_embeddings_per_partition * embedding_dim * 1, dtype=torch.float).view(num_user_embeddings_per_partition, embedding_dim)
# local_vocab_embedding = local_vocab_embedding.cuda()
local_vocab_embedding = torch.rand(num_user_embeddings_per_partition, embedding_dim, device="cuda:0")

# global_idx = torch.tensor(num_work * batch_size * [3], dtype=torch.int64).cuda()


local_rank = 1
local_vocab_start_index = local_rank * num_user_embeddings_per_partition
local_vocab_end_index = local_vocab_start_index + num_user_embeddings_per_partition - 1
global_idx = torch.randint(0, vocab_end_index, (num_work * batch_size,), dtype=torch.int64 , device="cuda:0")

global_out_embedding = torch.empty(num_work * batch_size * embedding_dim, dtype=torch.float).cuda()
if output_half:
    global_out_embedding = global_out_embedding.half()

torch.cuda.synchronize()
import time
start = time.time()
fused_ops.distributed_embedding_lookup(global_idx, local_vocab_start_index, local_vocab_end_index,
                                       local_vocab_embedding,
                                       global_out_embedding)

torch.cuda.synchronize()
print(global_out_embedding.shape, global_out_embedding, time.time()- start)

torch.cuda.synchronize()
input_mask = (global_idx < local_vocab_start_index) | \
            (global_idx >= local_vocab_end_index)
global_idx[input_mask] = 0
torch.cuda.synchronize()
start = time.time()
global_out_embedding_new = local_vocab_embedding[global_idx - local_vocab_start_index]
if output_half:
    global_out_embedding_new = global_out_embedding_new.half()
torch.cuda.synchronize()
end = time.time()
global_out_embedding_new[input_mask, :] = 0.0
torch.cuda.synchronize()
print(global_out_embedding_new.view(-1).shape, global_out_embedding_new.view(-1), end - start)

print("is_equal:", torch.all(global_out_embedding_new.view(-1).eq(global_out_embedding)))#
print("torch.count_nonzero: ", torch.count_nonzero(global_out_embedding))
# at::Tensor& global_idx,
#                                   const int local_vocab_start_index,
#                                   const int local_vocab_end_index,
#                                   const at::Tensor& local_vocab_embedding,
#                                   at::Tensor& global_out_embedding,
#                                   at::Tensor& valid_global_indices);
