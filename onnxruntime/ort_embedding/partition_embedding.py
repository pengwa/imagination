import timeit
from turtle import pos
import torch
import random
import time
# import torch.optim as optim
import torch.optim._multi_tensor as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
import nvtx
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import nvtx




def get_data_parallel_rank():
    return torch.distributed.get_rank()

def get_data_parallel_world_size():
    return torch.distributed.get_world_size()

def get_global_index_range(per_partition_vocab_size, rank):
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l

def get_local_index_range(global_vocab_size, rank, world_size):
    assert global_vocab_size % world_size == 0, '{} is not divisible by {}'.format(
        global_vocab_size, world_size)
    per_partition_vocab_size = global_vocab_size // world_size
    return get_global_index_range(per_partition_vocab_size, rank)

def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    # set_tensor_model_parallel_attributes(tensor=weight,
    #                                      is_parallel=True,
    #                                      dim=partition_dim,
    #                                      stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=torch.float)

    # Split and copy
    per_partition_per_stride_size = per_partition_size // stride
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_data_parallel_rank()
    world_size = get_data_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None

def _gather(input_):
    world_size = get_data_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    # last_dim = input_.dim() - 1
    rank = get_data_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_)

    # output = torch.flatten(tensor_list)
    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=0).contiguous()

    return output

def _reduce_scatter(input_list):
    with nvtx.annotate(message="_reduce_scatter.initial", color="green"):
        world_size = get_data_parallel_world_size()
        # Bypass the function if we are using only 1 GPU.
        if world_size==1:
            return input_list

    # Size and dimension.
    with nvtx.annotate(message="_reduce_scatter.empty_like", color="green"):
        output_tensor_ = torch.empty_like(input_list[0])
    with nvtx.annotate(message="_reduce_scatter.reduce_scatter", color="green"):
        torch.distributed.reduce_scatter(output_tensor_, input_list)

    return output_tensor_


class _PartitionedEmbeddingRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, user_weight, user_ids, user_vocab_start_index, user_vocab_end_index,
                item_weight, item_ids, item_vocab_start_index, item_vocab_end_index,
                ne_item_ids,
                sparse, embedding_dim,
                padding_idx, max_norm, norm_type, scale_grad_by_freq):

        with nvtx.annotate(message="_PartitionedEmbeddingRegion.forward", color="green"):
            # gather all input ids across all ranks
            user_count = torch.numel(user_ids)
            pos_item_count = torch.numel(item_ids)

            num_neg_sample = len(ne_item_ids)
            neg_sample_batch = ne_item_ids[0].shape[0]
            each_neg_sample_numel = torch.numel(ne_item_ids[0])
            neg_item_count = each_neg_sample_numel * num_neg_sample

            num_of_batch = 2 + num_neg_sample

            assert user_count == pos_item_count and pos_item_count * num_neg_sample == neg_item_count
            per_rank_inputs_size = user_count + pos_item_count + neg_item_count

            # storing tensors in same order as arguments
            packet_inputs = torch.empty([per_rank_inputs_size], device=user_ids.device, dtype=user_ids.dtype)
            curr_pos = 0
            packet_inputs[curr_pos:curr_pos + user_count] = user_ids.view(-1)[:]
            curr_pos += user_count
            packet_inputs[curr_pos:curr_pos + pos_item_count] = item_ids.view(-1)[:]
            curr_pos += pos_item_count

            for i in range(num_neg_sample):
                packet_inputs[curr_pos:curr_pos + each_neg_sample_numel] = ne_item_ids[i].view(-1)[:]
                curr_pos += each_neg_sample_numel

            # prepare for allgather stuff.
            # tensor in shape [2, 204800 * 3]
            tensors_buffer_ = torch.zeros(get_data_parallel_world_size() * per_rank_inputs_size, device=packet_inputs.device, dtype=packet_inputs.dtype)

            # list (len=2) of tensor in shape [204800 * 3]
            # tensors_ = list(torch.split(tensors_, 1))

            tensors_ = []
            for i in range(get_data_parallel_world_size()):
                buffer_tensor = tensors_buffer_[i * per_rank_inputs_size: (i+1) * per_rank_inputs_size]
                tensors_.append(buffer_tensor.view(per_rank_inputs_size))


            torch.distributed.all_gather(tensors_, packet_inputs)

            with nvtx.annotate(message="_PartitionedEmbeddingRegion.forward1", color="blue"):
                # [2, 204800 * 3]
                # all_gathered_packet_inputs = torch.stack(tensors_) #.view(torch.int16).long()
                all_gathered_packet_inputs = tensors_buffer_.view(get_data_parallel_world_size(), -1)
                # [2, 3, 204800]
                packet = all_gathered_packet_inputs.view(get_data_parallel_world_size(), num_of_batch, per_rank_inputs_size // num_of_batch)
                # [3, 2 * 204800]
                packet = torch.transpose(packet, 0, 1).contiguous().view(num_of_batch, -1)

                def retrieve_embedding(weight, all_inputs, vocab_start_index, vocab_end_index):
                    # Build the mask.
                    input_mask = (all_inputs < vocab_start_index) | \
                                (all_inputs >= vocab_end_index)
                    with nvtx.annotate(message="_PartitionedEmbeddingRegion.forward.nonzero", color="blue"):
                        valid_id_indices = torch.nonzero(torch.logical_not(input_mask))
                    
                    _embedding_loop_up_with_sparse_index = True
                    valid_input_ids_on_current_rank = all_inputs[valid_id_indices] - vocab_start_index
                    if _embedding_loop_up_with_sparse_index is False:
                        # Mask the input.
                        masked_input = all_inputs - vocab_start_index
                        masked_input[input_mask] = 0

                        # Get the embeddings.
                        output_parallel = F.embedding(masked_input, weight,
                                                    padding_idx, max_norm,
                                                    norm_type, scale_grad_by_freq,
                                                    sparse)

                        # Mask the output embedding.
                        output_parallel[input_mask, :] = 0.0
                    else:
                        # #[D]
                        # valid_input_ids_on_current_rank = all_inputs[valid_id_indices] - vocab_start_index
                        # I = torch.unsqueeze(valid_input_ids_on_current_rank, 0)
                        # valid_index = torch.sparse_coo_tensor(I,  torch.empty(4, weight.shape[1]), list(weight.shape)).coalesce()
                        # w.sparse_mask(new_index)
                        with nvtx.annotate(message="_PartitionedEmbeddingRegion.forward.embedding", color="blue"):
                            # valid_input_ids_on_current_rank = all_inputs[valid_id_indices] - vocab_start_index
                            # Get the embeddings.
                            # valid_input_embeddings = F.embedding(valid_input_ids_on_current_rank, weight,
                            #                             padding_idx, max_norm,
                            #                             norm_type, scale_grad_by_freq,
                            #                             sparse)
                            with nvtx.annotate(message="forward.cpu_embedding", color="blue"):
                                valid_input_embeddings = weight[valid_input_ids_on_current_rank]

                            with nvtx.annotate(message="forward.gpu_embedding", color="blue"):
                                valid_input_embeddings = valid_input_embeddings.cuda()

                            output_parallel = torch.zeros((all_inputs.shape[0], weight.shape[1]), dtype=weight.dtype, device=all_inputs.device)
                            output_parallel[valid_id_indices] = valid_input_embeddings
                            

                    return output_parallel, valid_id_indices, valid_input_ids_on_current_rank
                    # return output_parallel.half(), masked_input, input_mask

                # print('CKPT - Tracing 0 - ', packet[0].shape, packet[1:].shape, packet.shape)

                # [2 * 204800, 64], [2 * 204800], [2 * 204800]
            with nvtx.annotate(message="_PartitionedEmbeddingRegion.forward2", color="blue"):
                user_embedding_parallel, user_valid_indices, user_valid_ids = retrieve_embedding(user_weight, packet[0], user_vocab_start_index, user_vocab_end_index)
                # [2 * 2 * 204800, 64], [2 * 2 * 204800], [2 * 2 * 204800]
                pos_and_neg_item_embedding_parallel, item_valid_indices, item_valid_ids = retrieve_embedding(item_weight, packet[1:].view(-1), item_vocab_start_index, item_vocab_end_index)

                # print('CKPT - Tracing 1 - ', user_embedding_parallel.shape, user_masked_input.shape, user_input_mask.shape, pos_and_neg_item_embedding_parallel.shape, item_masked_input.shape, item_input_mask.shape)
                # [3 * 2 * 204800, 64] --> [3, 2, 204800 * 64]
                embedding_all = torch.cat((user_embedding_parallel, pos_and_neg_item_embedding_parallel), 0).view(num_of_batch, get_data_parallel_world_size(), -1)
                # print('CKPT - Tracing 2 - ', embedding_all.shape)

                # [2, 3, 204800 * 64] --> [2, 3 * 204800 * 64]
                embedding_all = torch.transpose(embedding_all, 0, 1).contiguous().view(get_data_parallel_world_size(), -1)
                # print('CKPT - Tracing 3 - ', embedding_all.shape)

                ctx.save_for_backward(user_valid_indices, item_valid_indices, user_weight, item_weight, user_valid_ids, item_valid_ids)

                #print('ckpt3----> output_parallel.shape: ', output_parallel.shape, output_parallel, ', input_mask.shape', input_mask.shape, input_mask)
                # Reduce across all the model parallel GPUs.
            with nvtx.annotate(message="_PartitionedEmbeddingRegion.forward3", color="blue"):
                # list (len=2) of tensor in shape [3 * 204800 * 64]
                # output_tensor_list = list(torch.split(embedding_all, 1))
                output_tensor_list = []
                per_rank_embedding_size = embedding_all.shape[1]
                embedding_all_buffer_ = embedding_all.view(-1)
                for i in range(get_data_parallel_world_size()):
                    buffer_tensor = embedding_all_buffer_[i * per_rank_embedding_size: (i+1) * per_rank_embedding_size]
                    output_tensor_list.append(buffer_tensor.view(per_rank_embedding_size))

                # print('CKPT - Tracing 4 - ', len(output_tensor_list), output_tensor_list[0].shape)
                #print('ckpt3.5----> output_tensor_list[0].shape: ', output_tensor_list[0].shape, output_tensor_list[0])
            with nvtx.annotate(message="_PartitionedEmbeddingRegion.forward4", color="blue"):
                output_tensor_on_current_rank = _reduce_scatter(output_tensor_list)
                # print('CKPT - Tracing 5 - ', output_tensor_on_current_rank.shape)

                # [3, 204800 * 64]
                output_tensor_on_current_rank = output_tensor_on_current_rank.view(num_of_batch, -1)
                # print('CKPT - Tracing 6 - ', output_tensor_on_current_rank.shape)
                user_emebddings, pos_item_embedding, neg_item_embedding_0, \
                    neg_item_embedding_1, neg_item_embedding_2, neg_item_embedding_3, \
                    neg_item_embedding_4 = list(torch.split(output_tensor_on_current_rank, 1))
                # print('CKPT - Tracing 7 - ', user_emebddings.shape, pos_item_embedding.shape, neg_item_embedding.shape)
            with nvtx.annotate(message="_PartitionedEmbeddingRegion.forward5", color="blue"):
                return user_emebddings.view(-1, embedding_dim), pos_item_embedding.view(-1, embedding_dim), \
                    neg_item_embedding_0.view(-1, embedding_dim), neg_item_embedding_1.view(-1, embedding_dim), \
                    neg_item_embedding_2.view(-1, embedding_dim), neg_item_embedding_3.view(-1, embedding_dim), \
                    neg_item_embedding_4.view(-1, embedding_dim)

                #print('ckpt4----> output_tensor_on_current_rank.shape: ', output_tensor_on_current_rank.shape, output_tensor_on_current_rank)

                # return output_tensor_on_current_rank


    @staticmethod
    def backward(ctx, user_emebddings_grad_output, pos_item_embedding_grad_output, \
        neg_item_embedding_grad_output_0, neg_item_embedding_grad_output_1, neg_item_embedding_grad_output_2,\
        neg_item_embedding_grad_output_3, neg_item_embedding_grad_output_4):
        with nvtx.annotate(message="_PartitionedEmbeddingRegion.backward", color="yellow"):
            user_valid_indices, item_valid_indices, user_weight, item_weight, valid_id_indices, item_valid_ids = ctx.saved_tensors

            def compute_gradient(weight, valid_indices, valid_ids, grad_output):
                # with nvtx.annotate(message="logical_not", color="orange"):
                #     valid_mask = torch.logical_not(input_mask)
                # print('ckpt5----> grad_output.shape: ', grad_output.shape, grad_output.dtype)
                with nvtx.annotate(message="_gather", color="orange"):
                    # shape of grad_output: [204800, 64]

                    # [2, 204800 * 64]
                    numel_per_rank = torch.numel(grad_output)
                    tensors_buffer_ = torch.empty(get_data_parallel_world_size() * numel_per_rank, device=grad_output.device, dtype=grad_output.dtype)

                    tensors_ = []
                    for i in range(get_data_parallel_world_size()):
                        buffer_tensor = tensors_buffer_[i * numel_per_rank: (i+1) * numel_per_rank]
                        tensors_.append(buffer_tensor.view(numel_per_rank))

                    # tensors_ = [tensors_buffer_[i].view() for i in range(get_data_parallel_world_size())]#list(torch.split(tensors_, 1))
                    torch.distributed.all_gather(tensors_, grad_output.view(-1))

                # with nvtx.annotate(message="masked_input", color="orange"):
                #     valid_inputs = masked_input[valid_mask] #torch.masked_select(masked_input, valid_mask) #masked_input[valid_mask]

                valid_indices = torch.squeeze(valid_indices, 1)
                valid_ids = torch.squeeze(valid_ids, 1)
                i = torch.unsqueeze(valid_ids, 0)
                v = tensors_buffer_.view(-1, grad_output.shape[1])[valid_indices]

                with nvtx.annotate(message="sparse_coo_tensor", color="orange"):
                    ret = torch.sparse_coo_tensor(i, v, list(weight.shape), device=grad_output.device)
                # print('ckpt7----> ret._nnz(): {} on rank {}: '.format(ret._nnz(), get_data_parallel_rank()))
                return ret

            user_grad = compute_gradient(user_weight, user_valid_indices, valid_id_indices, user_emebddings_grad_output)
            item_embedding_grad_output = torch.stack(
                (
                    pos_item_embedding_grad_output, neg_item_embedding_grad_output_0,
                    neg_item_embedding_grad_output_1, neg_item_embedding_grad_output_2,
                    neg_item_embedding_grad_output_3, neg_item_embedding_grad_output_4)
                ).view(-1, neg_item_embedding_grad_output_0.shape[1])
            item_grad = compute_gradient(item_weight, item_valid_indices, item_valid_ids, item_embedding_grad_output)
            print('>>>>>>>>>>>>>>>>>>>>>> item_grad._nnz(): {} on rank {}: '.format(item_grad._nnz(), get_data_parallel_rank()))
            

            return user_grad, None, None, None, item_grad, None, None, None, None, None, None, None, None, None, None


class PartitionedEmbedding(torch.nn.Module):
    def __init__(self, num_user_embeddings, num_item_embeddings, embedding_dim,
                 sparse=False, init_method=init.xavier_normal_):
        super(PartitionedEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_user_embeddings = num_user_embeddings
        self.num_item_embeddings = num_item_embeddings

        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = sparse
        # self._weight = None

        self.world_size = get_data_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.user_vocab_start_index, self.user_vocab_end_index = \
            get_local_index_range(
                self.num_user_embeddings, get_data_parallel_rank(),
                self.world_size)
        self.num_user_embeddings_per_partition = self.user_vocab_end_index - \
            self.user_vocab_start_index

        self.item_vocab_start_index, self.item_vocab_end_index = \
            get_local_index_range(
                self.num_item_embeddings, get_data_parallel_rank(),
                self.world_size)
        self.num_item_embeddings_per_partition = self.item_vocab_end_index - \
            self.item_vocab_start_index


        # self.batch_size_embeddings = Parameter(torch.empty(
        #     self.batch_size, self.embedding_dim,
        #     dtype=torch.float))


        # Allocate weights and initialize.
        # args = get_args()
        # if args.use_cpu_initialization:
        self.user_weight = Parameter(torch.empty(
            self.num_user_embeddings_per_partition, self.embedding_dim,
            dtype=torch.float))
        _initialize_affine_weight_cpu(
            self.user_weight, self.num_user_embeddings, self.embedding_dim,
            self.num_user_embeddings_per_partition, 0, init_method)

        self.item_weight = Parameter(torch.empty(
            self.num_item_embeddings_per_partition, self.embedding_dim,
            dtype=torch.float))
        _initialize_affine_weight_cpu(
            self.item_weight, self.num_item_embeddings, self.embedding_dim,
            self.num_item_embeddings_per_partition, 0, init_method)

    def forward(self, user_ids, item_ids, ne_item_ids):
        if self.world_size > 1:
            return _PartitionedEmbeddingRegion.apply(
                self.user_weight, user_ids, self.user_vocab_start_index, self.user_vocab_end_index,
                self.item_weight, item_ids, self.item_vocab_start_index, self.item_vocab_end_index,
                ne_item_ids,
                self.sparse, self.embedding_dim,
                self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq)
        else:
            # Get the embeddings.
            user_output_parallel = F.embedding(user_ids, self.user_weight,
                                          self.padding_idx, self.max_norm,
                                          self.norm_type, self.scale_grad_by_freq,
                                          self.sparse)

            item_output_parallel = F.embedding(item_ids, self.item_weight,
                                          self.padding_idx, self.max_norm,
                                          self.norm_type, self.scale_grad_by_freq,
                                          self.sparse)

            ne_item_output_parallel = [F.embedding(ne_item_ids[i], self.item_weight,
                                        self.padding_idx, self.max_norm,
                                        self.norm_type, self.scale_grad_by_freq,
                                        self.sparse) for i in range(len(ne_item_ids))]

            return user_output_parallel, item_output_parallel, ne_item_output_parallel[0], ne_item_output_parallel[1],\
                ne_item_output_parallel[2], ne_item_output_parallel[3], ne_item_output_parallel[4]

use_ort=True
use_embedding_as_input = True
sparse_embedding_grad = False
batch = 2048 #00 
num_users = 7000000
num_items = 34820728


torch.distributed.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(torch.distributed.get_rank() % torch.distributed.get_world_size())
print('current cuda device is ', torch.cuda.current_device())



factors = 128
all_embeddings = PartitionedEmbedding(
    num_user_embeddings=num_users, num_item_embeddings=num_items,
    embedding_dim=factors, sparse=True)
