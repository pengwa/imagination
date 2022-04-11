// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright NVIDIA/apex
// This file is adapted from NVIDIA/apex, commit 0c7d8e3fa9a095a1641a2290877436d0314b69c6

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <THC/THCDeviceUtils.cuh>

// Another possibility:
#include <torch/all.h>

#include <assert.h>
// Stringstream is a big hammer, but I want to rely on operator<< for dtype.
#include <sstream>
#include "type_shim.h"


#define BLOCK_SIZE 128
#define GPU_WARP_SIZE 32


template<typename in_t, typename out_t, int64_t embedding_dim>
__global__ void index_elementwise_kernel(
  int64_t* global_idx_ptr,
  const int64_t global_size,
  const int64_t local_vocab_start_index,
  const int64_t local_vocab_end_index,
  const in_t* local_vocab_embedding_ptr,
  out_t* global_out_embedding_ptr,
  const int64_t total_output_embedding_element_count) {
  constexpr int64_t ILP = embedding_dim / GPU_WARP_SIZE;
  int64_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t start_of_element_index_for_this_thread = linearId * ILP;
  if (start_of_element_index_for_this_thread >= total_output_embedding_element_count) return;

  int64_t idx_index = start_of_element_index_for_this_thread / embedding_dim;
  int64_t embedding_idx = global_idx_ptr[idx_index];

  if (embedding_idx < local_vocab_start_index || embedding_idx >= local_vocab_end_index) {
    out_t* out = global_out_embedding_ptr + start_of_element_index_for_this_thread;
    *out = static_cast<out_t>(0.);
    *(out + 1) = static_cast<out_t>(0.);
    *(out + 2) = static_cast<out_t>(0.);
    *(out + 3) = static_cast<out_t>(0.);
    return;
  }

// reinterpret_cast<const float4*>
  int64_t local_embedding_idx = embedding_idx - local_vocab_start_index;

// # if __CUDA_ARCH__>=200
//     printf("%ld ---> %ld --> %ld --> %ld --> %ld \n", linearId, local_embedding_idx, local_vocab_start_index, local_vocab_end_index, idx_index);
// #endif  

  int64_t offset = start_of_element_index_for_this_thread % embedding_dim;
  const float4 emb = *reinterpret_cast<const float4*>(local_vocab_embedding_ptr + embedding_dim * local_embedding_idx + offset);
  out_t* out = global_out_embedding_ptr + start_of_element_index_for_this_thread;
  *out = static_cast<out_t>(emb.x);
  *(out + 1) = static_cast<out_t>(emb.y);
  *(out + 2) = static_cast<out_t>(emb.z);
  *(out + 3) = static_cast<out_t>(emb.w);
}



__host__ __device__ __forceinline__ int64_t THCCeilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

static void launch_kernel(at::Tensor& global_idx,
                   const int local_vocab_start_index,
                   const int local_vocab_end_index,
                   const at::Tensor& local_vocab_embedding,
                   at::Tensor& global_out_embedding) {

  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t output_element_count = global_idx.size(0) * local_vocab_embedding.size(1);
  int64_t ILP = local_vocab_embedding.size(1) / GPU_WARP_SIZE;
  int64_t grid_size = THCCeilDiv(output_element_count, BLOCK_SIZE * ILP);

  // if (fp16_output) {
  int64_t embedding_dim = local_vocab_embedding.size(1);
  if (embedding_dim == 128) {
    DISPATCH_DOUBLE_FLOAT_AND_HALF(local_vocab_embedding.scalar_type(), 0, "index_elementwise_kernel",
      DISPATCH_DOUBLE_FLOAT_AND_HALF(global_out_embedding.scalar_type(), 1, "index_elementwise_kernel",
        index_elementwise_kernel<scalar_t_0, scalar_t_1, 128><<<grid_size, BLOCK_SIZE, 0, stream>>>(
          global_idx.data_ptr<int64_t>(),
          global_idx.size(0),
          static_cast<int64_t>(local_vocab_start_index),
          static_cast<int64_t>(local_vocab_end_index),
          local_vocab_embedding.data_ptr<scalar_t_0>(),
          global_out_embedding.data_ptr<scalar_t_1>(),
          output_element_count)
      ));
  }

  // } else {
  //   index_elementwise_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
  //     global_idx.data_ptr<int64_t>(),
  //     global_idx.size(0),
  //     local_vocab_start_index,
  //     local_vocab_end_index,
  //     local_vocab_embedding.data_ptr<in_t>(),
  //     global_out_embedding.data_ptr<in_t>(),
  //     local_vocab_embedding.size(1));
  //     C10_CUDA_KERNEL_LAUNCH_CHECK();
  // }
  
}

void distributed_embedding_lookup(at::Tensor& global_idx,
                                  const int local_vocab_start_index,
                                  const int local_vocab_end_index,
                                  const at::Tensor& local_vocab_embedding,
                                  at::Tensor& global_out_embedding)
{
  using namespace at;
  // The output (downscaled) type is always float.
  // If build times suffer, think about where to put this dispatch,
  // and what logic should be moved out of multi_tensor_apply.
  launch_kernel(
    global_idx,
    local_vocab_start_index,
    local_vocab_end_index,
    local_vocab_embedding,
    global_out_embedding
  );
  AT_CUDA_CHECK(cudaGetLastError());
}
