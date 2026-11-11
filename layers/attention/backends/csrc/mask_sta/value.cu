/*
 * Copyright (c) 2024 by SageAttention team.
 * Licensed under the Apache License, Version 2.0.
 * See vendor/LICENSE for the license text.
 *
 * Modified for fixed-shape indexed input in EraserDiT.
 */
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

constexpr uint32_t L = 32640, H = 32, D = 64;
#include "sage/cp_async.cuh"

template <uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero=false, typename T>
__global__ void StaIndexedVTransposeKernel(T *__restrict__ input, T *__restrict__ output, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output, const int32_t* __restrict__ row_order)
{

  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  uint32_t num_threads_per_token = head_dim / pack_size;
  uint32_t num_threads_per_cta = CTA_SIZE / pack_size;

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;

  const uint32_t src_token = thread_base_token < num_tokens ? row_order[thread_base_token] : 0;
  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + src_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T* output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + bx * CTA_SIZE + thread_id % num_threads_per_cta * pack_size + thread_id / num_threads_per_cta * stride_d_output;

  __shared__ T shared_load[CTA_SIZE][head_dim];
  __shared__ T shared_store[head_dim][CTA_SIZE];

  // 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
  // permute on the seq dimension for fp8 mma
  uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
  uint32_t smem_load_row_mod = (thread_id / num_threads_per_token) % 16;
  uint32_t smem_load_row = smem_load_row_base + (smem_load_row_mod  / 8) * 2 + ((smem_load_row_mod / 2) % 4) * 4 + (smem_load_row_mod % 2);

  constexpr cp_async::SharedMemFillMode fill_mode = pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;
  cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(shared_load[smem_load_row] + thread_id % num_threads_per_token * pack_size, input_ptr_base, thread_base_token < num_tokens);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  uint32_t smem_row_base = thread_id % CTA_SIZE;
  uint32_t smem_col_base = thread_id / CTA_SIZE;
  uint32_t smem_col_stride = head_dim / 8;

  // TODO: use ldmatrix to do permutation
#pragma unroll
  for (uint32_t i = 0; i < 8; i++)
  {
    shared_store[smem_col_base + i * smem_col_stride][smem_row_base] = shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
  }

  __syncthreads();

  *(float4*)(output_ptr_base) = *(float4*)(&shared_store[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size]);
}



extern "C" int sta_mask_prepare_v(void* value, void* transposed, const int32_t* row_order, cudaStream_t stream) {
  constexpr uint32_t TILE = 64;
  StaIndexedVTransposeKernel<D, TILE, true, nv_bfloat16>
    <<<dim3(L / TILE, H, 1), TILE * (D / 8), 0, stream>>>(
      static_cast<nv_bfloat16*>(value), static_cast<nv_bfloat16*>(transposed), L,
      L * H * D, H * D, D, L * H * D, H * L, L, row_order);
  return cudaGetLastError();
}
