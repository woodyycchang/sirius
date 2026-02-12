/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_helper.cuh"
#include "gpu_buffer_manager.hpp"
#include "gpu_physical_nested_loop_join.hpp"
#include "log/logging.hpp"

namespace duckdb {

// TODO: Currently only support a single key
template <typename T, int B, int I>
__global__ void nested_loop_join_count(T* left_keys,
                                       T* right_keys,
                                       uint64_t* offset_each_thread,
                                       unsigned long long* total_count,
                                       uint64_t left_size,
                                       uint64_t right_size,
                                       int condition_mode)
{
  typedef cub::BlockScan<int, B> BlockScanInt;

  __shared__ union TempStorage {
    typename BlockScanInt::TempStorage scan;
  } temp_storage;

  int items_count[I];

  uint64_t tile_size   = B * I;
  uint64_t tile_offset = blockIdx.x * tile_size;

  uint64_t num_tiles      = (left_size + tile_size - 1) / tile_size;
  uint64_t num_tile_items = tile_size;

  int t_count   = 0;  // Number of items selected per thread
  int c_t_count = 0;  // Prefix sum of t_count
  __shared__ uint64_t block_off;

  if (blockIdx.x == num_tiles - 1) { num_tile_items = left_size - tile_offset; }

#pragma unroll
  for (int ITEM = 0; ITEM < I; ITEM++) {
    items_count[ITEM] = 0;
  }

#pragma unroll
  for (int ITEM = 0; ITEM < I; ITEM++) {
    if (threadIdx.x + (ITEM * B) < num_tile_items) {
      for (int i = 0; i < right_size; i++) {
        bool local_found = 1;
        if (condition_mode == 0 &&
            left_keys[tile_offset + threadIdx.x + ITEM * B] != right_keys[i]) {
          local_found = 0;
        } else if (condition_mode == 1 &&
                   left_keys[tile_offset + threadIdx.x + ITEM * B] == right_keys[i]) {
          local_found = 0;
        } else if (condition_mode == 2 &&
                   left_keys[tile_offset + threadIdx.x + ITEM * B] >= right_keys[i]) {
          local_found = 0;
        } else if (condition_mode == 3 &&
                   left_keys[tile_offset + threadIdx.x + ITEM * B] <= right_keys[i]) {
          local_found = 0;
        }

        if (local_found) { items_count[ITEM]++; }
      }
      t_count += items_count[ITEM];
    }
  }

  // Barrier
  __syncthreads();

  BlockScanInt(temp_storage.scan)
    .ExclusiveSum(t_count, c_t_count);  // doing a prefix sum of all the previous threads in the
                                        // block and store it to c_t_count
  if (threadIdx.x ==
      blockDim.x - 1) {  // if the last thread in the block, add the prefix sum of all the prev
                         // threads + sum of my threads to global variable total
    block_off =
      atomicAdd(total_count,
                (unsigned long long)t_count +
                  c_t_count);  // the previous value of total is gonna be assigned to block_off
  }  // block_off does not need to be global (it's just need to be shared), because it will get the
     // previous value from total which is global

  __syncthreads();

  if (blockIdx.x * tile_size + threadIdx.x < left_size) {
    offset_each_thread[blockIdx.x * B + threadIdx.x] = block_off + c_t_count;
  }
}

template <typename T, int B, int I>
__global__ void nested_loop_join(T* left_keys,
                                 T* right_keys,
                                 uint64_t* offset_each_thread,
                                 uint64_t* row_ids_left,
                                 uint64_t* row_ids_right,
                                 uint64_t left_size,
                                 uint64_t right_size,
                                 int condition_mode)
{
  uint64_t tile_size   = B * I;
  uint64_t tile_offset = blockIdx.x * tile_size;

  uint64_t num_tiles      = (left_size + tile_size - 1) / tile_size;
  uint64_t num_tile_items = tile_size;

  if (blockIdx.x == num_tiles - 1) { num_tile_items = left_size - tile_offset; }

  uint64_t output_offset = 0;
  if (blockIdx.x * tile_size + threadIdx.x < left_size) {
    output_offset = offset_each_thread[blockIdx.x * B + threadIdx.x];
  }

#pragma unroll
  for (int ITEM = 0; ITEM < I; ITEM++) {
    if (threadIdx.x + (ITEM * B) < num_tile_items) {
      for (int i = 0; i < right_size; i++) {
        bool local_found = 1;
        if (condition_mode == 0 &&
            left_keys[tile_offset + threadIdx.x + ITEM * B] != right_keys[i]) {
          local_found = 0;
        } else if (condition_mode == 1 &&
                   left_keys[tile_offset + threadIdx.x + ITEM * B] == right_keys[i]) {
          local_found = 0;
        } else if (condition_mode == 2 &&
                   left_keys[tile_offset + threadIdx.x + ITEM * B] >= right_keys[i]) {
          local_found = 0;
        } else if (condition_mode == 3 &&
                   left_keys[tile_offset + threadIdx.x + ITEM * B] <= right_keys[i]) {
          local_found = 0;
        }

        if (local_found) {
          row_ids_right[output_offset] = i;
          row_ids_left[output_offset]  = tile_offset + threadIdx.x + ITEM * B;
          output_offset++;
        }
      }
    }
  }
}


// Multi-key NLJ: evaluates ALL join conditions (not just the first key)
template <typename T, int B, int I>
__global__ void nested_loop_join_count_multikey(T** all_left_keys, T** all_right_keys,
                                                int* all_condition_modes, int num_keys,
                                                uint64_t* offset_each_thread,
                                                unsigned long long* total_count,
                                                uint64_t left_size, uint64_t right_size)
{
  typedef cub::BlockScan<int, B> BlockScanInt;
  __shared__ union TempStorage {
    typename BlockScanInt::TempStorage scan;
  } temp_storage;
  int items_count[I];
  uint64_t tile_size      = B * I;
  uint64_t tile_offset    = blockIdx.x * tile_size;
  uint64_t num_tiles      = (left_size + tile_size - 1) / tile_size;
  uint64_t num_tile_items = tile_size;
  int t_count             = 0;
  int c_t_count           = 0;
  __shared__ uint64_t block_off;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = left_size - tile_offset;
  }
#pragma unroll
  for (int ITEM = 0; ITEM < I; ITEM++) {
    items_count[ITEM] = 0;
  }
#pragma unroll
  for (int ITEM = 0; ITEM < I; ITEM++) {
    if (threadIdx.x + (ITEM * B) < num_tile_items) {
      uint64_t left_idx = tile_offset + threadIdx.x + ITEM * B;
      for (uint64_t i = 0; i < right_size; i++) {
        bool local_found = true;
        for (int k = 0; k < num_keys; k++) {
          T left_val  = all_left_keys[k][left_idx];
          T right_val = all_right_keys[k][i];
          int mode    = all_condition_modes[k];
          bool key_match = false;
          if (mode == 0)      key_match = (left_val == right_val);
          else if (mode == 1) key_match = (left_val != right_val);
          else if (mode == 2) key_match = (left_val < right_val);
          else if (mode == 3) key_match = (left_val > right_val);
          if (!key_match) { local_found = false; break; }
        }
        if (local_found) {
          items_count[ITEM]++;
        }
      }
      t_count += items_count[ITEM];
    }
  }
  __syncthreads();
  BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count);
  if (threadIdx.x == blockDim.x - 1) {
    block_off = atomicAdd(total_count, (unsigned long long)t_count + c_t_count);
  }
  __syncthreads();
  if (blockIdx.x * tile_size + threadIdx.x < left_size) {
    offset_each_thread[blockIdx.x * B + threadIdx.x] = block_off + c_t_count;
  }
}

template <typename T, int B, int I>
__global__ void nested_loop_join_multikey(T** all_left_keys, T** all_right_keys,
                                          int* all_condition_modes, int num_keys,
                                          uint64_t* offset_each_thread,
                                          uint64_t* row_ids_left, uint64_t* row_ids_right,
                                          uint64_t left_size, uint64_t right_size)
{
  uint64_t tile_size      = B * I;
  uint64_t tile_offset    = blockIdx.x * tile_size;
  uint64_t num_tiles      = (left_size + tile_size - 1) / tile_size;
  uint64_t num_tile_items = tile_size;
  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = left_size - tile_offset;
  }
  uint64_t output_offset = 0;
  if (blockIdx.x * tile_size + threadIdx.x < left_size) {
    output_offset = offset_each_thread[blockIdx.x * B + threadIdx.x];
  }
#pragma unroll
  for (int ITEM = 0; ITEM < I; ITEM++) {
    if (threadIdx.x + (ITEM * B) < num_tile_items) {
      uint64_t left_idx = tile_offset + threadIdx.x + ITEM * B;
      for (uint64_t i = 0; i < right_size; i++) {
        bool local_found = true;
        for (int k = 0; k < num_keys; k++) {
          T left_val  = all_left_keys[k][left_idx];
          T right_val = all_right_keys[k][i];
          int mode    = all_condition_modes[k];
          bool key_match = false;
          if (mode == 0)      key_match = (left_val == right_val);
          else if (mode == 1) key_match = (left_val != right_val);
          else if (mode == 2) key_match = (left_val < right_val);
          else if (mode == 3) key_match = (left_val > right_val);
          if (!key_match) { local_found = false; break; }
        }
        if (local_found) {
          row_ids_right[output_offset] = i;
          row_ids_left[output_offset]  = left_idx;
          output_offset++;
        }
      }
    }
  }
}

template __global__ void nested_loop_join_count<double, BLOCK_THREADS, 1>(
  double* left_keys,
  double* right_keys,
  uint64_t* offset_each_thread,
  unsigned long long* total_count,
  uint64_t left_size,
  uint64_t right_size,
  int condition_mode);

// Multi-key kernel instantiations
template __global__ void nested_loop_join_count_multikey<double, BLOCK_THREADS, 1>(
  double** all_left_keys, double** all_right_keys, int* all_condition_modes, int num_keys,
  uint64_t* offset_each_thread, unsigned long long* total_count,
  uint64_t left_size, uint64_t right_size);
template __global__ void nested_loop_join_count_multikey<uint64_t, BLOCK_THREADS, 1>(
  uint64_t** all_left_keys, uint64_t** all_right_keys, int* all_condition_modes, int num_keys,
  uint64_t* offset_each_thread, unsigned long long* total_count,
  uint64_t left_size, uint64_t right_size);
template __global__ void nested_loop_join_multikey<double, BLOCK_THREADS, 1>(
  double** all_left_keys, double** all_right_keys, int* all_condition_modes, int num_keys,
  uint64_t* offset_each_thread, uint64_t* row_ids_left, uint64_t* row_ids_right,
  uint64_t left_size, uint64_t right_size);
template __global__ void nested_loop_join_multikey<uint64_t, BLOCK_THREADS, 1>(
  uint64_t** all_left_keys, uint64_t** all_right_keys, int* all_condition_modes, int num_keys,
  uint64_t* offset_each_thread, uint64_t* row_ids_left, uint64_t* row_ids_right,
  uint64_t left_size, uint64_t right_size);


template __global__ void nested_loop_join_count<uint64_t, BLOCK_THREADS, 1>(
  uint64_t* left_keys,
  uint64_t* right_keys,
  uint64_t* offset_each_thread,
  unsigned long long* total_count,
  uint64_t left_size,
  uint64_t right_size,
  int condition_mode);

template __global__ void nested_loop_join<double, BLOCK_THREADS, 1>(double* left_keys,
                                                                    double* right_keys,
                                                                    uint64_t* offset_each_thread,
                                                                    uint64_t* row_ids_left,
                                                                    uint64_t* row_ids_right,
                                                                    uint64_t left_size,
                                                                    uint64_t right_size,
                                                                    int condition_mode);

template __global__ void nested_loop_join<uint64_t, BLOCK_THREADS, 1>(uint64_t* left_keys,
                                                                      uint64_t* right_keys,
                                                                      uint64_t* offset_each_thread,
                                                                      uint64_t* row_ids_left,
                                                                      uint64_t* row_ids_right,
                                                                      uint64_t left_size,
                                                                      uint64_t right_size,
                                                                      int condition_mode);

template <typename T>
void nestedLoopJoin(T** left_data,
                    T** right_data,
                    uint64_t*& row_ids_left,
                    uint64_t*& row_ids_right,
                    uint64_t*& count,
                    uint64_t left_size,
                    uint64_t right_size,
                    int* condition_mode,
                    int num_keys)
{
  CHECK_ERROR();
  SETUP_TIMING();
  START_TIMER();
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  if (left_size == 0 || right_size == 0) {
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    h_count[0]        = 0;
    count             = h_count;
    SIRIUS_LOG_DEBUG("Input size is 0");
    return;
  }
  SIRIUS_LOG_DEBUG("Launching Nested Loop Join Kernel");
  int tile_items = BLOCK_THREADS * 1;
  count          = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
  cudaMemset(count, 0, sizeof(uint64_t));
  uint64_t* offset_each_thread = gpuBufferManager->customCudaMalloc<uint64_t>(
    ((left_size + tile_items - 1) / tile_items) * BLOCK_THREADS, 0, 0);
  if (num_keys > 1) {
    T** d_left_keys;
    T** d_right_keys;
    int* d_condition_modes;
    cudaMalloc(&d_left_keys, num_keys * sizeof(T*));
    cudaMalloc(&d_right_keys, num_keys * sizeof(T*));
    cudaMalloc(&d_condition_modes, num_keys * sizeof(int));
    cudaMemcpy(d_left_keys, left_data, num_keys * sizeof(T*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_keys, right_data, num_keys * sizeof(T*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_condition_modes, condition_mode, num_keys * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_ERROR();
    nested_loop_join_count_multikey<T, BLOCK_THREADS, 1>
      <<<(left_size + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
        d_left_keys, d_right_keys, d_condition_modes, num_keys,
        offset_each_thread, (unsigned long long*)count, left_size, right_size);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    SIRIUS_LOG_DEBUG("Nested Loop Join Result Count: {}", h_count[0]);
    if (h_count[0] == 0) {
      row_ids_left  = nullptr;
      row_ids_right = nullptr;
      gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset_each_thread), 0);
      gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
      cudaFree(d_left_keys);
      cudaFree(d_right_keys);
      cudaFree(d_condition_modes);
      count = h_count;
      STOP_TIMER();
      return;
    }
    row_ids_left  = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    nested_loop_join_multikey<T, BLOCK_THREADS, 1>
      <<<(left_size + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
        d_left_keys, d_right_keys, d_condition_modes, num_keys,
        offset_each_thread, row_ids_left, row_ids_right, left_size, right_size);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    cudaFree(d_left_keys);
    cudaFree(d_right_keys);
    cudaFree(d_condition_modes);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset_each_thread), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
    count = h_count;
  } else {
    nested_loop_join_count<T, BLOCK_THREADS, 1>
      <<<(left_size + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
        left_data[0], right_data[0], offset_each_thread,
        (unsigned long long*)count, left_size, right_size, condition_mode[0]);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    SIRIUS_LOG_DEBUG("Nested Loop Join Result Count: {}", h_count[0]);
    if (h_count[0] == 0) {
      row_ids_left  = nullptr;
      row_ids_right = nullptr;
      gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset_each_thread), 0);
      gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
      count = h_count;
      STOP_TIMER();
      return;
    }
    row_ids_left  = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    nested_loop_join<T, BLOCK_THREADS, 1>
      <<<(left_size + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
        left_data[0], right_data[0], offset_each_thread,
        row_ids_left, row_ids_right, left_size, right_size, condition_mode[0]);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset_each_thread), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
    count = h_count;
  }
  STOP_TIMER();
}

template void nestedLoopJoin<double>(double** left_data,
                                     double** right_data,
                                     uint64_t*& row_ids_left,
                                     uint64_t*& row_ids_right,
                                     uint64_t*& count,
                                     uint64_t left_size,
                                     uint64_t right_size,
                                     int* condition_mode,
                                     int num_keys);

template void nestedLoopJoin<uint64_t>(uint64_t** left_data,
                                       uint64_t** right_data,
                                       uint64_t*& row_ids_left,
                                       uint64_t*& row_ids_right,
                                       uint64_t*& count,
                                       uint64_t left_size,
                                       uint64_t right_size,
                                       int* condition_mode,
                                       int num_keys);

}  // namespace duckdb
