// tern_kernel.cu
// Core ternary quantization kernels.
//
// This is the main CUDA file. It has three jobs:
//   1. Quantize FP32 weights into packed ternary uint32 words
//   2. Run the forward pass (ternary weights x INT8 activations)
//   3. Decode packed weights back to float (for debug/testing)
//
// Written targeting sm_86 (RTX 3060 / Ampere).
// Should work on sm_80+ without changes.
//
// First attempt notes (keeping these so I remember what I tried):
//   - Originally tried Base-3 packing (5 values per byte) like EL_X's version
//     but the decode math was annoying and slower. Switched to 2-bit / 16-per-word.
//   - Tried __int128 for the packed type. Don't do that. Just use uint32.
//   - The absmean scale is per output-row, not per whole matrix. Learned that the
//     hard way when accuracy was garbage.

#include <cuda_runtime.h>
#include <cuda_fp16.h>    // for __half
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "../include/tern_kernel.h"
#include "../include/gpu_utils.h"


// ============================================================
// INTERNAL CONSTANTS
// ============================================================

// Tile size for the shared-memory GEMM in the forward pass.
// 16x16 tiles work well on the 3060 - good occupancy without
// blowing the shared memory budget.
//
// Tried 32x32 - too much shared mem, occupancy dropped.
// Tried 8x8  - too many global memory transactions.
// 16x16 seems like the sweet spot here.
#define TILE_M 16
#define TILE_N 16


// ============================================================
// KERNEL 1 - WEIGHT QUANTIZER
// ============================================================
//
// One thread block handles one output row.
// Threads in the block cooperate to:
//   a) find the row's absmean scale
//   b) threshold each weight to -1, 0, or +1
//   c) pack 16 weights into one uint32
//
// Grid:  (K,)       -- one block per output row
// Block: (256,)     -- 256 threads per row
//
// __global__ means this function runs on the GPU, called from CPU.

__global__ void kernel_quantize_weights(
    const float* __restrict__  weights,   // [K, N] input
    uint32_t*                  packed,    // [K, ceil(N/16)] output
    float*                     scales,    // [K] output (one scale per row)
    int K,
    int N
) {
    // Which output row are we handling?
    int row = blockIdx.x;
    if (row >= K) return;

    // Pointer to the start of this row in the weight matrix.
    const float* row_ptr = weights + row * N;

    // ----------------------------------------------------------
    // Step 1: Compute absmean of this row.
    // Each thread handles a strided subset of the columns,
    // then we reduce across threads using shared memory.
    // ----------------------------------------------------------
    __shared__ float shared_sum[256];

    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        thread_sum += fabsf(row_ptr[col]);
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction: fold the 256-element array down to 1 value.
    // Each round halves the active threads.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the final scale.
    // Add a tiny epsilon so we never divide by zero on an all-zero row.
    float scale = 1.0f;
    if (threadIdx.x == 0) {
        scale = shared_sum[0] / (float)N + 1e-8f;
        scales[row] = scale;
    }
    // Broadcast the scale to all threads in the block.
    __shared__ float shared_scale;
    if (threadIdx.x == 0) shared_scale = scale;
    __syncthreads();
    scale = shared_scale;

    // ----------------------------------------------------------
    // Step 2: Threshold and pack.
    // We process 16 weights at a time (one uint32 output word).
    // Each thread handles one or more words independently.
    // ----------------------------------------------------------
    int num_words = (N + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD;
    uint32_t* packed_row = packed + row * num_words;

    for (int word_idx = threadIdx.x; word_idx < num_words; word_idx += blockDim.x) {
        uint32_t word = 0;

        for (int bit = 0; bit < WEIGHTS_PER_WORD; bit++) {
            int col = word_idx * WEIGHTS_PER_WORD + bit;

            // If we're past the end of the row, treat as zero weight.
            // This handles the case where N isn't a multiple of 16.
            uint32_t code = TERN_ZERO;
            if (col < N) {
                float w_normalized = row_ptr[col] / scale;

                // The threshold is 0.5 -- anything above +0.5 -> +1,
                // below -0.5 -> -1, otherwise 0.
                // Some papers use different thresholds; 0.5 works here.
                if (w_normalized > 0.5f) {
                    code = TERN_POS_ONE;
                } else if (w_normalized < -0.5f) {
                    code = TERN_NEG_ONE;
                }
                // else code stays TERN_ZERO = 0b00
            }

            // Shift this 2-bit code into position within the word.
            // bit 0 occupies bits [1:0], bit 1 occupies bits [3:2], etc.
            word |= (code << (bit * 2));
        }

        packed_row[word_idx] = word;
    }
}


// ============================================================
// KERNEL 2 - FORWARD PASS  (BitLinear W1.58 A8)
// ============================================================
//
// Compute:  Y = dequant( quantize(X) @ W_ternary ) * scale_w
//
// X is FP16 activations [M, N], quantized to INT8 per row.
// W is ternary packed weights [K, ceil(N/16)].
// Y is FP16 output [M, K].
//
// Grid:  (ceil(M/TILE_M), ceil(K/TILE_N))
// Block: (TILE_N, TILE_M)  = (16, 16) = 256 threads
//
// Each block computes one TILE_M x TILE_N sub-tile of the output.

__global__ void kernel_bitlinear_forward(
    const __half* __restrict__ X,
    const uint32_t* __restrict__ W_packed,
    const float* __restrict__ W_scales,
    __half* Y,
    int M, int N, int K
) {
    int out_row = blockIdx.x * TILE_M + threadIdx.y;
    int out_col = blockIdx.y * TILE_N + threadIdx.x;

    // Do NOT early-return -- out-of-bounds threads must still
    // hit __syncthreads() calls or the whole block hangs.
    bool valid = (out_row < M) && (out_col < K);

    __shared__ int8_t   smem_X[TILE_M][TILE_N];
    __shared__ float    smem_absmax[TILE_M];
    __shared__ uint32_t smem_W[TILE_N];

    int32_t acc = 0;
    float row_absmax = 0.0f;
    int num_words = (N + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD;

    for (int word_idx = 0; word_idx < num_words; word_idx++) {
        int tile_col_start = word_idx * WEIGHTS_PER_WORD;

        // ---- Phase A+B combined: single read, compute absmax + quantize ----
        {
            int row    = out_row;
            int col_lo = tile_col_start + threadIdx.x * 2;
            int col_hi = col_lo + 1;

            float vals_x = (row < M && col_lo < N) ? __half2float(X[row * N + col_lo]) : 0.0f;
            float vals_y = (row < M && col_hi < N) ? __half2float(X[row * N + col_hi]) : 0.0f;

            float v = fmaxf(fabsf(vals_x), fabsf(vals_y));
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 8));
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 4));
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 2));
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 1));

            if (threadIdx.x == 0) smem_absmax[threadIdx.y] = v;
            __syncthreads();

            float tile_absmax = smem_absmax[threadIdx.y];
            if (tile_absmax > row_absmax) row_absmax = tile_absmax;

            float act_scale = tile_absmax / 127.0f + 1e-8f;
            if (threadIdx.x < TILE_N / 2) {
                smem_X[threadIdx.y][threadIdx.x * 2]     = (int8_t)fminf(fmaxf(vals_x / act_scale, -127.0f), 127.0f);
                smem_X[threadIdx.y][threadIdx.x * 2 + 1] = (int8_t)fminf(fmaxf(vals_y / act_scale, -127.0f), 127.0f);
            }
        }

        // ---- Phase C: load weight words into smem_W ----
        if (threadIdx.y == 0) {
            int w_col = blockIdx.y * TILE_N + threadIdx.x;
            smem_W[threadIdx.x] = (w_col < K)
                ? W_packed[w_col * num_words + word_idx]
                : 0u;
        }
        __syncthreads();   // smem_X and smem_W both ready

        // ---- Phase D: dot product from shared memory ----
        if (valid) {
            uint32_t word = smem_W[threadIdx.x];
            int cols_in_tile = min(WEIGHTS_PER_WORD, N - tile_col_start);

            if (cols_in_tile == WEIGHTS_PER_WORD) {
#pragma unroll
                for (int bit = 0; bit < WEIGHTS_PER_WORD; bit++) {
                    uint32_t code = (word >> (bit * 2)) & 0x3;
                    int      w_q = (int)(code & 1) - (int)((code >> 1) & 1);
                    acc += (int32_t)smem_X[threadIdx.y][bit] * w_q;
                }
            } else {
                for (int bit = 0; bit < cols_in_tile; bit++) {
                    uint32_t code = (word >> (bit * 2)) & 0x3;
                    int      w_q = (int)(code & 1) - (int)((code >> 1) & 1);
                    acc += (int32_t)smem_X[threadIdx.y][bit] * w_q;
                }
            }
        }
        __syncthreads();   // done with smem before next iteration
    }

    // ---- Dequantize and write output ----
    if (valid) {
        float act_scale = row_absmax / 127.0f + 1e-8f;
        float result = (float)acc * act_scale * W_scales[out_col];
        Y[out_row * K + out_col] = __float2half(result);
    }
}


// ============================================================
// KERNEL 3 - DECODE (for debugging)
// ============================================================
//
// Just unpacks the ternary weights back to float.
// Not used in inference - only for verifying the quantizer
// didn't mess up.
//
// Grid:  (K,)
// Block: (256,)

__global__ void kernel_decode_weights(
    const uint32_t* __restrict__ packed,   // [K, ceil(N/16)]
    const float* __restrict__    scales,   // [K]
    float*                       decoded,  // [K, N] output
    int K, int N
) {
    int row = blockIdx.x;
    if (row >= K) return;

    float scale = scales[row];
    int num_words = (N + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD;
    const uint32_t* packed_row = packed + row * num_words;
    float* decoded_row = decoded + row * N;

    for (int word_idx = threadIdx.x; word_idx < num_words; word_idx += blockDim.x) {
        uint32_t word = packed_row[word_idx];

        for (int bit = 0; bit < WEIGHTS_PER_WORD; bit++) {
            int col = word_idx * WEIGHTS_PER_WORD + bit;
            if (col >= N) break;

            uint32_t code = (word >> (bit * 2)) & 0x3;
            int w_q = (int)(code & 1) - (int)((code >> 1) & 1);

            // Multiply back by scale to get the approximate original value
            decoded_row[col] = (float)w_q * scale;
        }
    }
}


// ============================================================
// HOST-SIDE LAUNCHERS
// These are the C-linkage functions declared in tern_kernel.h
// ============================================================

void launch_quantize_weights(
    const float*  weights_fp32,
    uint32_t*     packed_out,
    float*        scales_out,
    int K, int N
) {
    // One block per output row. 256 threads per block.
    dim3 grid(K);
    dim3 block(256);

    kernel_quantize_weights<<<grid, block>>>(
        weights_fp32, packed_out, scales_out, K, N
    );

    // Always sync and check after a kernel launch during development.
    // (Can remove in final release builds for speed.)
    CUDA_CHECK(cudaDeviceSynchronize());
}


void launch_bitlinear_forward(
    const void*       activations,
    const uint32_t*   packed_w,
    const float*      scales_w,
    void*             output,
    int M, int N, int K
) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (K + TILE_N - 1) / TILE_N
    );

    kernel_bitlinear_forward<<<grid, block>>>(
        (const __half*)activations,
        packed_w,
        scales_w,
        (__half*)output,
        M, N, K
    );
}


void launch_decode_weights(
    const uint32_t* packed_w,
    const float*    scales_w,
    float*          decoded_out,
    int K, int N
) {
    dim3 grid(K);
    dim3 block(256);

    kernel_decode_weights<<<grid, block>>>(
        packed_w, scales_w, decoded_out, K, N
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}
