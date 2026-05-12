#pragma once

// tern_kernel.h
// Header for our ternary quantization CUDA kernels.
//
// Weight encoding used throughout this project:
//   00 (binary) -> 0
//   01 (binary) -> +1
//   10 (binary) -> -1
//   11 (binary) -> reserved / treated as 0
//
// We pack 16 ternary weights into a single uint32.
// That means a weight row of 1024 floats -> 64 uint32 values.
// Huge memory savings vs FP16 (which would be 2048 bytes vs 256 bytes).
//
// This header is included by both the CUDA side (.cu) and the
// Python extension wrapper (.cpp), so keep it clean of CUDA-only types
// unless you guard them with __CUDACC__.

#ifndef TERN_KERNEL_H
#define TERN_KERNEL_H

#include <stdint.h>

// ------------------------------------------------------------
// Packing constants
// ------------------------------------------------------------

// How many ternary weights fit in one uint32.
// 32 bits / 2 bits per weight = 16 weights per word.
#define WEIGHTS_PER_WORD 16

// The 2-bit codes for each ternary value.
// Tried using an enum first but the CUDA compiler got grumpy
// about mixing it with bitwise ops, so plain defines it is.
#define TERN_ZERO     0b00   // 0
#define TERN_POS_ONE  0b01   // +1
#define TERN_NEG_ONE  0b10   // -1


// ------------------------------------------------------------
// Kernel launch parameters
// A simple struct so callers don't have to know the magic
// thread/block numbers.
// ------------------------------------------------------------
typedef struct {
    int block_size;    // threads per block (default 256)
    int rows;          // output features  (K in standard GEMM notation)
    int cols;          // input features   (N)
    int batch;         // number of input vectors (M)
} KernelParams;

// Default params - good starting point on a 3060
static inline KernelParams default_params(int rows, int cols, int batch) {
    KernelParams p;
    p.block_size = 256;
    p.rows  = rows;
    p.cols  = cols;
    p.batch = batch;
    return p;
}


// ------------------------------------------------------------
// C-linkage declarations so the .cpp wrapper can call these
// without name-mangling headaches.
// ------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// Pack a float weight matrix into ternary uint32 words.
// Inputs:
//   weights_fp32  - [K, N] float matrix on GPU
//   packed_out    - [K, ceil(N/16)] uint32 on GPU  (caller allocates)
//   scales_out    - [K] float on GPU               (caller allocates)
//   K, N          - matrix dimensions
void launch_quantize_weights(
    const float*  weights_fp32,
    uint32_t*     packed_out,
    float*        scales_out,
    int K, int N
);

// Forward pass: W1.58 x A8 -> FP16 output
// Inputs:
//   activations   - [M, N] __half on GPU
//   packed_w      - [K, ceil(N/16)] uint32 on GPU
//   scales_w      - [K] float on GPU
//   output        - [M, K] __half on GPU (caller allocates)
//   M, N, K       - matrix dimensions
void launch_bitlinear_forward(
    const void*       activations,   // __half* cast to void* for C linkage
    const uint32_t*   packed_w,
    const float*      scales_w,
    void*             output,        // __half* cast to void* for C linkage
    int M, int N, int K
);

// Utility: decode packed weights back to float for debugging / correctness checks
void launch_decode_weights(
    const uint32_t* packed_w,
    const float*    scales_w,
    float*          decoded_out,
    int K, int N
);

#ifdef __cplusplus
}
#endif

#endif // TERN_KERNEL_H
