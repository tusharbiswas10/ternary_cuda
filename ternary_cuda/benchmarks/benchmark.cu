// benchmark.cu
// Compares our ternary forward pass against a naive FP16 matmul.
//
// Run this AFTER main.cu tests pass.
// Goal: show that ternary kernel uses less memory bandwidth,
// which is the main win on memory-bound operations.
//
// On an RTX 3060 (336 GB/s memory bandwidth) we expect:
//   FP16 matmul:        ~300-400 GFLOPS (memory bound)
//   Ternary matmul:     faster due to smaller weight footprint
//
// The benchmark prints GB/s of memory bandwidth used,
// which is a better metric than GFLOPS for this type of op.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/tern_kernel.h"
#include "../include/gpu_utils.h"


// ============================================================
// Naive FP16 matmul kernel (baseline to compare against)
// Not optimized - just a straightforward implementation.
// ============================================================
__global__ void kernel_fp16_matmul_naive(
    const __half* __restrict__ A,   // [M, N]
    const __half* __restrict__ B,   // [K, N]  (note: B is row-major, we do A @ B.T)
    __half*                    C,   // [M, K]
    int M, int N, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // M dimension
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // K dimension

    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += __half2float(A[row * N + n]) * __half2float(B[col * N + n]);
    }
    C[row * K + col] = __float2half(sum);
}


// ============================================================
// Run one configuration and print results
// ============================================================
static void run_benchmark(int M, int N, int K, int warmup, int iters) {
    printf("Config: M=%-4d N=%-4d K=%-4d | ", M, N, K);

    // --- Allocate and initialize ---
    __half*   d_X_half   = gpu_alloc<__half>(M * N);
    __half*   d_W_half   = gpu_alloc<__half>(K * N);
    float*    d_W_fp32   = gpu_alloc<float>(K * N);
    int       num_words  = (N + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD;
    uint32_t* d_W_packed = gpu_alloc<uint32_t>(K * num_words);
    float*    d_W_scales = gpu_alloc<float>(K);
    __half*   d_Y_tern   = gpu_alloc<__half>(M * K);
    __half*   d_Y_fp16   = gpu_alloc<__half>(M * K);

    // Fill with random data (just zeros is fine for timing)
    CUDA_CHECK(cudaMemset(d_X_half,   0, M * N * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_W_half,   0, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_W_fp32,   0, K * N * sizeof(float)));

    // Quantize weights once
    launch_quantize_weights(d_W_fp32, d_W_packed, d_W_scales, K, N);

    // --- Benchmark FP16 baseline ---
    dim3 blk(16, 16);
    dim3 grd((M + 15) / 16, (K + 15) / 16);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel_fp16_matmul_naive<<<grd, blk>>>(d_X_half, d_W_half, d_Y_fp16, M, N, K);
    }
    //CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iters
    CudaTimer t_fp16;
    t_fp16.start();
    for (int i = 0; i < iters; i++) {
        kernel_fp16_matmul_naive<<<grd, blk>>>(d_X_half, d_W_half, d_Y_fp16, M, N, K);
    }
    float ms_fp16 = t_fp16.stop() / iters;

    // --- Benchmark ternary forward ---
    for (int i = 0; i < warmup; i++) {
        launch_bitlinear_forward(d_X_half, d_W_packed, d_W_scales, d_Y_tern, M, N, K);
    }

    CudaTimer t_tern;
    t_tern.start();
    for (int i = 0; i < iters; i++) {
        launch_bitlinear_forward(d_X_half, d_W_packed, d_W_scales, d_Y_tern, M, N, K);
    }
    float ms_tern = t_tern.stop() / iters;

    // --- Memory footprint ---
    float fp16_weight_mb = (float)(K * N * 2) / 1e6f;
    float tern_weight_mb = (float)(K * num_words * 4) / 1e6f;

    printf("FP16: %6.2f ms  |  Tern: %6.2f ms  |  Speedup: %.2fx  |  "
           "Weight mem: FP16=%.1fMB Tern=%.1fMB (%.1fx smaller)\n",
           ms_fp16, ms_tern,
           ms_fp16 / ms_tern,
           fp16_weight_mb, tern_weight_mb,
           fp16_weight_mb / tern_weight_mb);

    gpu_free(d_X_half); gpu_free(d_W_half); gpu_free(d_W_fp32);
    gpu_free(d_W_packed); gpu_free(d_W_scales); gpu_free(d_Y_tern); gpu_free(d_Y_fp16);
}


// ============================================================
// MAIN
// ============================================================
int main() {
    printf("=== Ternary vs FP16 Benchmark ===\n\n");
    print_gpu_info();

    printf("Warmup=3 iters, Timed=10 iters\n\n");

    // Sweep over typical transformer FFN sizes
    // These M values represent different batch sizes during inference
    run_benchmark(  1,  256,  128, 3, 10);
    run_benchmark(  1,  512,  256, 3, 10);
    run_benchmark(  1, 1024,  512, 3, 10);
    run_benchmark(  4,  512,  256, 3, 10);
    run_benchmark(  8, 1024,  512, 3, 10);
    run_benchmark( 16, 1024,  512, 3, 10);
    run_benchmark( 32, 2048, 1024, 3, 10);

    // The ternary kernel should shine at small batch sizes (M=1,4)
    // because the bottleneck there is loading weights, not compute.
    // At large M the advantage shrinks.

    printf("\n=== Benchmark done ===\n");
    return 0;
}
