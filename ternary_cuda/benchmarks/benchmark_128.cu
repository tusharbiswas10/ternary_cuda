// benchmark_128.cu
// Minimal benchmark - only M=128 case for profiling

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/tern_kernel.h"
#include "../include/gpu_utils.h"


__global__ void kernel_fp16_matmul_naive(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*                    C,
    int M, int N, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += __half2float(A[row * N + n]) * __half2float(B[col * N + n]);
    }
    C[row * K + col] = __float2half(sum);
}


static void run_benchmark(int M, int N, int K, int warmup, int iters) {
    __half*   d_X_half   = gpu_alloc<__half>(M * N);
    __half*   d_W_half   = gpu_alloc<__half>(K * N);
    float*    d_W_fp32   = gpu_alloc<float>(K * N);
    int       num_words  = (N + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD;
    uint32_t* d_W_packed = gpu_alloc<uint32_t>(K * num_words);
    float*    d_W_scales = gpu_alloc<float>(K);
    __half*   d_Y_tern   = gpu_alloc<__half>(M * K);
    __half*   d_Y_fp16   = gpu_alloc<__half>(M * K);

    CUDA_CHECK(cudaMemset(d_X_half,   0, M * N * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_W_half,   0, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_W_fp32,   0, K * N * sizeof(float)));

    launch_quantize_weights(d_W_fp32, d_W_packed, d_W_scales, K, N);

    dim3 blk(16, 16);
    dim3 grd((M + 15) / 16, (K + 15) / 16);

    for (int i = 0; i < warmup; i++) {
        kernel_fp16_matmul_naive<<<grd, blk>>>(d_X_half, d_W_half, d_Y_fp16, M, N, K);
    }

    CudaTimer t_fp16;
    t_fp16.start();
    for (int i = 0; i < iters; i++) {
        kernel_fp16_matmul_naive<<<grd, blk>>>(d_X_half, d_W_half, d_Y_fp16, M, N, K);
    }
    float ms_fp16 = t_fp16.stop() / iters;

    for (int i = 0; i < warmup; i++) {
        launch_bitlinear_forward(d_X_half, d_W_packed, d_W_scales, d_Y_tern, M, N, K);
    }

    CudaTimer t_tern;
    t_tern.start();
    for (int i = 0; i < iters; i++) {
        launch_bitlinear_forward(d_X_half, d_W_packed, d_W_scales, d_Y_tern, M, N, K);
    }
    float ms_tern = t_tern.stop() / iters;

    float fp16_weight_mb = (float)(K * N * 2) / 1e6f;
    float tern_weight_mb = (float)(K * num_words * 4) / 1e6f;

    double bytes_moved = (double)(K * num_words * 4)
                       + (double)(M * N * 2) * 2.0
                       + (double)(M * K * 2);
    double bw_used_gbs  = bytes_moved / (ms_tern / 1000.0) / 1e9;
    double bw_peak_gbs  = 336.0;
    double bw_efficiency = bw_used_gbs / bw_peak_gbs * 100.0;

    printf("M=%d N=%d K=%d | FP16: %.2f ms | Tern: %.2f ms | Speedup: %.2fx | BW: %.1f%%\n",
           M, N, K, ms_fp16, ms_tern, ms_fp16 / ms_tern, bw_efficiency);

    gpu_free(d_X_half); gpu_free(d_W_half); gpu_free(d_W_fp32);
    gpu_free(d_W_packed); gpu_free(d_W_scales); gpu_free(d_Y_tern); gpu_free(d_Y_fp16);
}


int main() {
    printf("=== M=128 Profile Run ===\n\n");
    print_gpu_info();
    printf("\n");
    run_benchmark(128, 2048, 1024, 3, 10);
    printf("\n=== Done ===\n");
    return 0;
}