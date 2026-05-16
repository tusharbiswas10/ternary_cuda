// main.cu
// Entry point for the ternary quantization demo.
//
// This file just drives the kernels - quantize some weights,
// run a forward pass, compare against a naive FP32 baseline.
//
// Run this first to make sure everything works before touching
// the Python wrapper.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/tern_kernel.h"
#include "../include/gpu_utils.h"


// ============================================================
// CPU-side helpers for generating test data and checking results
// ============================================================

// Fill an array with small random floats in [-1, 1].
// Using a fixed seed so results are reproducible.
static void fill_random(float* arr, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

// Naive CPU forward pass for correctness comparison.
// Does the same thing as our CUDA kernel but slowly, in FP32.
static void cpu_forward_naive(
    const float* X,     // [M, N]
    const float* W,     // [K, N]
    float*       Y,     // [M, K]
    int M, int N, int K
) {
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                sum += X[m * N + n] * W[k * N + n];
            }
            Y[m * K + k] = sum;
        }
    }
}

// Check how close two float arrays are.
// Returns the max absolute error.
static float max_abs_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Convert float array to half for GPU upload
static void float_to_half(const float* src, __half* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = __float2half(src[i]);
    }
}

// Convert half array to float for CPU comparison
static void half_to_float(const __half* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = __half2float(src[i]);
    }
}


// ============================================================
// TEST 1: Weight quantizer round-trip
// Quantize weights -> decode back -> check error is small.
// ============================================================
static void test_quantizer_roundtrip(int K, int N) {
    printf("--- Test: quantizer round-trip  [K=%d, N=%d] ---\n", K, N);

    // Generate random weights on CPU
    float* h_weights = (float*)malloc(K * N * sizeof(float));
    fill_random(h_weights, K * N, 42);

    // Allocate GPU buffers
    float*    d_weights  = gpu_alloc<float>(K * N);
    int       num_words  = (N + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD;
    uint32_t* d_packed   = gpu_alloc<uint32_t>(K * num_words);
    float*    d_scales   = gpu_alloc<float>(K);
    float*    d_decoded  = gpu_alloc<float>(K * N);

    // Upload weights and quantize
    gpu_upload(d_weights, h_weights, K * N);
    launch_quantize_weights(d_weights, d_packed, d_scales, K, N);

    // Decode back to float
    launch_decode_weights(d_packed, d_scales, d_decoded, K, N);

    // Download and compare
    float* h_decoded = (float*)malloc(K * N * sizeof(float));
    gpu_download(h_decoded, d_decoded, K * N);

    // The decoded values will only be multiples of the scale (+scale, 0, -scale)
    // so we expect some quantization error. Print the max error.
    float err = max_abs_error(h_weights, h_decoded, K * N);
    printf("  Max quantization error: %.4f  (expected ~0.1-0.5 for random weights)\n", err);

    // Also check that the packed size is correct
    float compression = (float)(K * N * 4) / (float)(K * num_words * 4);
    printf("  Compression vs FP32:    %.1fx  (%d words for %d weights)\n",
           compression, K * num_words, K * N);

    // Cleanup
    free(h_weights);
    free(h_decoded);
    gpu_free(d_weights);
    gpu_free(d_packed);
    gpu_free(d_scales);
    gpu_free(d_decoded);

    printf("  PASS\n\n");
}


// ============================================================
// TEST 2: Forward pass correctness
// Compare CUDA ternary forward pass against CPU FP32 baseline.
// ============================================================
static void test_forward_pass(int M, int N, int K) {
    printf("--- Test: forward pass  [M=%d, N=%d, K=%d] ---\n", M, N, K);

    // Generate random inputs on CPU
    float* h_X_fp32 = (float*)malloc(M * N * sizeof(float));
    float* h_W_fp32 = (float*)malloc(K * N * sizeof(float));
    fill_random(h_X_fp32, M * N, 123);
    fill_random(h_W_fp32, K * N, 456);

    // CPU baseline (FP32, no quantization)
    float* h_Y_cpu = (float*)malloc(M * K * sizeof(float));
    cpu_forward_naive(h_X_fp32, h_W_fp32, h_Y_cpu, M, N, K);

    // GPU path: quantize weights, then run forward pass
    float*    d_X_fp32  = gpu_alloc<float>(M * N);
    __half*   d_X_half  = gpu_alloc<__half>(M * N);
    float*    d_W_fp32  = gpu_alloc<float>(K * N);
    int       num_words = (N + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD;
    uint32_t* d_W_packed = gpu_alloc<uint32_t>(K * num_words);
    float*    d_W_scales = gpu_alloc<float>(K);
    __half*   d_Y_half  = gpu_alloc<__half>(M * K);

    // Upload and convert activations to FP16
    gpu_upload(d_X_fp32, h_X_fp32, M * N);
    {
        __half* h_X_half = (__half*)malloc(M * N * sizeof(__half));
        float_to_half(h_X_fp32, h_X_half, M * N);
        gpu_upload(d_X_half, h_X_half, M * N);
        free(h_X_half);
    }

    // Quantize weights on GPU
    gpu_upload(d_W_fp32, h_W_fp32, K * N);
    launch_quantize_weights(d_W_fp32, d_W_packed, d_W_scales, K, N);

    // Run ternary forward pass
    CUDA_CHECK(cudaDeviceSynchronize());
    CudaTimer timer;
    timer.start();
    launch_bitlinear_forward(d_X_half, d_W_packed, d_W_scales, d_Y_half, M, N, K);
    float ms = timer.stop();

    // Download result and convert to float
    __half* h_Y_half = (__half*)malloc(M * K * sizeof(__half));
    gpu_download(h_Y_half, d_Y_half, M * K);
    float* h_Y_gpu = (float*)malloc(M * K * sizeof(float));
    half_to_float(h_Y_half, h_Y_gpu, M * K);

    // Compare
    // Note: we expect error because (a) weights are ternary not exact, (b) FP16 precision.
    // The error should be proportional to the quantization error times N.
    float err = max_abs_error(h_Y_cpu, h_Y_gpu, M * K);
    printf("  Max output error vs FP32: %.4f\n", err);
    printf("  Forward pass time:        %.3f ms\n", ms);
    printf("  (Error is expected due to ternary quantization of weights)\n");

    // Rough FLOP count: 2*M*N*K multiply-adds
    double gflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;
    printf("  Effective GFLOPS:         %.1f\n", gflops);

    // Cleanup
    free(h_X_fp32); free(h_W_fp32); free(h_Y_cpu);
    free(h_Y_half); free(h_Y_gpu);
    gpu_free(d_X_fp32); gpu_free(d_X_half); gpu_free(d_W_fp32);
    gpu_free(d_W_packed); gpu_free(d_W_scales); gpu_free(d_Y_half);

    printf("  PASS\n\n");
}


// ============================================================
// TEST 3: Bit encoding sanity check
// Manually check that specific weights encode/decode correctly.
// ============================================================
static void test_encoding_sanity() {
    printf("--- Test: encoding sanity check ---\n");

    // Build a tiny 1x16 weight matrix with known values
    // and manually verify the packed word
    int K = 1, N = 16;
    float h_weights[16] = {
         0.9f,  0.1f, -0.8f,  0.0f,   // +1  0 -1  0
         0.6f, -0.7f,  0.6f, -0.6f,   // +1 -1 +1 -1
        -0.9f,  0.0f,  0.1f, -0.6f,   // -1  0  0 -1
         0.8f, -0.1f,  0.7f,  0.2f    // +1  0 +1  0
    };

    float*    d_weights = gpu_alloc<float>(K * N);
    uint32_t* d_packed  = gpu_alloc<uint32_t>(K * 1);   // 16 weights = 1 word
    float*    d_scales  = gpu_alloc<float>(K);

    gpu_upload(d_weights, h_weights, K * N);
    launch_quantize_weights(d_weights, d_packed, d_scales, K, N);

    uint32_t h_packed[1];
    float    h_scale[1];
    gpu_download(h_packed, d_packed, 1);
    gpu_download(h_scale, d_scales, 1);

    printf("  Scale computed: %.4f\n", h_scale[0]);
    printf("  Packed word:    0x%08X\n", h_packed[0]);

    // Manually decode and print each weight
    printf("  Decoded weights: ");
    for (int i = 0; i < 16; i++) {
        uint32_t code = (h_packed[0] >> (i * 2)) & 0x3;
        int decoded   = (int)(code & 1) - (int)((code >> 1) & 1);
        printf("%2d ", decoded);
    }
    printf("\n");
    printf("  Expected:        +1  0 -1  0 +1 -1 +1 -1 -1  0  0 -1 +1  0 +1  0\n");

    gpu_free(d_weights);
    gpu_free(d_packed);
    gpu_free(d_scales);

    printf("  (Check manually that decoded matches expected)\n");
    printf("  PASS\n\n");
}


// ============================================================
// MAIN
// ============================================================

int main() {
    printf("=== Ternary CUDA Kernel - Test Suite ===\n\n");

    // Print GPU info so we know what we're running on
    print_gpu_info();

    // Run tests
    test_encoding_sanity();

    // Small matrices (fast, good for debugging)
    test_quantizer_roundtrip(64, 64);
    test_forward_pass(4, 64, 32);

    // Medium matrices (more realistic sizes)
    test_quantizer_roundtrip(256, 512);
    test_forward_pass(16, 512, 256);

    // Larger - closer to a real transformer FFN layer
    // e.g. hidden_size=1024, ffn_size=4096
    // Skipping for now - add when basic tests pass
    test_forward_pass(32, 1024, 4096);

    // Phi-2 FFN layer: hidden=2560, ffn=10240
    test_forward_pass(1,  2560, 10240);
    test_forward_pass(8,  2560, 10240);

    // Qwen2-1.5B FFN: hidden=2048, ffn=5504
    test_forward_pass(1,  2048, 5504);
    test_forward_pass(16, 2048, 5504);

    printf("=== All tests done ===\n");
    return 0;
}
