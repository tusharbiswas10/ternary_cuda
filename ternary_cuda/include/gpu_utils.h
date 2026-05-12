#pragma once

// gpu_utils.h
// Small helpers for GPU memory management and error checking.
//
// I kept copy-pasting cudaMalloc + error checks everywhere so I
// just put them here. Nothing fancy.

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ------------------------------------------------------------
// Error check macro
// Wraps any CUDA call and prints file/line info if it fails.
// Usage:  CUDA_CHECK(cudaMalloc(...));
// ------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr,                                                     \
                "[CUDA ERROR] %s  (line %d in %s)\n"                           \
                "  -> %s\n",                                                    \
                #call, __LINE__, __FILE__,                                      \
                cudaGetErrorString(_err));                                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


// ------------------------------------------------------------
// GPU buffer helpers
// These just save typing and make the code read more clearly.
// ------------------------------------------------------------

// Allocate a typed buffer on the GPU.
// Returns a pointer to device memory.
template<typename T>
static inline T* gpu_alloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&ptr, count * sizeof(T)));
    return ptr;
}

// Copy host -> device
template<typename T>
static inline void gpu_upload(T* dst_device, const T* src_host, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst_device, src_host, count * sizeof(T), cudaMemcpyHostToDevice));
}

// Copy device -> host
template<typename T>
static inline void gpu_download(T* dst_host, const T* src_device, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst_host, src_device, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// Free and null the pointer
template<typename T>
static inline void gpu_free(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}


// ------------------------------------------------------------
// GPU info printer - useful at startup to confirm the right
// device is being used.
// ------------------------------------------------------------
static inline void print_gpu_info() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("=== GPU Info ===\n");
    printf("  Device name     : %s\n", prop.name);
    printf("  Compute cap     : %d.%d\n", prop.major, prop.minor);
    printf("  Global memory   : %.0f MB\n", prop.totalGlobalMem / 1e6);
    printf("  Shared mem/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Max threads/blk : %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size       : %d\n", prop.warpSize);
    printf("  SM count        : %d\n", prop.multiProcessorCount);
    printf("================\n\n");
}


// ------------------------------------------------------------
// Simple timer using CUDA events
// Usage:
//   CudaTimer t;
//   t.start();
//   launch_something<<<...>>>(...);
//   float ms = t.stop();
// ------------------------------------------------------------
struct CudaTimer {
    cudaEvent_t _start, _stop;

    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }

    ~CudaTimer() {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(_start));
    }

    float stop() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(_stop));
        CUDA_CHECK(cudaEventSynchronize(_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, _start, _stop));
        return ms;
    }
};

#endif // GPU_UTILS_H
