#include <torch/extension.h>
#include "../include/tern_kernel.h"

void py_quantize_weights(
    torch::Tensor weights,
    torch::Tensor packed,
    torch::Tensor scales
) {
    TORCH_CHECK(weights.is_cuda(),  "weights must be on CUDA");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");

    int K = weights.size(0);
    int N = weights.size(1);

    launch_quantize_weights(
        weights.data_ptr<float>(),
        (uint32_t*)packed.data_ptr<int32_t>(),
        scales.data_ptr<float>(),
        K, N
    );
}

void py_bitlinear_forward(
    torch::Tensor X,
    torch::Tensor packed_w,
    torch::Tensor scales_w,
    torch::Tensor Y
) {
    TORCH_CHECK(X.is_cuda(),       "X must be on CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat16, "X must be float16");

    int M = X.size(0);
    int N = X.size(1);
    int K = packed_w.size(0);

    launch_bitlinear_forward(
        X.data_ptr<at::Half>(),
        (uint32_t*)packed_w.data_ptr<int32_t>(),
        scales_w.data_ptr<float>(),
        Y.data_ptr<at::Half>(),
        M, N, K
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_weights",  &py_quantize_weights,
          "Quantize FP32 weight matrix to packed ternary");
    m.def("bitlinear_forward", &py_bitlinear_forward,
          "Ternary BitLinear forward pass (W1.58 x A8)");
}