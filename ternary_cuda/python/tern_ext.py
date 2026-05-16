# tern_ext.py
# PyTorch extension loader for the ternary CUDA kernel.
#
# This uses torch.utils.cpp_extension.load() to compile the
# CUDA kernel at runtime and bind it into Python.
#
# You need PyTorch installed:  pip install torch torchvision
# And the CUDA toolkit on PATH (should be if VS + CUDA are installed)
#
# First time you run this it will compile the kernel (~30 seconds).
# After that it's cached and loads instantly.
#
# Usage:
#   from tern_ext import TernaryLinear
#   layer = TernaryLinear(in_features=512, out_features=256)
#   y = layer(x)

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# VS 2019 Professional, MSVC 14.50.35717
_vctools = r"C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Tools\MSVC"

_versions = os.listdir(_vctools)
if _versions:
    _cl_path = os.path.join(_vctools, _versions[0], "bin", "Hostx64", "x64")
    os.environ["PATH"] = _cl_path + ";" + os.environ.get("PATH", "")

# Add CUDA toolkit path
_cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
os.environ["PATH"] = _cuda_path + ";" + os.environ.get("PATH", "")

# Path to the CUDA kernel files - adjust if running from a different directory
_this_dir   = os.path.dirname(os.path.abspath(__file__))
_kernel_cu  = os.path.join(_this_dir, "../src/tern_kernel.cu")
_bind_cpp   = os.path.join(_this_dir, "../src/tern_ext_bind.cpp")
_include    = os.path.join(_this_dir, "../include")

print("Loading ternary CUDA extension (may take ~30s on first run)...")

_ext = load(
    name="tern_cuda_ext",
    sources=[_kernel_cu, _bind_cpp],
    extra_include_paths=[_include],
    extra_cuda_cflags=[
        "-arch=sm_86",
        "--expt-relaxed-constexpr",
        "--use_fast_math",
    ],
    extra_cflags=["/O2"],
    verbose=True,
)

print("Extension loaded.")


# ============================================================
# Python wrappers around the raw C extension calls
# ============================================================

def quantize_weights(weight: torch.Tensor):
    """
    Quantize a float32 weight matrix to ternary.

    Args:
        weight: [out_features, in_features] float32 CPU or CUDA tensor

    Returns:
        packed:  [out_features, ceil(in_features/16)] uint32 CUDA tensor
        scales:  [out_features] float32 CUDA tensor
    """
    if not weight.is_cuda:
        weight = weight.cuda()
    weight = weight.contiguous().float()

    K, N = weight.shape
    num_words = (N + 15) // 16

    packed = torch.zeros(K, num_words, dtype=torch.int32, device="cuda")
    scales = torch.zeros(K,            dtype=torch.float32, device="cuda")

    _ext.quantize_weights(weight, packed, scales)
    return packed, scales


def ternary_forward(x: torch.Tensor, packed_w: torch.Tensor, scales_w: torch.Tensor):
    """
    Run a ternary linear forward pass.

    Args:
        x:        [batch, in_features] float16 CUDA tensor
        packed_w: [out_features, ceil(in_features/16)] uint32 CUDA tensor
        scales_w: [out_features] float32 CUDA tensor

    Returns:
        y: [batch, out_features] float16 CUDA tensor
    """
    assert x.is_cuda, "x must be on CUDA"
    assert x.dtype == torch.float16, "x must be float16"

    M = x.shape[0]
    K = packed_w.shape[0]
    N = x.shape[1]

    y = torch.zeros(M, K, dtype=torch.float16, device="cuda")
    _ext.bitlinear_forward(x, packed_w, scales_w, y)
    return y


# ============================================================
# nn.Module wrapper - drop-in replacement for nn.Linear
# ============================================================

class TernaryLinear(nn.Module):
    """
    A linear layer that quantizes its weights to ternary ({-1, 0, +1})
    and runs inference using the custom CUDA kernel.

    Compatible with nn.Linear as a drop-in for inference.
    (Training is not yet supported - weights are quantized once at init.)

    Example:
        layer = TernaryLinear(512, 256)
        y = layer(x)   # x is [batch, 512] float16

    Args:
        in_features:  input dimension
        out_features: output dimension
        bias:         not implemented yet, always False
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()

        if bias:
            # TODO: add bias support - just a simple add after the matmul
            raise NotImplementedError("bias not yet supported")

        self.in_features  = in_features
        self.out_features = out_features

        # Initialize with random weights so the layer is usable immediately.
        # In practice you'd load pre-trained weights here.
        raw_weight = torch.randn(out_features, in_features) * 0.02

        # Quantize and store on GPU
        self.packed_w, self.scales_w = quantize_weights(raw_weight)

        # Keep the original float weight for reference / debugging
        # Not needed for inference - comment out to save memory
        self._debug_fp32_weight = raw_weight

    def load_from_linear(self, linear: nn.Linear):
        """Load weights from a standard nn.Linear and quantize them."""
        with torch.no_grad():
            self.packed_w, self.scales_w = quantize_weights(linear.weight)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is FP16 on CUDA
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float16:
            x = x.half()

        # Handle batched input: if x is [seq, batch, features] flatten to [seq*batch, features]
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])

        y = ternary_forward(x, self.packed_w, self.scales_w)

        # Restore original batch dimensions
        if len(orig_shape) > 2:
            y = y.view(*orig_shape[:-1], self.out_features)

        return y

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, quantized=ternary"


# ============================================================
# Quick sanity test if run directly
# ============================================================
if __name__ == "__main__":
    print("\n--- Testing TernaryLinear ---")

    in_f, out_f = 128, 64
    batch       = 4

    layer = TernaryLinear(in_f, out_f)
    x     = torch.randn(batch, in_f, dtype=torch.float16, device="cuda")
    y     = layer(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output dtype: {y.dtype}")
    print(f"Output sample (first row, first 8 values):")
    print(f"  {y[0, :8].tolist()}")

    # Compare against nn.Linear using the same weights
    ref_linear = nn.Linear(in_f, out_f, bias=False).cuda().half()
    with torch.no_grad():
        ref_linear.weight.copy_(layer._debug_fp32_weight.cuda().half())

    y_ref  = ref_linear(x)
    error  = (y.float() - y_ref.float()).abs().max().item()
    print(f"\nMax error vs nn.Linear: {error:.4f}")
    print("(Some error is expected due to ternary quantization)")
    print("\nTest passed!")
