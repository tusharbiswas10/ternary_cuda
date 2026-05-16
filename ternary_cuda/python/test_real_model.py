import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tern_ext import quantize_weights, ternary_forward

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

print("Loading TinyLlama (downloads ~2.2GB on first run)...")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float32
)
model.eval()

layer   = model.model.layers[0]
up_proj = layer.mlp.up_proj.weight.cuda()

print(f"\nWeight shape : {up_proj.shape}")
print(f"Weight dtype : {up_proj.dtype}")

print("\nQuantizing with ternary kernel...")
t0 = time.time()
packed, scales = quantize_weights(up_proj)
t1 = time.time()
print(f"Quantization time: {(t1-t0)*1000:.1f} ms")

fp32_mb    = up_proj.numel() * 4 / 1e6
fp16_mb    = up_proj.numel() * 2 / 1e6
ternary_mb = packed.numel()  * 4 / 1e6

print(f"\nMemory footprint for this layer's weights:")
print(f"  FP32    : {fp32_mb:.1f} MB")
print(f"  FP16    : {fp16_mb:.1f} MB")
print(f"  Ternary : {ternary_mb:.2f} MB  ({fp16_mb/ternary_mb:.1f}x smaller than FP16)")

print("\nRunning forward pass...")
x = torch.randn(1, 2048, dtype=torch.float16, device="cuda")
y = ternary_forward(x, packed, scales)
print(f"Input  shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Output sample (first 6 values): {y[0, :6].tolist()}")

print("\nLatency vs batch size (real TinyLlama weights):")
print(f"{'Batch':>6} | {'Time':>8} | {'GFLOPS':>8}")
print("-" * 32)

for M in [1, 4, 8, 16, 32]:
    x_batch = torch.randn(M, 2048, dtype=torch.float16, device="cuda")
    y_batch = torch.zeros(M, 5632, dtype=torch.float16, device="cuda")

    for _ in range(3):
        ternary_forward(x_batch, packed, scales)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(10):
        ternary_forward(x_batch, packed, scales)
    torch.cuda.synchronize()
    t1 = time.time()

    ms      = (t1 - t0) / 10 * 1000
    gflops  = (2 * M * 2048 * 5632) / (ms / 1000) / 1e9
    print(f"{M:>6} | {ms:>7.3f}ms | {gflops:>7.1f}")

print("\nDone. Real model test complete.")