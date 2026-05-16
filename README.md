# Ternary CUDA Kernel — 1.58-bit Weight Quantization

A from-scratch implementation of ternary weight quantization with a custom CUDA kernel,
PyTorch extension wrapper, and validation on real LLM weights.

Built as a learning project to understand GPU memory hierarchy, warp primitives,
and low-bit inference — starting from zero CUDA knowledge.

**Hardware:** RTX 3060 12GB (sm_86, Ampere)  
**Stack:** CUDA C++, Visual Studio 2019, PyTorch C++ extension  
**Validated on:** TinyLlama-1.1B-Chat

---

## What this does

Neural network weights are normally stored as FP16 — 2 bytes per value.
This project compresses them to ternary: {-1, 0, +1}, encoded as 2 bits,
with 16 weights packed into a single `uint32`.

```
FP32:    4 bytes/weight   (baseline)
FP16:    2 bytes/weight   (standard inference)
Ternary: 0.25 bytes/weight  ← this project  (8x vs FP16, 16x vs FP32)
```

For a 7B parameter model:
```
FP16:    ~14 GB  (doesn't fit on RTX 3060)
Ternary: ~1.75 GB  (fits with 10 GB to spare)
```

---

## Results

### On real TinyLlama-1.1B weights (FFN layer 0, shape 5632×2048)

**Memory compression:**
| Format   | Size      | vs FP16 |
|----------|-----------|---------|
| FP32     | 46.1 MB   | —       |
| FP16     | 23.1 MB   | 1x      |
| Ternary  | 2.88 MB   | **8x smaller** |

**Inference latency and throughput:**
| Batch | Time    | GFLOPS |
|-------|---------|--------|
| 1     | 0.30 ms | 76.7   |
| 4     | 0.30 ms | 307.6  |
| 8     | 0.30 ms | 616.6  |
| 16    | 0.40 ms | 921.6  |
| 32    | 1.00 ms | 737.8  |

### Ternary vs naive FP16 matmul (same hardware)

| Batch | FP16    | Ternary | Speedup |
|-------|---------|---------|---------|
| 1     | 0.29 ms | 0.16 ms | 1.83x   |
| 8     | 0.28 ms | 0.20 ms | 1.40x   |
| 16    | 0.50 ms | 0.20 ms | 2.46x   |
| 32    | 0.92 ms | 0.30 ms | 3.02x   |
| 128   | 3.30 ms | 1.07 ms | 3.09x   |

Crossover point: ~M=4. Below that, decode overhead dominates.
Above M=4, the 8x smaller weight footprint compounds and ternary pulls ahead.

Note: FP16 baseline is a naive unoptimized kernel. cuBLAS would be faster.
The real value proposition is memory compression, not raw speed.

---

## How it works

### Encoding

```
2 bits per weight:
  0b00 →  0   (zero)
  0b01 → +1   (positive)
  0b10 → -1   (negative)
  0b11 →  0   (reserved)

16 weights packed per uint32.
Decode is branchless: w = bit0 - bit1
```

### Quantization (per output row)

```
1. Compute absmean scale:  s = mean(|W|) + eps
2. Normalize:              W_norm = W / s
3. Threshold:              W_tern = sign(W_norm) if |W_norm| > 0.5 else 0
4. Pack:                   16 values → 1 uint32
```

### Forward pass

```
X (FP16) → quantize to INT8 per row (scale = absmax/127)
W (ternary packed) → decode 2-bit codes on the fly
Accumulate: INT32 dot product
Dequantize: result * act_scale * weight_scale → FP16 output
```

---

## File structure

```
ternary_cuda/
├── include/
│   ├── tern_kernel.h      API declarations, encoding constants
│   └── gpu_utils.h        CUDA_CHECK macro, CudaTimer, gpu_alloc helpers
├── src/
│   ├── tern_kernel.cu     All CUDA kernels (quantizer, forward, decode)
│   ├── main.cu            Test suite entry point
│   └── tern_ext_bind.cpp  PyTorch C++ extension bridge
├── benchmarks/
│   └── benchmark.cu       Ternary vs FP16 timing comparison
└── python/
    ├── tern_ext.py        PyTorch extension loader + TernaryLinear nn.Module
    └── test_real_model.py Validation on TinyLlama-1.1B weights
```

---

## Build — Visual Studio

Requirements (install in this order):
1. Visual Studio 2019/2022 with "Desktop development with C++" workload
2. CUDA Toolkit 12.x or 13.x
3. Python 3.10+ with PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu128`)

**C++ tests and benchmarks:**
Open `ternary_cuda.slnx` in Visual Studio.
Set configuration to `Release | x64`.
Build and run `ternary_cuda` project for tests, `ternary_benchmark` for benchmarks.

Important: set CUDA code generation to `compute_86,sm_86` for RTX 3060.
For other GPUs: RTX 20xx → sm_75, RTX 40xx → sm_89, A100 → sm_80.

**Python extension:**
```bat
cd python
python tern_ext.py          # compiles kernel + runs basic test (~30s first run)
python test_real_model.py   # downloads TinyLlama and runs on real weights
```

---

## Optimizations applied (in order of impact)

**1. Removed cudaDeviceSynchronize from kernel launcher — 26x speedup**  
The kernel was fast. The CPU was waiting for GPU acknowledgement after every
launch. Removing the sync and letting CUDA events handle timing was the
single biggest improvement.

**2. Shared memory tiling (16×16)**  
Threads cooperate to load activation tiles into fast on-chip shared memory
instead of each thread independently reading from slow global memory.

**3. Warp shuffle absmax reduction**  
Replaced serialized atomicMax with `__shfl_down_sync` tree reduction.
Register-to-register communication within the warp, no shared memory needed.

**4. Combined single-pass activation read**  
Originally activations were read twice per tile: once for absmax, once for
quantization. Merged into a single read — values stay in registers between
the two operations.

**5. Branchless ternary decode**  
```cuda
uint32_t code = (word >> (bit * 2)) & 0x3;
int w_q = (int)(code & 1) - (int)((code >> 1) & 1);
```
No branches, no lookup table. Compiles to two shifts and a subtract.

**6. #pragma unroll on inner dot product loop**  
The 16-iteration inner loop is fully unrolled by the compiler,
eliminating loop counter overhead and enabling better register scheduling.

---

## Things learned the hard way

**Out-of-bounds threads cannot early-return before `__syncthreads()`.**  
If one thread in a block exits early, the remaining threads wait forever
for a sync that never comes. The block deadlocks. Use a `bool valid` flag
and guard the output write instead.

**`__syncthreads()` is needed both before AND after shared memory use.**  
One sync after loading ensures data is ready before computation starts.
A second sync after computation ensures nobody overwrites shared memory
before slower threads finish reading it.

**The absmax must track across all tiles, not just the last one.**  
During tiling, each tile computes absmax for its 16-column slice.
The dequantization scale must be the maximum across the entire row,
so the running max must be accumulated in a register across all tile iterations.

---

## What's not done yet

- **QAT (quantization-aware training):** weights currently quantized post-training.
  QAT would fine-tune with the quantization in the loop for much better accuracy.
- **Dense model support:** currently validated on FFN layers. Attention projections
  work the same way but haven't been benchmarked separately.
- **Vectorized float4 loads:** loading 8 FP16 values per instruction instead of 2
  would better utilize the 128-bit memory bus.
- **Bias support in TernaryLinear:** straightforward addition, not yet implemented.

---

## References

- BitNet paper (1.58-bit weights): https://arxiv.org/abs/2402.17764
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PyTorch C++ Extension docs: https://pytorch.org/docs/stable/cpp_extension.html

---

## License

Apache 2.0
