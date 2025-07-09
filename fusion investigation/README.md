# 3×3 Dense Convolution + Bias + ReLU (Fused vs. Unfused)

**Author:** Xavier Wang

## Overview
This project demonstrates a simple 3×3 dense convolution followed by bias addition and ReLU activation on a 4D tensor (`B × C_in × H × W`) using CUDA. the base fused_op.cu use a untiled version, loading from DRAM, whereas the shared_mem version utilizes \_\_constant\_\_ memory and \_\_shared\_\_ memory to improve performance. In the end, a Triton and native Pytorch implementation are also provided as benchmarks.

In each file, two implementations are provided:

- **Unfused:** Separate kernel launches for convolution and ReLU.  
- **Fused:** Single kernel that performs convolution, bias add, and ReLU in one pass.

We use a host‐side `time_kernel` helper to measure and compare the latency of each approach, and a CPU reference implementation to verify correctness.

## Dependencies
- CUDA Toolkit (11.0+)  
- CMake (≥3.18)  
- A GPU with Compute Capability ≥ 7.5

## Building
```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
```

## Running
./fused_op

## Results
 - unoptimized loop structure: H->w->input channel
 ```bash
  for (int i = 0; i < 3; i++) 
    for (int j = 0; j < 3; j++) 
        for (int c_in = 0; c_in < C_in; c_in++) 
 ```
```bash
-------------------------------------------------------------------------------------------------------------------------------
itr0
B = 4
C_IN = 32
C_OUT = 64
H = W = 64

Unfused conv: 0.151552 ms
Unfused ReLU: 0.00496 ms
Fused total:  0.150528 ms
Speedup (conv+relu): 1.03975×
-------------------------------------------------------------------------------------------------------------------------------
itr1
B = 4
C_IN = 32
C_OUT = 32
H = W = 64

Unfused conv: 0.083968 ms
Unfused ReLU: 0.004096 ms
Fused total:  0.086016 ms
Speedup (conv+relu): 1.02381×
-------------------------------------------------------------------------------------------------------------------------------
itr2
B = 8
C_IN = 32
C_OUT = 64
H = W = 128

Unfused conv: 1.46125 ms
Unfused ReLU: 0.021504 ms
Fused total:  1.44512 ms
Speedup (conv+relu): 1.02604×
-------------------------------------------------------------------------------------------------------------------------------
itr3
B = 8
C_IN = 64
C_OUT = 64
H = W = 512

Unfused conv: 66.9193 ms
Unfused ReLU: 0.605184 ms
Fused total:  65.2489 ms
Speedup (conv+relu): 1.03487×
```
- Perform loop interchange to use better cache locality and memory coalescing
```bash
for (int c_in = 0; c_in < C_in; c_in++)
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
```

```bash
-------------------------------------------------------------------------------------------------------------------------------
itr0
B = 4
C_IN = 32
C_OUT = 64
H = W = 64

Unfused conv: 0.12288 ms
Unfused ReLU: 0.005376 ms
Fused total:  0.121856 ms
Speedup (conv+relu): 1.05252×
-------------------------------------------------------------------------------------------------------------------------------
itr1
B = 4
C_IN = 32
C_OUT = 32
H = W = 64

Unfused conv: 0.065248 ms
Unfused ReLU: 0.004096 ms
Fused total:  0.064256 ms
Speedup (conv+relu): 1.07918×
-------------------------------------------------------------------------------------------------------------------------------
itr2
B = 8
C_IN = 32
C_OUT = 64
H = W = 128

Unfused conv: 0.915296 ms
Unfused ReLU: 0.021152 ms
Fused total:  0.917632 ms
Speedup (conv+relu): 1.0205×
-------------------------------------------------------------------------------------------------------------------------------
itr3
B = 8
C_IN = 64
C_OUT = 64
H = W = 512

Unfused conv: 30.7866 ms
Unfused ReLU: 0.5888 ms
Fused total:  31.5525 ms
Speedup (conv+relu): 0.994386×
```


- Shared Memory investigation

A optimized version is in fused_op_shared_mem.cu
The convolution kernels are first loaded to SRAM for faster read
The bias values are now also stored in \__constant__ memory because the bias tensor is small
** Note that an earlier attempt was to load input tensor into share_mem but that degrades performance

result is as follows:
```bash
--- iter 0 ---
B=4 C_in=32 C_out=64 H=64 W=64
unf conv:0.121856  relu:0.00512
fused-glob:0.121728
fused-shK:0.118784  speedup:1.02478×

--- iter 1 ---
B=4 C_in=32 C_out=32 H=64 W=64
unf conv:0.062464  relu:0.004096
fused-glob:0.063584
fused-shK:0.06144  speedup:1.0349×

--- iter 2 ---
B=8 C_in=32 C_out=64 H=128 W=128
unf conv:0.91648  relu:0.021408
fused-glob:0.915456
fused-shK:0.927552  speedup:0.986959×

--- iter 3 ---
B=8 C_in=128 C_out=128 H=512 W=512
unf conv:145.617  relu:2.27296
fused-glob:145.568
fused-shK:143.831  speedup:1.01207×
```
