# 3×3 Dense Convolution + Bias + ReLU (Fused vs. Unfused)

**Author:** Xavier Wang

## Overview
This project demonstrates a simple 3×3 dense convolution followed by bias addition and ReLU activation on a 4D tensor (`B × C_in × H × W`) using CUDA. Two implementations are provided:

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

## Running
./fused_op
