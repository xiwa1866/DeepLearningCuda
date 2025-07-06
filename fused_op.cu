#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

/*
 Author: Xavier Wang
 Demo: per-batch, per-output-channel 3×3 convolution + bias + ReLU
      – Untiled “naïve” conv over all input channels
      – Fused vs. separate kernels
*/

__global__ void
conv3x3_dense(const float *__restrict__ X,    // [B][C_in][H][W]
              const float *__restrict__ K,    // [C_out][C_in][3][3]
              float *Y,                       // [B][C_out][H][W]
              const float *__restrict__ bias, // [C_out]
              int B, int C_in, int C_out, int H, int W);

__global__ void relu4d(float *Y, // [B][C_out][H][W]
                       int B, int C_out, int H, int W);

// Fused version: conv3x3 + bias + ReLU
__global__ void fusedConvBiasReLU_dense(const float *__restrict__ X,
                                        const float *__restrict__ K,
                                        const float *__restrict__ bias,
                                        float *Y, int B, int C_in, int C_out,
                                        int H, int W);

bool verifyCPU(const std::vector<float> &h_X,     // [B*C_in*H*W]
               const std::vector<float> &h_K,     // [C_out*C_in*3*3]
               const std::vector<float> &h_bias,  // [C_out]
               const std::vector<float> &h_Y_gpu, // [B*C_out*H*W]
               int B, int C_in, int C_out, int H, int W) {
  size_t elemsY = size_t(B) * C_out * H * W;
  std::vector<float> h_Y_cpu(elemsY);

  // Zero-padded 3×3 conv + bias + ReLU
  for (int b = 0; b < B; ++b) {
    for (int k = 0; k < C_out; ++k) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          float acc = h_bias[k];
          // sum over input channels and 3×3 window
          for (int c = 0; c < C_in; ++c) {
            for (int dy = -1; dy <= 1; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                int hn = h + dy, wn = w + dx;
                if (0 <= hn && hn < H && 0 <= wn && wn < W) {
                  size_t x_idx = ((size_t(b) * C_in + c) * H + hn) * W + wn;
                  size_t k_idx =
                      ((size_t(k) * C_in + c) * 3 + (dy + 1)) * 3 + (dx + 1);
                  acc += h_X[x_idx] * h_K[k_idx];
                }
              }
            }
          }
          // ReLU
          acc = std::max(0.0f, acc);
          // copy to memory
          size_t y_idx = ((size_t(b) * C_out + k) * H + h) * W + w;
          h_Y_cpu[y_idx] = acc;
        }
      }
    }
  }

  // Compare to GPU output
  const float eps = 1e-4f;
  for (size_t i = 0; i < elemsY; ++i) {
    if (std::fabs(h_Y_cpu[i] - h_Y_gpu[i]) > eps) {
      std::cerr << "Mismatch at idx " << i << ": CPU=" << h_Y_cpu[i]
                << " GPU=" << h_Y_gpu[i] << "\n";
      return false;
    }
  }
  std::cout << "CPU verification passed!\n";
  return true;
}

int main() {
  // dimensions
  int B = 8, C_in = 32, C_out = 32, H = 64, W = 64;

  size_t elemsX = size_t(B) * C_in * H * W, elemsY = size_t(B) * C_out * H * W,
         bytesX = elemsX * sizeof(float), bytesY = elemsY * sizeof(float),
         bytesK = size_t(C_out) * C_in * 3 * 3 * sizeof(float),
         bytesB = size_t(C_out) * sizeof(float);

  // host buffers
  std::vector<float> h_X(elemsX), h_Y(elemsY), h_Y_CPU(elemsY),
      h_K(bytesK / sizeof(float)), h_bias(bytesB / sizeof(float));

  // Initialize input
  // rand engine
  std::mt19937 rng(12345); // fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (int i = 0; i < h_X.size(); i++) {
    h_X[i] = dist(rng);
  }
  for (int i = 0; i < h_Y.size(); i++) {
    h_Y[i] = dist(rng);
  }
  for (int i = 0; i < h_K.size(); i++) {
    h_K[i] = dist(rng);
  }
  for (int i = 0; i < h_bias.size(); i++) {
    h_bias[i] = dist(rng);
  }

  // device buffers
  float *d_X, *d_Y, *d_K, *d_bias;
  cudaMalloc(&d_X, bytesX);
  cudaMalloc(&d_Y, bytesY);
  cudaMalloc(&d_K, bytesK);
  cudaMalloc(&d_bias, bytesB);

  cudaMemcpy(d_X, h_X.data(), bytesX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K.data(), bytesK, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, h_bias.data(), bytesB, cudaMemcpyHostToDevice);

  // build a 3D grid:
  //   grid.x → flatten H×W
  //   grid.y → output channels C_out
  //   grid.z → batch B
  int HW = H * W;
  int threads = 256;
  int blocks_x = (HW + threads - 1) / threads;
  dim3 grid(blocks_x, C_out, B), block(threads);

  // ——— Unfused ———
  // // 1) Dense 3×3 conv (accumulates bias inside or outside)
  // conv3x3_dense<<<grid,block>>>(d_X, d_K, d_Y, d_bias,
  //                               B, C_in, C_out, H, W);
  // 2) ReLU
  relu4d<<<grid,block>>>(d_Y, B, C_out, H, W);

  // // copy back to host & time/verify
  // cudaMemcpy(h_Y.data(), d_Y, bytesY, cudaMemcpyDeviceToHost);

  // ——— Fused ———
  // fusedConvBiasReLU_dense<<<grid,block>>>(
  //     d_X, d_K, d_bias, d_Y,
  //     B, C_in, C_out, H, W);

  cudaMemcpy(h_Y.data(), d_Y, bytesY, cudaMemcpyDeviceToHost);
  if (verifyCPU(h_X, h_K, h_bias, h_Y_CPU, B, C_in, C_out, H, W)) {
    std::cout << "GPU implemented correctly" << std::endl;
  }

  // cleanup
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_K);
  cudaFree(d_bias);
  return 0;
}
