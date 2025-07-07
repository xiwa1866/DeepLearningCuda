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

// ------------------------------------------------------------------------------------------------
// error‐checking macro
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t e = call;                                               \
    if (e != cudaSuccess) {                                             \
      std::cerr << "CUDA error " << cudaGetErrorString(e)               \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

// ------------------------------------------------------------------------------------------------
// 1) time_kernel helper: takes grid, block, a kernel function & its args
template<typename Kernel, typename... Args>
float time_kernel(dim3 grid, dim3 block, Kernel kernel, Args&&... args) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // warm‐up
  kernel<<<grid, block>>>(std::forward<Args>(args)...);
  CUDA_CHECK(cudaDeviceSynchronize());

  // timed launch
  CUDA_CHECK(cudaEventRecord(start));
  kernel<<<grid, block>>>(std::forward<Args>(args)...);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

__global__ void
conv3x3_dense(const float *__restrict__ X,    // [B][C_in][H][W]
              const float *__restrict__ K,    // [C_out][C_in][3][3]
              float *Y,                       // [B][C_out][H][W]
              const float *__restrict__ bias, // [C_out]
              int B, int C_in, int C_out, int H, int W) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= H * W)
    return;

  int w = idx % W;
  int h = idx / W;
  int c = blockIdx.y; // output channel idx
  int b = blockIdx.z;
  float tmp = bias[c];

  // prior knowledge that kernel is always 3x3
  int start_w = w - 1;
  int start_h = h - 1;
  // apply conv2d
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      int d_w = start_w + j;
      int d_h = start_h + i;
      if (d_w >= 0 && d_h >= 0 && d_w < W && d_h < H) {
        for (int c_in = 0; c_in < C_in; c_in++) {
          int kernel_offset = ((c * C_in + c_in) * 3 + i) * 3 + j; // which output idx >> which intput tensor idx >> which row >> which col
          int input_offset = ((b * C_in + c_in) * H + d_h) * W + d_w;
          tmp += K[kernel_offset] * X[input_offset];
        }
      }
    }
  }

  // load back to result memory
  int global_idx = ((b * C_out + c) * H + h) * W + w;
  Y[global_idx] = tmp;
}

__global__ void relu4d(float *Y, // [B][C_out][H][W]
                       int B, int C_out, int H, int W) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= H * W)
    return;
  int h = idx / W;
  int w = idx % W;
  int c = blockIdx.y;
  int b = blockIdx.z;

  int global_idx = ((b * C_out + c) * H + h) * W + w;
  if (Y[global_idx] < 0) {
    Y[global_idx] = 0;
  }
}

// Fused version: conv3x3 + bias + ReLU
__global__ void fusedConvBiasReLU_dense(const float *__restrict__ X,
                                        const float *__restrict__ K,
                                        const float *__restrict__ bias,
                                        float *Y, int B, int C_in, int C_out,
                                        int H, int W) {
                                          int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= H * W)
    return;

  int w = idx % W;
  int h = idx / W;
  int c = blockIdx.y; // output channel idx
  int b = blockIdx.z;
  float tmp = bias[c];

  // prior knowledge that kernel is always 3x3
  int start_w = w - 1;
  int start_h = h - 1;
  // apply conv2d
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      int d_w = start_w + j;
      int d_h = start_h + i;
      if (d_w >= 0 && d_h >= 0 && d_w < W && d_h < H) {
        for (int c_in = 0; c_in < C_in; c_in++) {
          int kernel_offset = ((c * C_in + c_in) * 3 + i) * 3 + j; // which output idx >> which intput tensor idx >> which row >> which col
          int input_offset = ((b * C_in + c_in) * H + d_h) * W + d_w;
          tmp += K[kernel_offset] * X[input_offset];
        }
      }
    }
  }
  if(tmp<0.0f) {
    tmp = 0.0f;
  }
  // load back to result memory
  int global_idx = ((b * C_out + c) * H + h) * W + w;
  Y[global_idx] = tmp;
                                        }

bool verifyCPU(const std::vector<float> &h_X,     // [B*C_in*H*W]
               const std::vector<float> &h_K,     // [C_out*C_in*3*3]
               const std::vector<float> &h_bias,  // [C_out]
               const std::vector<float> &h_Y_gpu, // [B*C_out*H*W]
               int B, int C_in, int C_out, int H, int W) {
  size_t elemsY = size_t(B) * C_out * H * W;
  std::vector<float> h_Y_cpu(elemsY);

  // Zero-padded 3×3 conv + bias + ReLU
  for (int b = 0; b < B; ++b) { // batch
    for (int k = 0; k < C_out; ++k) { // output channel
      for (int h = 0; h < H; ++h) { // height
        for (int w = 0; w < W; ++w) { // weight
          float acc = h_bias[k];
          // sum over input channels and 3×3 window
          for (int c = 0; c < C_in; ++c) { // input channel
            for (int dy = -1; dy <= 1; ++dy) { // kernel
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
  const float eps = 1e-2f;
  for (size_t i = 0; i < elemsY; ++i) {
    if (std::fabs(h_Y_cpu[i] - h_Y_gpu[i]) > eps) {
      std::cerr << "Mismatch at idx " << i << ": CPU=" << h_Y_cpu[i]
                << " GPU=" << h_Y_gpu[i] << "\n";
      return false;
    }
  }
  return true;
}


void one_iteration(int B, int C_in, int C_out, int H, int W, bool verify) {
    // dimensions

  size_t elemsX = size_t(B) * C_in * H * W, elemsY = size_t(B) * C_out * H * W,
         bytesX = elemsX * sizeof(float), bytesY = elemsY * sizeof(float),
         bytesK = size_t(C_out) * C_in * 3 * 3 * sizeof(float),
         bytesB = size_t(C_out) * sizeof(float);

  // host buffers
  std::vector<float> h_X(elemsX), h_Y(elemsY), h_K(bytesK / sizeof(float)),
      h_bias(bytesB / sizeof(float));

  // Initialize input
  // rand engine
  std::mt19937 rng(12345); // fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
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
  dim3 grid(blocks_x, C_out, B);
  dim3 block(threads);

  // ——— Unfused ———
  // 1) Dense 3×3 conv (accumulates bias inside or outside)
  // 2) ReLU
    float t_unfused = time_kernel(
    grid, block,
    conv3x3_dense, d_X, d_K, d_Y, d_bias, B, C_in, C_out, H, W
  );
  // follow it with ReLU (included in timing? up to you)
cudaDeviceSynchronize();
  float t_relu = time_kernel(
    grid, block,
    relu4d, d_Y, B, C_out, H, W
  );
  // copy back to host & verify
  cudaMemcpy(h_Y.data(), d_Y, bytesY, cudaMemcpyDeviceToHost);

  if (verify && !verifyCPU(h_X, h_K, h_bias, h_Y, B, C_in, C_out, H, W)) {
    std::cout << "GPU unfused implemented incorrectly" << std::endl;
  }

  // ——— Fused ———
  float t_fused = time_kernel(
    grid, block,
    fusedConvBiasReLU_dense, d_X, d_K, d_bias, d_Y, B, C_in, C_out, H, W
  );

  cudaMemcpy(h_Y.data(), d_Y, bytesY, cudaMemcpyDeviceToHost);
  if (verify && !verifyCPU(h_X, h_K, h_bias, h_Y, B, C_in, C_out, H, W)) {
    std::cout << "GPU fused implemented incorrectly" << std::endl;
  }
  std::cout
    << "B = " << B
    << "\nC_IN = " << C_in
    << "\nC_OUT = " << C_out
    << "\nH = W = " << H << "\n\n";
  std::cout
    << "Unfused conv: " << t_unfused << " ms\n"
    << "Unfused ReLU: " << t_relu    << " ms\n"
    << "Fused total:  " << t_fused   << " ms\n"
    << "Speedup (conv+relu): " << ((t_unfused + t_relu) / t_fused) << "×\n";

  // cleanup
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_K);
  cudaFree(d_bias);
}
int main() {
  int B[] = {4, 4, 8, 8};
  int C_IN[] = {32, 32, 32, 64};
  int C_OUT[] = {64, 32, 64, 64};
  int length[] = {64, 64, 128, 512};
  for(int i=0; i<4; i++) {
    std::cout << "-------------------------------------------------------------------------------------------------------------------------------\nitr" << i << "\n";
    bool verify = true ? i==0 : false;
    one_iteration(B[i], C_IN[i], C_OUT[i], length[i], length[i], verify);
  }
  return 0;
}
