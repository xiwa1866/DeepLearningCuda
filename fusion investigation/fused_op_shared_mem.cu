#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

/*
 Author: Xavier Wang
 Demo: per-batch, per-output-channel 3×3 convolution + bias + ReLU
      – Naïve vs. fused kernels
      – Fused version loads only the 3×3×C_in weight patch into shared memory,
        inputs still from global, bias from constant memory.
*/

// ------------------------------------------------------------------------------------------------
// error‐checking macro
inline void checkCuda(cudaError_t e, const char *file, int line) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA error " << cudaGetErrorString(e) << " at " << file << ":"
              << line << "\n";
    std::exit(1);
  }
}
#define CUDA_CHECK(call) checkCuda((call), __FILE__, __LINE__)

// ------------------------------------------------------------------------------------------------
// bias in constant memory (max 128 channels)
__constant__ float d_b[128];

// ------------------------------------------------------------------------------------------------
// time_kernel: [grid][block][shmem] kernel(args...)
template <typename Kernel, typename... Args>
float time_kernel(dim3 grid, dim3 block, int shmem_bytes, Kernel kernel,
                  Args &&...args) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // warm-up
  kernel<<<grid, block, shmem_bytes>>>(std::forward<Args>(args)...);
  CUDA_CHECK(cudaDeviceSynchronize());

  // timed launch
  CUDA_CHECK(cudaEventRecord(start));
  kernel<<<grid, block, shmem_bytes>>>(std::forward<Args>(args)...);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// ------------------------------------------------------------------------------------------------
// Naïve dense 3×3 convolution
__global__ void conv3x3_dense(const float *__restrict__ X,
                              const float *__restrict__ K, float *Y, int B,
                              int C_in, int C_out, int H, int W) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= H * W)
    return;
  int w = tid % W, h = tid / W;
  int cout = blockIdx.y, b = blockIdx.z;

  float acc = d_b[cout];
  int sw = w - 1, sh = h - 1;
  for (int cin = 0; cin < C_in; ++cin) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        int ww = sw + j, hh = sh + i;
        if (ww >= 0 && hh >= 0 && ww < W && hh < H) {
          int xoff = ((b * C_in + cin) * H + hh) * W + ww;
          int koff = ((cout * C_in + cin) * 3 + i) * 3 + j;
          acc += X[xoff] * K[koff];
        }
      }
    }
  }
  Y[((b * C_out + cout) * H + h) * W + w] = acc;
}

// ------------------------------------------------------------------------------------------------
// Naïve ReLU
__global__ void relu4d(float *Y, int B, int C_out, int H, int W) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= H * W)
    return;
  int w = tid % W, h = tid / W;
  int cout = blockIdx.y, b = blockIdx.z;
  int off = ((b * C_out + cout) * H + h) * W + w;
  float v = Y[off];
  Y[off] = v > 0.f ? v : 0.f;
}

// ------------------------------------------------------------------------------------------------
// Fused conv + bias + ReLU, reading weights from global
__global__ void fusedConvBiasReLU_dense(const float *__restrict__ X,
                                        const float *__restrict__ K, float *Y,
                                        int B, int C_in, int C_out, int H,
                                        int W) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= H * W)
    return;
  int w = tid % W, h = tid / W;
  int cout = blockIdx.y, b = blockIdx.z;

  float acc = d_b[cout];
  int sw = w - 1, sh = h - 1;
  for (int cin = 0; cin < C_in; ++cin) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        int ww = sw + j, hh = sh + i;
        if (ww >= 0 && hh >= 0 && ww < W && hh < H) {
          int xoff = ((b * C_in + cin) * H + hh) * W + ww;
          int koff = ((cout * C_in + cin) * 3 + i) * 3 + j;
          acc += X[xoff] * K[koff];
        }
      }
    }
  }
  acc = acc > 0.f ? acc : 0.f;
  Y[((b * C_out + cout) * H + h) * W + w] = acc;
}

// ------------------------------------------------------------------------------------------------
// Fused conv + bias + ReLU, loading kernels into shared memory
__global__ void fusedConvBiasReLU_sharedK(const float *__restrict__ X,
                                          const float *__restrict__ K, float *Y,
                                          int B, int C_in, int C_out, int H,
                                          int W) {
  extern __shared__ float sK[]; // C_in*3*3 floats
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= H * W)
    return;
  int w = tid % W, h = tid / W;
  int cout = blockIdx.y, b = blockIdx.z;

  // load this cout’s weights
  int totalW = C_in * 3 * 3;
  for (int i = threadIdx.x; i < totalW; i += blockDim.x) {
    sK[i] = K[cout * totalW + i];
  }
  __syncthreads();

  // compute
  float acc = d_b[cout];
  int sw = w - 1, sh = h - 1;
  for (int cin = 0; cin < C_in; ++cin) {
    int base = cin * 9;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        int ww = sw + j, hh = sh + i;
        if (ww >= 0 && hh >= 0 && ww < W && hh < H) {
          int xoff = ((b * C_in + cin) * H + hh) * W + ww;
          acc += X[xoff] * sK[base + i * 3 + j];
        }
      }
    }
  }
  acc = acc > 0.f ? acc : 0.f;
  Y[((b * C_out + cout) * H + h) * W + w] = acc;
}

// ------------------------------------------------------------------------------------------------
// CPU reference
bool verifyCPU(const std::vector<float> &h_X, const std::vector<float> &h_K,
               const std::vector<float> &h_b, const std::vector<float> &h_Y,
               int B, int C_in, int C_out, int H, int W) {
  size_t N = size_t(B) * C_out * H * W;
  std::vector<float> ref(N);
  for (int b = 0; b < B; ++b)
    for (int cout = 0; cout < C_out; ++cout)
      for (int h0 = 0; h0 < H; ++h0)
        for (int w0 = 0; w0 < W; ++w0) {
          float acc = h_b[cout];
          for (int cin = 0; cin < C_in; ++cin)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                int hh = h0 + dy, ww = w0 + dx;
                if (hh >= 0 && ww >= 0 && hh < H && ww < W) {
                  size_t xoff = ((size_t(b) * C_in + cin) * H + hh) * W + ww;
                  size_t koff =
                      ((size_t(cout) * C_in + cin) * 3 + (dy + 1)) * 3 +
                      (dx + 1);
                  acc += h_X[xoff] * h_K[koff];
                }
              }
          ref[((size_t(b) * C_out + cout) * H + h0) * W + w0] =
              std::max(0.f, acc);
        }
  for (size_t i = 0; i < N; ++i)
    if (fabs(ref[i] - h_Y[i]) > 1e-2f)
      return false;
  return true;
}

void one_iteration(int B, int C_in, int C_out, int H, int W, bool verify) {
  size_t Xn = B * size_t(C_in) * H * W, Yn = B * size_t(C_out) * H * W;
  size_t bytesX = Xn * sizeof(float), bytesY = Yn * sizeof(float),
         bytesK = C_out * size_t(C_in) * 3 * 3 * sizeof(float),
         bytesB = C_out * sizeof(float);

  std::vector<float> h_X(Xn), h_K(bytesK / 4), h_b(C_out), h_Y(Yn);
  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> d(-4, 4);
  for (auto &v : h_X)
    v = d(gen);
  for (auto &v : h_K)
    v = d(gen);
  for (auto &v : h_b)
    v = d(gen);

  float *d_X, *d_Y, *d_K;
  CUDA_CHECK(cudaMalloc(&d_X, bytesX));
  CUDA_CHECK(cudaMalloc(&d_Y, bytesY));
  CUDA_CHECK(cudaMalloc(&d_K, bytesK));

  CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), bytesX, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytesK, cudaMemcpyHostToDevice));
  // copy bias into constant memory
  CUDA_CHECK(
      cudaMemcpyToSymbol(d_b, h_b.data(), bytesB, 0, cudaMemcpyHostToDevice));

  int HW = H * W, thr = 256;
  dim3 block(thr), grid((HW + thr - 1) / thr, C_out, B);
  std::vector<float> out(Yn);

  // unfused
  float t1 = time_kernel(grid, block, 0, conv3x3_dense, d_X, d_K, d_Y, B, C_in,
                         C_out, H, W);
  CUDA_CHECK(cudaDeviceSynchronize());
  float t2 = time_kernel(grid, block, 0, relu4d, d_Y, B, C_out, H, W);
  CUDA_CHECK(cudaMemcpy(out.data(), d_Y, bytesY, cudaMemcpyDeviceToHost));
  if (verify && !verifyCPU(h_X, h_K, h_b, out, B, C_in, C_out, H, W))
    std::cerr << "unfused failed\n";

  // fused global weights
  float t3 = time_kernel(grid, block, 0, fusedConvBiasReLU_dense, d_X, d_K, d_Y,
                         B, C_in, C_out, H, W);
  CUDA_CHECK(cudaMemcpy(out.data(), d_Y, bytesY, cudaMemcpyDeviceToHost));
  if (verify && !verifyCPU(h_X, h_K, h_b, out, B, C_in, C_out, H, W))
    std::cerr << "fused-global failed\n";

  // fused shared-K
  int shmem = C_in * 3 * 3 * sizeof(float);
  float t4 = time_kernel(grid, block, shmem, fusedConvBiasReLU_sharedK, d_X,
                         d_K, d_Y, B, C_in, C_out, H, W);
  CUDA_CHECK(cudaMemcpy(out.data(), d_Y, bytesY, cudaMemcpyDeviceToHost));
  if (verify && !verifyCPU(h_X, h_K, h_b, out, B, C_in, C_out, H, W))
    std::cerr << "fused-sharedK failed\n";

  std::cout << "B=" << B << " C_in=" << C_in << " C_out=" << C_out << " H=" << H
            << " W=" << W << "\n"
            << "unf conv:" << t1 << "  relu:" << t2 << "\n"
            << "fused-glob:" << t3 << "\n"
            << "fused-shK:" << t4 << "  speedup:" << t3 / t4 << "×\n\n";

  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_Y));
  CUDA_CHECK(cudaFree(d_K));
  // note: do NOT cudaFree(d_b) – it's constant memory
}

int main() {
  int Bs[] = {4, 4, 8, 8}, CIns[] = {32, 32, 32, 128},
      COs[] = {64, 32, 64, 128}, Ls[] = {64, 64, 128, 512};

  for (int i = 0; i < 4; ++i) {
    std::cout << "--- iter " << i << " ---\n";
    one_iteration(Bs[i], CIns[i], COs[i], Ls[i], Ls[i], i == 0);
  }
  return 0;
}
