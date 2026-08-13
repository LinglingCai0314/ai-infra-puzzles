#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) do {                                              \
  cudaError_t status = (call);                                             \
  if (status != cudaSuccess) {                                             \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                 cudaGetErrorString(status));                              \
    return 1;                                                              \
  }                                                                        \
} while (0)

__global__ void vector_affine(const float* x, float* y, int n,
                              float scale, float bias) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) y[index] = x[index] * scale + bias;
}

int launch_vector_affine(const float* x, float* y, int n,
                         float scale, float bias, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vector_affine<<<blocks, threads, 0, stream>>>(x, y, n, scale, bias);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}
