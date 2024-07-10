#include <cuda_runtime.h>
#include <iostream>
#include <random>
std::mt19937 rd(114514);
std::normal_distribution<float> dist;
__global__ void cuda_matmul(const float *A, const float *B, size_t m, size_t n, size_t K, float *C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i < m && j < K && k < n)
		atomicAdd(&C[i * n + k], A[i * K + j] * B[j * n + k]);
}
void matmul(const float *A, const float *B, size_t m, size_t n, size_t k, float *output) {
	float *cuda_A, *cuda_B, *cuda_C;
	cudaMalloc((void **)&cuda_A, m * k * sizeof(float));
	cudaMalloc((void **)&cuda_B, k * n * sizeof(float));
	cudaMalloc((void **)&cuda_C, m * n * sizeof(float));
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			output[i * n + j] = 0;
	cudaMemcpy(cuda_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_C, output, m * n * sizeof(float), cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(8, 8, 4);
	dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (k + threadsPerBlock.y - 1) / threadsPerBlock.y,
				   (n + threadsPerBlock.z - 1) / threadsPerBlock.z);
	cuda_matmul<<<numBlocks, threadsPerBlock>>>(cuda_A, cuda_B, m, n, k, cuda_C);
	cudaMemcpy(output, cuda_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(cuda_A);
	cudaFree(cuda_B);
	cudaFree(cuda_C);
}
int main() {
	int m = 2048, k = 2048, n = 2048;
	float *A, *B, *C;
	A = (float *)malloc(m * k * sizeof(float));
	B = (float *)malloc(k * n * sizeof(float));
	C = (float *)malloc(m * n * sizeof(float));
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			A[i * k + j] = dist(rd);
		}
	}
	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < n; ++j) {
			B[i * n + j] = dist(rd);
		}
	}
	matmul(A, B, m, n, k, C);
	for (int i = 0; i < min(m, 10); ++i) {
		for (int j = 0; j < min(n, 10); ++j)
			std::cout << C[i * n + j] << ' ';
		std::cout << '\n';
	}
	return 0;
}