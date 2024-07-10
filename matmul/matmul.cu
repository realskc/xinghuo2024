#include <cuda_runtime.h>
#include <iostream>
#include <random>
std::mt19937 rd(114514);
std::normal_distribution<float> dist;
// constexpr int batchSize = 128; // 16: 1140ms, 32: 1100ms, 64: 1090ms, 128: 1070ms, 256: 1070ms
__global__ void cuda_matmul(const float *A, const float *B, size_t m, size_t n, size_t K, float *C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float tmp = 0;
	if (i < m && j < n) {
		for (int k = 0; k < K; ++k)
			tmp += A[i * K + k] * B[j * K + k];
	}
	atomicAdd(&C[i * n + j], tmp);
}
// dy/dA = dy/dC * B^T
// dy/dB = A^T * dy/dC
void matmul(const float *A, const float *B, size_t m, size_t n, size_t k, float *output) {
	float *BT = (float *)malloc(k * n * sizeof(float));
	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < n; ++j) {
			BT[j * k + i] = B[i * n + j];
		}
	}
	float *cuda_A, *cuda_B, *cuda_C;
	cudaMalloc((void **)&cuda_A, m * k * sizeof(float));
	cudaMalloc((void **)&cuda_B, n * k * sizeof(float));
	cudaMalloc((void **)&cuda_C, m * n * sizeof(float));
	memset(output, 0, m * n * sizeof(float));
	cudaMemcpy(cuda_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B, BT, n * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_C, output, m * n * sizeof(float), cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
	cuda_matmul<<<numBlocks, threadsPerBlock>>>(cuda_A, cuda_B, m, n, k, cuda_C);
	cudaMemcpy(output, cuda_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(cuda_A);
	cudaFree(cuda_B);
	cudaFree(cuda_C);
	free(BT);
}
int main() {
	int m = 2048, k = 2048, n = 2048;
	float *A, *B, *C;
	A = (float *)malloc(m * k * sizeof(float));
	B = (float *)malloc(k * n * sizeof(float));
	C = (float *)malloc(m * n * sizeof(float));
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			// A[i * k + j] = dist(rd);
			A[i * k + j] = 1;
		}
	}
	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < n; ++j) {
			// B[i * n + j] = dist(rd);
			B[i * n + j] = 1;
		}
	}
	matmul(A, B, m, n, k, C);
	for (int i = 0; i < min(m, 10); ++i) {
		for (int j = 0; j < min(n, 10); ++j)
			std::cout << C[i * n + j] << ' ';
		std::cout << '\n';
	}
	free(A);
	free(B);
	free(C);
	return 0;
}