#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#define now() std::chrono::high_resolution_clock::now()
#define time_diff(start, end) std::chrono::duration<double>(end - start).count()
#define PERF(msg, behavior)                                                                                            \
	{                                                                                                                  \
		auto t0 = now();                                                                                               \
		behavior;                                                                                                      \
		auto t1 = now();                                                                                               \
		auto cost = time_diff(t0, t1);                                                                                 \
		std::cout << std::scientific << msg << ": " << cost << " secs" << std::endl;                                   \
	}
#define PERF_DEV(msg, behavior)                                                                                        \
	{                                                                                                                  \
		cudaEvent_t start, stop;                                                                                       \
		cudaEventCreate(&start);                                                                                       \
		cudaEventCreate(&stop);                                                                                        \
		cudaEventRecord(start);                                                                                        \
		behavior;                                                                                                      \
		cudaEventRecord(stop);                                                                                         \
		cudaError_t err = cudaStreamSynchronize(0);                                                                    \
		float cost = 0;                                                                                                \
		cudaEventElapsedTime(&cost, start, stop);                                                                      \
		std::cout << std::scientific << msg << ": " << cost / 1000 << " secs" << std::endl;                            \
		if (err != cudaSuccess) {                                                                                      \
			fprintf(stderr, "PERF_DEV run %s failed. ret=%d\n", #behavior, err);                                       \
		}                                                                                                              \
	}
#define LIMITED_KERNEL_LOOP(i, n)                                                                                      \
	for (size_t i = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += gridDim.x * blockDim.x)
#define LIMITED_BLOCK_LOOP(i, n) for (size_t i = static_cast<size_t>(threadIdx.x); i < n; i += blockDim.x)
void cpuReduceSum(const float *input_vecs, int n, int dim, float *output_vec) {
	for (int i = 0; i < dim; ++i)
		output_vec[i] = 0;
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < dim; ++j)
			output_vec[j] += input_vecs[i * dim + j];
}
constexpr int TILE_SIZE = 8;
__global__ void gpuReduceSumKernel(const float *input_vecs, float *output_vec, int n, int st, int dim, int realDim) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.f;
	extern __shared__ float shM[];
	LIMITED_BLOCK_LOOP(i, TILE_SIZE) shM[i] = 0;
	__syncthreads();
	LIMITED_KERNEL_LOOP(i, n * dim) {
		int x = i / dim, y = i - x * dim;
		if (y + st < realDim) {
			int pos = x * realDim + y + st;
			float tmp = __ldg(input_vecs + pos);
			sum += tmp;
		}
	}
	atomicAdd(shM + id % TILE_SIZE, sum);
	__syncthreads();
	LIMITED_BLOCK_LOOP(i, TILE_SIZE) {
		if (i >= dim)
			atomicAdd(shM + i % dim, shM[i]);
	}
	__syncthreads();
	LIMITED_BLOCK_LOOP(i, dim) {
		if (st + i < realDim) {
			atomicAdd(output_vec + st + i, shM[i]);
		}
	}
}
void gpuReduceSum(const float *input_vecs, int n, int dim, float *output_vec) {
	cudaMemset(output_vec, 0, dim * sizeof(float));
	for (int i = 0; i < dim; i += TILE_SIZE) {
		int dim_ = std::min(TILE_SIZE, dim - i), dim__ = 1;
		while (dim__ < dim_)
			dim__ <<= 1;
		gpuReduceSumKernel<<<24, 1024, TILE_SIZE * sizeof(float)>>>(input_vecs, output_vec, n, i, dim__, dim);
	}
}
int main() {
	int n = 1000000, dim = 32;
	float *data = (float *)malloc(n * dim * sizeof(float)), *cpu_output = (float *)malloc(dim * sizeof(float)),
		  *gpu_content = (float *)malloc(dim * sizeof(float)), *gpu_input, *gpu_output;
	cudaMalloc((void **)&gpu_input, n * dim * sizeof(float));
	cudaMalloc((void **)&gpu_output, dim * sizeof(float));
	std::mt19937 randomEngine;
	std::normal_distribution<float> dist;
	for (int i = 0; i < n * dim; ++i)
		data[i] = dist(randomEngine);
	cudaMemcpy(gpu_input, data, n * dim * sizeof(float), cudaMemcpyHostToDevice);
	PERF("cpu_impl", cpuReduceSum(data, n, dim, cpu_output));
	PERF_DEV("gpu_impl", gpuReduceSum(gpu_input, n, dim, gpu_output));
	cudaMemcpy(gpu_content, gpu_output, dim * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < dim; ++i) {
		if (cpu_output[i] != gpu_content[i]) {
			std::cerr << "Wrong Answer on Index " << i << std::endl;
			for (int i = 0; i < min(dim, 10); ++i)
				std::cerr << cpu_output[i] << ' ';
			std::cerr << std::endl;
			for (int i = 0; i < min(dim, 10); ++i)
				std::cerr << gpu_content[i] << ' ';
			std::cerr << std::endl;
			return 0;
		}
	}
	std::cerr << "cpu and gpu are the same\n";
}