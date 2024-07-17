#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
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
std::mt19937 rd(114514);
int cpuDebubble(thrust::host_vector<int> &vec) {
	int top = 0;
	for (int i = 0; i < vec.size(); ++i) {
		if (vec[i]) {
			vec[top++] = vec[i];
		}
	}
	for (int i = top; i < vec.size(); ++i) {
		vec[i] = 0;
	}
	return top;
}
constexpr int TILE_SIZE = 32;
__global__ void gpu_getVal(const int *data, int *val, int *warpPresum, int n) {
	LIMITED_KERNEL_LOOP(i, n) {
		int o = !!__ldg(data + i), mask = __activemask();
		// printf("i: %lld, o: %d, mask: %d\n", i, o, mask);
		for (int j = 1; j < TILE_SIZE; j <<= 1) {
			int tmp = __shfl_up_sync(mask, o, j, TILE_SIZE);
			if (i % TILE_SIZE >= j)
				o += tmp;
		}
		// printf("i: %lld, o: %d\n", i, o);
		if (i < n) {
			val[i] = o;
			if ((i & TILE_SIZE - 1) == TILE_SIZE - 1 && i + 1 < n) {
				warpPresum[(i + 1) / TILE_SIZE] = o;
			}
		}
	}
}
__global__ void gpu_getPresum(int *data, int *output, int n, int step) {
	LIMITED_KERNEL_LOOP(i, n) {
		int tmp = data[i];
		if (i >= step)
			tmp += data[i - step];
		output[i] = tmp;
	}
}
__global__ void gpu_debubbleKernel(int *data, int *val, int *presum, int *output, int n) {
	LIMITED_KERNEL_LOOP(i, n) {
		// printf("i: %lld, val: %d\n", i, val[i]);
		int c = __ldg(data + i);
		if (c != 0) {
			output[presum[i / TILE_SIZE] + val[i] - 1] = c;
		}
	}
}
int gpuDebubble(thrust::device_vector<int> &vec) {
	const int numBlocks = 24, threadsPerBlock = 1024;
	int n = vec.size(), m = (n + TILE_SIZE - 1) / TILE_SIZE, *val, *warpPresum;
	cudaMalloc((void **)&val, n * sizeof(int));
	cudaMalloc((void **)&warpPresum, m * sizeof(int));
	cudaMemset(warpPresum, 0, m * sizeof(int));
	gpu_getVal<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(vec.data()), val, warpPresum, n);
	int *output;
	cudaMalloc(&output, m * sizeof(int));
	for (int i = 1; i < m; i <<= 1) {
		cudaMemset(output, 0, m * sizeof(int));
		gpu_getPresum<<<numBlocks, threadsPerBlock>>>(warpPresum, output, m, i);
		std::swap(warpPresum, output);
	}
	int *ans;
	cudaMalloc(&ans, n * sizeof(int));
	cudaMemset(ans, 0, n * sizeof(int));
	gpu_debubbleKernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(vec.data()), val, warpPresum, ans, n);
	int tmp1, tmp2;
	cudaMemcpy(&tmp1, val + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&tmp2, warpPresum + (n - 1) / TILE_SIZE, sizeof(int), cudaMemcpyDeviceToHost);
	int rv = tmp1 + tmp2;
	cudaMemcpy(thrust::raw_pointer_cast(vec.data()), ans, n * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaFree(val);
	cudaFree(warpPresum);
	cudaFree(output);
	cudaFree(ans);
	return rv;
}
int main() {
	int n = 10000000;
	thrust::host_vector<int> vec(n), gpu_content;
	for (int i = 0; i < n; ++i) {
		if (rd() % 2 == 0)
			vec[i] = 0;
		else
			vec[i] = rd();
	}
	int LIM = 30;
	std::cerr << "Input:\n";
	for (int i = 0; i < min(n, LIM); ++i)
		std::cerr << vec[i] << ' ';
	std::cerr << '\n';
	thrust::device_vector<int> gpu_vec;
	int cpuans, gpuans;
	PERF("cpu_impl", cpuans = cpuDebubble(vec));
	gpu_vec = vec;
	PERF_DEV("gpu_impl", gpuans = gpuDebubble(gpu_vec));
	gpu_vec = vec;
	PERF_DEV("gpu_impl", gpuans = gpuDebubble(gpu_vec));
	gpu_vec = vec;
	PERF_DEV("gpu_impl", gpuans = gpuDebubble(gpu_vec));
	gpu_vec = vec;
	PERF_DEV("gpu_impl", gpuans = gpuDebubble(gpu_vec));
	gpu_content = gpu_vec;
	if (cpuans != gpuans || vec != gpu_content) {
		if (cpuans != gpuans)
			std::cout << "Wrong Answer on Count\n";
		else {
			int i;
			for (i = 0; i < n; ++i) {
				if (vec[i] != gpu_content[i])
					break;
			}
			std::cout << "Wrong Answer on Position " << i << '\n';
		}
		std::cerr << "CPU:\n";
		for (int i = 0; i < min(n, LIM); ++i)
			std::cerr << vec[i] << ' ';
		std::cerr << '\n';
		std::cerr << "GPU:\n";
		for (int i = 0; i < min(n, LIM); ++i)
			std::cerr << gpu_content[i] << ' ';
		std::cerr << '\n';
	} else {
		std::cout << "Correct\n";
	}
	return 0;
}