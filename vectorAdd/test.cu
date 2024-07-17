#include <stdio.h>

__global__ void test_shfl_down_sync(int *out) {
	int val = threadIdx.x; // 每个线程的初始值等于其线程ID
	unsigned mask = __activemask();

	// 传输值：从较高线程ID到较低线程ID
	val = __shfl_down_sync(mask, val, 8, 16);

	// 将结果写入输出数组
	out[threadIdx.x] = val;
}

int main() {
	const int numThreads = 32; // warp 大小为 32
	int h_out[numThreads];

	// 分配设备内存
	int *d_out;
	cudaMalloc(&d_out, numThreads * sizeof(int));

	// 启动内核
	test_shfl_down_sync<<<1, numThreads>>>(d_out);

	// 复制结果回主机
	cudaMemcpy(h_out, d_out, numThreads * sizeof(int), cudaMemcpyDeviceToHost);

	// 输出结果
	for (int i = 0; i < numThreads; ++i) {
		printf("Thread %d: %d\n", i, h_out[i]);
	}

	// 释放设备内存
	cudaFree(d_out);

	return 0;
}