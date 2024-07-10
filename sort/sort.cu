#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
std::mt19937 rd(114514);
__global__ void cuda_merge(const int *a, int *b, int n, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i * 2ll * len >= n)
		return;
	i = i * 2 * len;
	if (i + len < n) {
		int l1 = i, l2 = i + len, r1 = l2, r2 = min(i + len * 2, n), top = l1;
		while (l1 < r1 && l2 < r2) {
			if (a[l1] < a[l2])
				b[top++] = a[l1++];
			else
				b[top++] = a[l2++];
		}
		while (l1 < r1)
			b[top++] = a[l1++];
		while (l2 < r2)
			b[top++] = a[l2++];
	} else {
		for (int j = i; j < n; ++j)
			b[j] = a[j];
	}
}
void sort(std::vector<int> &nums) {
	int n = nums.size();
	int *cuda_a, *cuda_b;
	cudaMalloc((void **)&cuda_a, n * sizeof(int));
	cudaMalloc((void **)&cuda_b, n * sizeof(int));
	cudaMemcpy(cuda_a, nums.data(), n * sizeof(int), cudaMemcpyHostToDevice);
	int len = 1;
	while (len < n) {
		int threadsPerBlock = 256;
		int numBlocks = ((n + len * 2 - 1) / (len * 2) + threadsPerBlock - 1) / threadsPerBlock;
		cuda_merge<<<numBlocks, threadsPerBlock>>>(cuda_a, cuda_b, n, len);
		std::swap(cuda_a, cuda_b);
		len <<= 1;
		// for (int i = 0; i < min(n, 10); ++i)
		// 	std::cout << nums[i] << ' ';
		// std::cout << '\n';
	}
	cudaMemcpy(nums.data(), cuda_a, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cuda_a);
	cudaFree(cuda_b);
}
int main() {
	int n = 10000000;
	std::vector<int> vec(n);
	for (int i = 0; i < n; ++i) {
		vec[i] = rd();
		// vec[i] = n - i;
	}
	sort(vec);
	for (int i = 0; i < min(n, 10); ++i)
		std::cout << vec[i] << ' ';
	std::cout << '\n';
	return 0;
}