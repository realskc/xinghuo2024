#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float *input_vecs, float *output_vec, size_t n, size_t dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < dim) {
        atomicAdd(&output_vec[j], input_vecs[i * dim + j]);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void reduce_sum(const float* input_vecs, size_t n, size_t dim, float* output_vec) {
    float *cuda_input_vecs, *cuda_output_vec;
    
    checkCudaError(cudaMalloc((void**)&cuda_input_vecs, n * dim * sizeof(float)), "cudaMalloc cuda_input_vecs");
    checkCudaError(cudaMalloc((void**)&cuda_output_vec, dim * sizeof(float)), "cudaMalloc cuda_output_vec");

    // Initialize the output vector to 0 on the host
    for (int i = 0; i < dim; ++i) {
        output_vec[i] = 0.0f;
    }

    // Copy the input vectors from host to device
    checkCudaError(cudaMemcpy(cuda_input_vecs, input_vecs, n * dim * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy input_vecs to cuda_input_vecs");

    // Copy the initialized output vector from host to device
    checkCudaError(cudaMemcpy(cuda_output_vec, output_vec, dim * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy output_vec to cuda_output_vec");

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (dim + threadsPerBlock.y - 1) / threadsPerBlock.y);

    vectorAdd<<<numBlocks, threadsPerBlock>>>(cuda_input_vecs, cuda_output_vec, n, dim);
    // checkCudaError(cudaGetLastError(), "Kernel launch");
    // checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    // Copy the result from device to host
    checkCudaError(cudaMemcpy(output_vec, cuda_output_vec, dim * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy cuda_output_vec to output_vec");

    cudaFree(cuda_input_vecs);
    cudaFree(cuda_output_vec);
}

int main() {
    int n = 1000, dim = 1000;
    float *a, *b;

    a = (float*)malloc(n * dim * sizeof(float));
    b = (float*)malloc(dim * sizeof(float));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) {
            a[i * dim + j] = 1.0f;
        }
    }

    reduce_sum(a, n, dim, b);

    for (int i = 0; i < std::min(dim, 10); ++i) {
        std::cout << b[i] << ' ';
    }
    std::cout << std::endl;

    free(a);
    free(b);

    return 0;
}