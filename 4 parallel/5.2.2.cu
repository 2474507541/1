#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int N = 66 * 1024;
const int threadPerBlock = 256;
const int blockPerGrid = std::min(32, (N + threadPerBlock - 1) / threadPerBlock);

float sum_square(float x)
{
	return x * (x + 1) * (2 * x + 1) / 6;
}

__global__ void dot(float* a, float* b, float* c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIdx = threadIdx.x;

	__shared__ float cache[threadPerBlock];

	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIdx] = temp;

	__syncthreads();

	//¹éÔ¼
	int i = threadPerBlock / 2;
	while (i)
	{
		if(cacheIdx < i)
			cache[cacheIdx] += cache[i + cacheIdx];

		__syncthreads();

		i /= 2;
	}

	if (cacheIdx == 0)
	{
		c[blockIdx.x] = cache[0];
	}
}

int main()
{
	float* a, * b, * partial_c, c;
	float* dev_a, * dev_b, * dev_partial_c;

	a = new float[N];
	b = new float[N];
	partial_c = new float[blockPerGrid];

	cudaMalloc((void**)&dev_a, N * sizeof(float));
	cudaMalloc((void**)&dev_b, N * sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blockPerGrid * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	cudaMemcpy(dev_a, a, N * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	dot << <blockPerGrid, threadPerBlock >> > (dev_a, dev_b, dev_partial_c);

	cudaMemcpy(partial_c, dev_partial_c, blockPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

	c = 0;
	for (int i = 0; i < blockPerGrid; i++) c += partial_c[i];

	printf("%.6g ?= %.6g", c, 2 * sum_square((float)N - 1));

	delete[] a;
	delete[] b;
	delete[] partial_c;
 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);

	return 0;
}