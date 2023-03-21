#include <iostream>
#include "cuda_runtime.h";
#include "device_launch_parameters.h"

#define N (66 * 1024)

__global__ void add(int* a, int* b, int* c)
{
	//转化线性索引
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main()
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add << <128, 128 >> > (dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	bool success = true;
	for (int i = 0; i < N; i++)
	{
		if (a[i] + b[i] != c[i])
		{
			printf("%d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}

	if (success) printf("Did it.");

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}