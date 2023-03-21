//#include <iostream>
//#include "cuda_runtime.h";
//#include "device_launch_parameters.h";
//
//#define N 10
//
//__global__ void add(int* a, int* b, int* c)
//{
//	int tid = blockIdx.x;
//	if (tid < N)
//	{
//		c[tid] = a[tid] + b[tid];
//	}
//}
//
//
//int main()
//{
//	int a[N], b[N], c[N];
//	int* dev_a, *dev_b, *dev_c;
//
//	//分配内存
//	cudaMalloc((void**)&dev_a, N * sizeof(int));
//	cudaMalloc((void**)&dev_b, N * sizeof(int));
//	cudaMalloc((void**)&dev_c, N * sizeof(int));
//
//	//赋初值
//	for (int i = 0; i < N; i++)
//	{
//		a[i] = -i;
//		b[i] = i * i;
//	}
//
//	//复制到设备
//	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
//	
//	//并行运行N次核函数
//	add << <N, 1 >> > (dev_a, dev_b, dev_c);
//
//	//将结果传回来
//	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
//
//	//遍历，显示结果
//	for (int i = 0; i < N; i++)
//		printf("%d + %d = %d\n", a[i], b[i], c[i]);
//
//	//释放内存，防止内存泄露
//	cudaFree(dev_a);
//	cudaFree(dev_b);
//	cudaFree(dev_c);
//
//	return 0;
//}