#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
	//两个变量，实数和虚数
	float r, i;

	//初始化构造函数参数列表
	__device__ cuComplex(float a, float b) : r(a), i(b){}

	//重载*，+
	__device__ cuComplex operator*(const cuComplex& a) const
	{
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a)const
	{
		return cuComplex(r + a.r, i + a.i);
	}

	//定义计算平方和的成员函数
	__device__ float magnitude2()
	{
		return r * r + i * i;
	}
};


__device__ int julia(int x, int y)
{
	//将像素点的起点移到图的中心，并进行放缩
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	//定义复数结构体
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	//遍历200次，判断是否收敛
	for (int i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
		{
			return 0;
		}
	}

	return 1;
}


__global__ void kernel(unsigned char* ptr)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	//计算每个像素点的索引
	int offset = x + y * gridDim.x;

	//判断是否满足julia条件
	int JuliaValue = julia(x, y);

	//计算每个像素点的颜色
	ptr[offset * 4 + 0] = 255 * JuliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main()
{
	//创建位图
	CPUBitmap bitmap(DIM, DIM);

	//定义设备指针并分配内存
	unsigned char* dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

	//定义线程格
	dim3 grid(DIM, DIM);

	//进入核函数运算
	kernel << <grid, 1 >> > (dev_bitmap);

	//复制回主机并演示
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	

	//释放内存
	cudaFree(dev_bitmap);
	
	bitmap.display_and_exit();

	return 0;
}