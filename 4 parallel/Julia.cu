#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
	//����������ʵ��������
	float r, i;

	//��ʼ�����캯�������б�
	__device__ cuComplex(float a, float b) : r(a), i(b){}

	//����*��+
	__device__ cuComplex operator*(const cuComplex& a) const
	{
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a)const
	{
		return cuComplex(r + a.r, i + a.i);
	}

	//�������ƽ���͵ĳ�Ա����
	__device__ float magnitude2()
	{
		return r * r + i * i;
	}
};


__device__ int julia(int x, int y)
{
	//�����ص������Ƶ�ͼ�����ģ������з���
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	//���帴���ṹ��
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	//����200�Σ��ж��Ƿ�����
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

	//����ÿ�����ص������
	int offset = x + y * gridDim.x;

	//�ж��Ƿ�����julia����
	int JuliaValue = julia(x, y);

	//����ÿ�����ص����ɫ
	ptr[offset * 4 + 0] = 255 * JuliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main()
{
	//����λͼ
	CPUBitmap bitmap(DIM, DIM);

	//�����豸ָ�벢�����ڴ�
	unsigned char* dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

	//�����̸߳�
	dim3 grid(DIM, DIM);

	//����˺�������
	kernel << <grid, 1 >> > (dev_bitmap);

	//���ƻ���������ʾ
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	

	//�ͷ��ڴ�
	cudaFree(dev_bitmap);
	
	bitmap.display_and_exit();

	return 0;
}