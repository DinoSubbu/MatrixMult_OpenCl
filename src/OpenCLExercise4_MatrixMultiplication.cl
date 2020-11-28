#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__kernel void matrixMulKernel1(__global const float* d_inputA, __global const float* d_inputB, __global float* d_outputC, const int countAX_BY) {
	int j = get_global_id(0);
	int i = get_global_id(1);

	unsigned long countAY = get_global_size(0);
	unsigned long countBX = get_global_size(1);

	float sum = 0;
	for (std::size_t k = 0; k < countAX_BY; k++) {
		float a = d_inputA[k + j * countAX_BY];
		float b = d_inputB[i + k * countBX];
		sum += a * b;
	}
	d_outputC[i + j * countBX] = sum;
}

// The preprocessor constant WG_SIZE will contain the size of a work group in X/Y-direction

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrixMulKernel2(__global const float* d_inputA, __global const float* d_inputB, __global float* d_outputC, const int countAX_BY) {
	//TODO
	int j = get_global_id(0);
	int i = get_global_id(1);

	unsigned long countAX_BY = get_global_size(0);
	unsigned long countBX = get_global_size(1);

	__local float l_A[16][16];
	__local float l_B[16][16];

	int k = get_local_id(0);
	int g = get_local_id(1);

	float sum = 0;
	// loop over the submatrices
	for (std::size_t bs = 0; bs < countAX_BY; bs+=16) {
		//Copy blocks of d_inputA , d_inputB to local memory
		l_A[g][k] = d_inputA[(k+bs) + j * countAX_BY];
		l_B[g][k] = d_inputB[i + (g+bs) * countBX];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (uint m = 0; m < 16; m++)
			sum += l_A[g][m] * l_B[m][k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	d_outputC[i + j * countBX] = sum;
}


__kernel void matrixMulKernel3(__global const float* d_inputA, __global const float* d_inputB, __global float* d_outputC, const int countAX_BY, __local float* localMem) {
	//TODO
	int j = get_global_id(0);
	int i = get_global_id(1);

	unsigned long countAX_BY = get_global_size(0);
	unsigned long countBX = get_global_size(1);

	__local float l_A = localMem;
	__local float l_B = localMem + get_local_size(0) * get_local_size(1);

	int k = get_local_id(0);
	int g = get_local_id(1);

	float sum = 0;
	// loop over the submatrices
	for (std::size_t bs = 0; bs < countAX_BY; bs+=16) {
		//Copy blocks of d_inputA , d_inputB to local memory
		l_A[g][k] = d_inputA[(k+bs) + j * countAX_BY];
		l_B[g][k] = d_inputB[i + (g+bs) * countBX];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (uint m = 0; m < 16; m++)
			sum += l_A[g][m] * l_B[m][k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	d_outputC[i + j * countBX] = sum;
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void matrixMulKernel4(__read_only image2d_t d_inputA, __read_only image2d_t d_inputB, __global float* d_outputC, const int countAX_BY) {
	int j = get_global_id(0);
	int i = get_global_id(1);

	unsigned long countAY = get_global_size(0);
	unsigned long countBX = get_global_size(1);

	float sum = 0;
	for (std::size_t k = 0; k < countAX_BY; k++) {
		float a = read_imagef(d_inputA, sampler, k + j * countAX_BY)
		float b = read_imagef(d_inputA, sampler, i + k * countBX)
		sum += a * b;
	}
	d_outputC[i + j * countBX] = sum;
}