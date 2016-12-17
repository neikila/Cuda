/*
 ============================================================================
 Name        : LAB3.cu
 Author      : Kineibe
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
using namespace std;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define H_T 0.0001
#define H_X 0.5
#define TOTAL_TIME 10

#define EPSILON 0.001

#define RIGHT_COND 1
#define LEFT_COND 0
#define BLOCK_SIZE_AMOUNT 256

const double A = H_T / (H_X * H_X);
const double B = 2 * A + 1;

double countSum(int k, double* t, int size) {
	if (k == 0) {
		return t[k] * 1;
	} else if (k == size - 1) {
		return -1 * t[k - 1] / H_X + t[k] / H_X;
	} else {
		return -1 * A * t[k - 1] + t[k] / B - A * t[k + 1];
	}
}

double iterationPart(double prev, double multiplier, double f, double sum) {
	return prev + (f - sum) / multiplier;
}

void iteration(double* t_prev, int size, double* f, double* t_result) {
	for (int i = 0; i < size; ++i) {
		double a;
		if (i == 0)
			a = 1;
		else if (i == size - 1)
			a = 1 / H_X;
		else
			a = B;

		double sum = countSum(i, t_prev, size);
		double newT = iterationPart(t_prev[i], a, f[i], sum);
		t_result[i] = newT;
	}
}

bool condition(double* t_prev, double* t_result, int size) {
	double result = 0;
	for (int i = 0; i < size; ++i) {
		result += abs(t_prev[i] - t_result[i]);
	}
	return result < EPSILON;
}

void iterationManager(double* t_prev, int size, double* f, double* t_target) {
	bool check = true;
	double* t_result = new double[size];

	do {
		iteration(t_prev, size, f, t_result);
		check = condition(t_prev, t_result, size);
		double* temp = t_result;
		t_result = t_prev;
		t_prev = temp;
	} while(!check);

	for (int i = 0; i < size; ++i) {
		t_target[i] = t_prev[i];
	}
	delete[] t_result;
}

void printMas(double* arr, int size) {
	for (int i = 0; i < size; ++i) {
		cout << arr[i] << ' ';
	}
	cout << endl;
}

void model(int size) {
	double* t = new double[size];
	for (int i = 0; i < size; ++i) {
		t[i] = 0;
	}

	double* t_next = new double[size];

	double* f = new double[size];
	f[0] = LEFT_COND;
	f[size - 1] = RIGHT_COND;

//	int iterationAmount = TOTAL_TIME / H_T;
	int iterationAmount = 10;
	for (int i = 0; i < iterationAmount; ++i) {
		cout << "Iteration num " << i << endl;
		for (int i = 1; i < size - 1; ++i) {
			f[i] = t[i];
		}
		cout << "F array" << endl;
		printMas(f, size);

		iterationManager(t, size, f, t_next);
		printMas(t_next, size);

		double* temp = t_next;
		t_next = t;
		t = temp;
	}

	delete[] t_next;
	delete[] f;
	delete[] t;
}

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *data, float *newData, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize) {
		if (idx == vectorSize - 1) {
			newData[idx] = RIGHT_COND * H_T + data[idx];
		} else if (idx == 0) {
			newData[idx] = LEFT_COND;
		} else {
			newData[idx] = data[idx] + (data[idx - 1] - 2 * data[idx] + data[idx + 1]) * H_T / (H_X * H_X);
		}
	}
}

/**
 * Host function that copies the data and launches the work on GPU
 */
void gpuReciprocal(float *data, unsigned size)
{
	cudaEvent_t GPUstart, GPUstop;
	float GPUtime = 0.0f;
	float *rc = new float[size];
	float *gpuOldData;
	float *gpuNewData;

	int iterationAmount = TOTAL_TIME / H_T;

	static const int BLOCK_SIZE = BLOCK_SIZE_AMOUNT;
	const int blockCount = 1000;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuOldData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuNewData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuOldData, data, sizeof(float)*size, cudaMemcpyHostToDevice));

	cudaEventCreate(&GPUstart);
	cudaEventCreate(&GPUstop);


	for (int i = 0; i < iterationAmount; ++i) {
		cudaEventRecord(GPUstart, 0);

		if (i % 2 == 0) {
			reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuOldData, gpuNewData, size);
			cudaEventRecord(GPUstop, 0);
			CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuNewData, sizeof(float)*size, cudaMemcpyDeviceToHost));
		} else {
			reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuNewData, gpuOldData, size);
			cudaEventRecord(GPUstop, 0);
			CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuOldData, sizeof(float)*size, cudaMemcpyDeviceToHost));
		}
		cudaEventSynchronize(GPUstop);

		float temp;
		cudaEventElapsedTime(&temp, GPUstart, GPUstop);
		GPUtime += temp;
//
//		for (int i = 0; i < size; ++i) {
//			std::cout << "t[" << i << "] = " << rc[i] << std::endl;
//		}
//		std::cout << std::endl;
	}

	printf("GPU time : %.3f ms\n", GPUtime);

	CUDA_CHECK_RETURN(cudaFree(gpuOldData));
	CUDA_CHECK_RETURN(cudaFree(gpuNewData));
}

void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = 0;
}

void cpuIteration(float *data, float *newData, unsigned vectorSize) {
	for (int idx = 0; idx < vectorSize; ++idx) {
		if (idx == vectorSize - 1) {
			newData[idx] = RIGHT_COND * H_T + data[idx];
		} else if (idx == 0) {
			newData[idx] = LEFT_COND;
		} else {
			newData[idx] = data[idx] + (data[idx - 1] - 2 * data[idx] + data[idx + 1]) * H_T / (H_X * H_X);
		}
	}
}

void cpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *oldData = new float[size];

	float* result;

	float CPUstart, CPUstop;
	float CPUtime = 0.0f;

	int iterationAmount = TOTAL_TIME / H_T;

	for (int i = 0; i < iterationAmount; ++i) {

		CPUstart = clock();
		if (i % 2 == 0) {
			cpuIteration(oldData, rc, size);
			result = rc;
		} else {
			cpuIteration(rc, oldData, size);
			result = oldData;
		}
		CPUstop = clock();
		CPUtime += 1000.*(CPUstop - CPUstart) / CLOCKS_PER_SEC;
//
//		for (int i = 0; i < size; ++i) {
//			std::cout << "t[" << i << "] = " << result[i] << std::endl;
//		}
//		std::cout << std::endl;
	}

	printf("CPU time : %.3f ms\n", CPUtime);
}

bool checkShodimost() {
	return true;
}

int main(void)
{
	static const int WORK_SIZE = 256000;
	float *data = new float[WORK_SIZE];

	model(5);


	/* Free memory */
	delete[] data;

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

