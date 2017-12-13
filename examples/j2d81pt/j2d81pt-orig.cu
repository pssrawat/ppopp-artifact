#include <stdio.h>
#include "cuda.h"
#define max(x,y)  ((x) > (y)? (x) : (y))
#define min(x,y)  ((x) < (y)? (x) : (y))
#define ceil(a,b) ((a) % (b) == 0 ? (a) / (b) : ((a) / (b)) + 1)

void check_error (const char* message) {
	cudaError_t error = cudaGetLastError ();
	if (error != cudaSuccess) {
		printf ("CUDA error : %s, %s\n", message, cudaGetErrorString (error));
		exit(-1);
	}
}

__global__ void j2d81pt (double * __restrict__ l_in, double * __restrict__ l_out, int N) {
	//Determing the block's indices
	int i0 = (int)(blockIdx.x)*(int)(blockDim.x);
	int i = max(i0,0) + (int)(threadIdx.x);
	int j0 = (int)(blockIdx.y)*(int)(blockDim.y);
	int j = max(j0,0) + (int)(threadIdx.y);

	double (*in)[8200] = (double (*)[8200]) l_in;
	double (*out)[8200] = (double (*)[8200]) l_out;

	if (i>=0 & j>=0 & i<=N-9 & j<=N-9) {
		out[j][i] =
			(in[j][i] + in[j][i+8] + in[j+8][i] + in[j+8][i+8]) * 3.1862206 +
			(in[j][i+1] + in[j][i+7] + in[j+1][i] + in[j+1][i+8] + in[j+7][i] + in[j+7][i+8] + in[j+8][i+1] + in[j+8][i+7]) * 4.5339005 +
			(in[j][i+2] + in[j][i+6] + in[j+2][i] + in[j+2][i+8] + in[j+6][i] + in[j+6][i+8] + in[j+8][i+2] + in[j+8][i+6]) * -0.000357000 +
			(in[j][i+3] + in[j][i+5] + in[j+3][i] + in[j+3][i+8] + in[j+5][i] + in[j+5][i+8] + in[j+8][i+3] + in[j+8][i+5]) * 0.00285600 +
			(in[j][i+4] + in[j+4][i+8] + in[j+4][i] + in[j+8][i+4]) * -0.00508225 +
			(in[j+1][i+1] + in[j+1][i+7] + in[j+7][i+1] + in[j+7][i+7]) * 0.000645160 +
			(in[j+1][i+2] + in[j+1][i+6] + in[j+2][i+1] + in[j+2][i+7] + in[j+6][i+1] + in[j+6][i+7] + in[j+7][i+2] + in[j+7][i+6]) * -0.00508000 +
			(in[j+1][i+3] + in[j+1][i+5] + in[j+3][i+1] + in[j+3][i+7] + in[j+5][i+1] + in[j+5][i+7] + in[j+7][i+3] + in[j+7][i+5]) * 0.0406400 +
			(in[j+1][i+4] + in[j+4][i+1] + in[j+4][i+7] + in[j+7][i+4]) * -0.0723189 +
			(in[j+2][i+2] + in[j+2][i+6] + in[j+6][i+2] + in[j+6][i+6]) * 0.0400000 +
			(in[j+2][i+3] + in[j+2][i+5] + in[j+3][i+2] + in[j+3][i+6] + in[j+5][i+2] + in[j+5][i+6] + in[j+6][i+3] + in[j+6][i+5]) * -0.320000 +
			(in[j+2][i+4] + in[j+4][i+2] + in[j+4][i+6] + in[j+6][i+4]) * 0.569440 +
			(in[j+3][i+3] + in[j+3][i+5] + in[j+5][i+3] + in[j+5][i+5]) * 2.56000 +
			(in[j+3][i+4] + in[j+4][i+3] + in[j+4][i+5] + in[j+5][i+4]) * -4.55552 +
			in[j+4][i+4] * 8.10655;
	} 
}

extern "C" void host_code (double *h_in, double *h_out, int N) {
	double *in;
	cudaMalloc (&in, sizeof(double)*N*N);
	check_error ("Failed to allocate device memory for in\n");
	cudaMemcpy (in, h_in, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	double *out;
	cudaMalloc (&out, sizeof(double)*N*N);
	check_error ("Failed to allocate device memory for out\n");

	dim3 blockconfig (16, 8);
	dim3 gridconfig (ceil(N, blockconfig.x), ceil(N, blockconfig.y));

	j2d81pt<<<gridconfig, blockconfig>>> (in, out, N);
	cudaMemcpy (h_out, out, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

	cudaFree (in); 
	cudaFree (out);
}
