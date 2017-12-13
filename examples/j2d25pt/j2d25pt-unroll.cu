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

__global__ void j2d25pt (double * __restrict__ l_in, double * __restrict__ l_out, int N) {
	//Determing the block's indices
	int i0 = (int)(blockIdx.x)*(int)(blockDim.x);
	int i = max(i0,2) + (int)(threadIdx.x);
	int j0 = 4*(int)(blockIdx.y)*(int)(blockDim.y);
	int j = max(j0,2) + 4*(int)(threadIdx.y);

	double (*in)[8196] = (double (*)[8196]) l_in;
	double (*out)[8196] = (double (*)[8196]) l_out;

	if (i>=2 & j>=2 & i<=N-3 & j<=N-3) {
#pragma unroll 4
		for (int jj=0; jj<=3; jj++) {
			out[j+jj][i] = 0.1*(in[j+jj-2][i-2] + in[j+jj-2][i+2] + in[j+jj+2][i-2] + in[j+jj+2][i+2]) +
				0.2*(in[j+jj-2][i-1] + in[j+jj-2][i+1] + in[j+jj+2][i-1] + in[j+jj+2][i+1]) +
				0.3*(in[j+jj-2][i] + in[j+jj+2][i]) +
				1.1*(in[j+jj-1][i-2] + in[j+jj-1][i+2] + in[j+jj+1][i-2] + in[j+jj+1][i+2]) +
				1.2*(in[j+jj-1][i-1] + in[j+jj-1][i+1] + in[j+jj+1][i-1] + in[j+jj+1][i+1]) +
				1.3*(in[j+jj-1][i] + in[j+jj+1][i]) +
				2.1*(in[j+jj][i-2] + in[j+jj][i+2]) +
				2.2*(in[j+jj][i-1] + in[j+jj][i+1]) +
				2.3*in[j+jj][i]; 
		}
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
	dim3 gridconfig (ceil(N, blockconfig.x), ceil(N, 4*blockconfig.y));

	j2d25pt<<<gridconfig, blockconfig>>> (in, out, N);
	cudaMemcpy (h_out, out, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

	cudaFree (in); 
	cudaFree (out);
}
