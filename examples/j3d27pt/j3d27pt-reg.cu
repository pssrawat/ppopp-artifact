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

__global__ void j3d27pt (double * __restrict__ t_in, double * __restrict__ t_out, int N) {
	//Determing the block's indices
	int i0 = (int)(blockIdx.x)*(int)(blockDim.x) + 1;
	int i = max(i0,1) + (int)(threadIdx.x);
	int j0 = 4*(int)(blockIdx.y)*(int)(blockDim.y) + 1;
	int j = max(j0,1) + 4*(int)(threadIdx.y);
	int k0 = (int)(blockIdx.z)*(int)(blockDim.z) + 1;
	int k = max(k0,1) + (int)(threadIdx.z);

	double (*in)[514][514] = (double (*)[514][514])t_in;
	double (*out)[514][514] = (double (*)[514][514])t_out;

	if (i<=N-2 & j<=N-2 && k<=N-2) {
#pragma begin stencil1 unroll k=1,j=4,i=1
                             out[k][j][i] = 0.125 * in[k][j][i] +
                                        1.14 * (in[k-1][j][i] + in[k+1][j][i] + in[k][j-1][i] +
                                                in[k][j+1][i] + in[k][j][i-1] + in[k][j][i+1]) +
                                        0.75 * (in[k-1][j-1][i-1] + in[k-1][j-1][i+1] + in[k-1][j+1][i-1] +
                                                in[k-1][j+1][i+1] + in[k+1][j-1][i-1] + in[k+1][j-1][i+1] +
                                                in[k+1][j+1][i-1] + in[k+1][j+1][i+1]) +
                                        1.031 * (in[k-1][j-1][i] + in[k-1][j][i-1] + in[k-1][j][i+1] +
                                                in[k-1][j+1][i] + in[k][j-1][i-1] + in[k][j-1][i+1] +
                                                in[k][j+1][i-1] + in[k][j+1][i+1] + in[k+1][j-1][i] +
                                                in[k+1][j][i-1] + in[k+1][j][i+1] + in[k+1][j+1][i]);
#pragma end stencil1
	}
}

extern "C" void host_code (double *h_in, double *h_out, int N) {
	double *in;
	cudaMalloc (&in, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for in\n");
	cudaMemcpy (in, h_in, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *out;
	cudaMalloc (&out, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for out\n");

	dim3 blockconfig (32,4,4);
	dim3 gridconfig (ceil(N-2, blockconfig.x), ceil(N-2, 4*blockconfig.y), ceil(N-2, blockconfig.z));

	j3d27pt<<<gridconfig, blockconfig>>> (in, out, N);
	cudaMemcpy (h_out, out, sizeof(double)*N*N*N, cudaMemcpyDeviceToHost);

	cudaFree (in); 
	cudaFree (out);
}
