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

__global__ void j3d125pt (double * __restrict__ t_in, double * __restrict__ t_out, int N) {
	//Determing the block's indices
	int i0 = (int)(blockIdx.x)*(int)(blockDim.x) + 2;
	int i = max(i0,2) + (int)(threadIdx.x);
	int j0 = 4*(int)(blockIdx.y)*(int)(blockDim.y) + 2;
	int j = max(j0,2) + 4*(int)(threadIdx.y);
	int k0 = (int)(blockIdx.z)*(int)(blockDim.z) + 2;
	int k = max(k0,2) + (int)(threadIdx.z);

	double (*in)[516][516] = (double (*)[516][516])t_in;
	double (*out)[516][516] = (double (*)[516][516])t_out;

	if (i<=N-3 & j<=N-3 && k<=N-3) {
#pragma unroll 4
		for (int jj=0; jj<=3; jj++) {
                out[k][j+jj][i] =
                        0.75 * (in[k-2][j+jj-2][i-2] + in[k-2][j+jj-2][i+2] + in[k-2][j+jj+2][i-2] + in[k-2][j+jj+2][i+2] +
                        in[k-1][j+jj-1][i-1] + in[k-1][j+jj-1][i+1] + in[k-1][j+jj+1][i-1] + in[k-1][j+jj+1][i+1] +
                        in[k][j+jj-1][i] + in[k][j+jj][i-1] +  in[k][j+jj][i+1] + in[k][j+jj+1][i] +
                        in[k+1][j+jj-1][i-1] + in[k+1][j+jj-1][i+1] + in[k+1][j+jj+1][i-1] + in[k+1][j+jj+1][i+1] +
                        in[k+2][j+jj-2][i-2] + in[k+2][j+jj-2][i+2] + in[k+2][j+jj+2][i-2] + in[k+2][j+jj+2][i+2]) + 

                        1.132 * (in[k-2][j+jj-2][i-1] + in[k-2][j+jj-2][i+1] + in[k-2][j+jj-1][i-2] + in[k-2][j+jj-1][i+2] +
                        in[k-2][j+jj][i] + in[k-2][j+jj+1][i-2] + in[k-2][j+jj+1][i+2] + in[k-2][j+jj+2][i-1] +  in[k-2][j+jj+2][i+1] +
                        in[k-1][j+jj-2][i-2] + in[k-1][j+jj-2][i+2] + in[k-1][j+jj+2][i-2] + in[k-1][j+jj+2][i+2] +
                        in[k][j+jj-2][i] + in[k][j+jj][i-2] + in[k][j+jj][i+2] + in[k][j+jj+2][i] +
                        in[k+1][j+jj-2][i-2] + in[k+1][j+jj-2][i+2] + in[k+1][j+jj+2][i-2] + in[k+1][j+jj+2][i+2] +
                        in[k+2][j+jj-2][i-1] + in[k+2][j+jj-2][i+1] + in[k+2][j+jj-1][i-2] + in[k+2][j+jj-1][i+2] + in[k+2][j+jj][i] + 
                        in[k+2][j+jj+1][i-2] + in[k+2][j+jj+1][i+2] + in[k+2][j+jj+2][i-1] +  in[k+2][j+jj+2][i+1]) +

                        0.217 * (in[k-2][j+jj-2][i] + in[k-2][j+jj][i-2] + in[k-2][j+jj][i+2] + in[k-2][j+jj+2][i] +
                        in[k-1][j+jj-1][i] + in[k-1][j+jj][i-1] +  in[k-1][j+jj][i+1] + in[k-1][j+jj+1][i] +
                        in[k][j+jj-2][i-2] + in[k][j+jj-2][i+2] + in[k][j+jj+2][i-2] + in[k][j+jj+2][i+2] +
                        in[k+1][j+jj-1][i] + in[k+1][j+jj][i-1] +  in[k+1][j+jj][i+1] + in[k+1][j+jj+1][i] +
                        in[k+2][j+jj-2][i] + in[k+2][j+jj][i-2] + in[k+2][j+jj][i+2] + in[k+2][j+jj+2][i]) +  

                        2.13 * (in[k-2][j+jj-1][i] + in[k-2][j+jj][i-1] +  in[k-2][j+jj][i+1] + in[k-2][j+jj+1][i] +
                        in[k-1][j+jj-2][i] + in[k-1][j+jj][i-2] + in[k-1][j+jj][i+2] + in[k-1][j+jj+2][i] +
                        in[k][j+jj-2][i-1] + in[k][j+jj-2][i+1] + in[k][j+jj-1][i-2] + in[k][j+jj-1][i+2] +
                        in[k][j+jj][i] + in[k][j+jj+1][i-2] + in[k][j+jj+1][i+2] + in[k][j+jj+2][i-1] +  in[k][j+jj+2][i+1] +
                        in[k+1][j+jj-2][i] + in[k+1][j+jj][i-2] + in[k+1][j+jj][i+2] + in[k+1][j+jj+2][i] +
                        in[k+2][j+jj-1][i] + in[k+2][j+jj][i-1] +  in[k+2][j+jj][i+1] + in[k+2][j+jj+1][i]) + 

                        0.331 * (in[k-2][j+jj-1][i-1] + in[k-2][j+jj-1][i+1] + in[k-2][j+jj+1][i-1] + in[k-2][j+jj+1][i+1] +
                        in[k-1][j+jj-2][i-1] + in[k-1][j+jj-2][i+1] + in[k-1][j+jj-1][i-2] + in[k-1][j+jj-1][i+2] + in[k-1][j+jj][i] +
                        in[k-1][j+jj+1][i-2] + in[k-1][j+jj+1][i+2] + in[k-1][j+jj+2][i-1] +  in[k-1][j+jj+2][i+1] +
                        in[k][j+jj-1][i-1] + in[k][j+jj-1][i+1] + in[k][j+jj+1][i-1] + in[k][j+jj+1][i+1] +
                        in[k+1][j+jj-2][i-1] + in[k+1][j+jj-2][i+1] + in[k+1][j+jj-1][i-2] + in[k+1][j+jj-1][i+2] + in[k+1][j+jj][i] +
                        in[k+1][j+jj+1][i-2] + in[k+1][j+jj+1][i+2] + in[k+1][j+jj+2][i-1] +  in[k+1][j+jj+2][i+1] +
                        in[k+2][j+jj-1][i-1] + in[k+2][j+jj-1][i+1] + in[k+2][j+jj+1][i-1] + in[k+2][j+jj+1][i+1]); 
		}
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

	dim3 blockconfig (16, 4, 4);
	dim3 gridconfig (ceil(N-4, blockconfig.x), ceil(N-4, 4*blockconfig.y), ceil(N-4, blockconfig.z));

	j3d125pt<<<gridconfig, blockconfig>>> (in, out, N);

	cudaMemcpy (h_out, out, sizeof(double)*N*N*N, cudaMemcpyDeviceToHost);

	cudaFree (in); 
	cudaFree (out);
}
