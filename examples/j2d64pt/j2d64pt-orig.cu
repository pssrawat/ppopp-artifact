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

__global__ void j2d64pt (double * __restrict__ l_in, double * __restrict__ l_out, int N) {
	//Determing the block's indices
	int i0 = (int)(blockIdx.x)*(int)(blockDim.x);
	int i = max(i0,4) + (int)(threadIdx.x);
	int j0 = (int)(blockIdx.y)*(int)(blockDim.y);
	int j = max(j0,4) + (int)(threadIdx.y);

        double (*in)[8200] = (double (*)[8200]) l_in;
        double (*out)[8200] = (double (*)[8200]) l_out;

	if (i>=4 & j>=4 & i<=N-5 & j<=N-5) {
                                out[j][i] =
                                        (in[j-4][i-4] - in[j-4][i+4] - in[j+4][i-4] + in[j+4][i+4]) * 1.274495 +
                                        (-in[j-4][i-3] + in[j-4][i+3] + in[j-3][i+4] - in[j-3][i-4] + in[j+3][i-4] - in[j+3][i+4] + in[j+4][i-3] - in[j+4][i+3]) * 0.000136017 +
                                        (in[j-4][i-2] - in[j-4][i+2] + in[j-2][i-4] - in[j-2][i+4] - in[j+2][i-4] + in[j+2][i+4] - in[j+4][i-2] + in[j+4][i+2]) * 0.000714000 +
                                        (-in[j-4][i-1] + in[j-4][i+1] - in[j-1][i-4] + in[j-1][i+4] + in[j+1][i-4] - in[j+1][i+4] + in[j+4][i-1] - in[j+4][i+1]) * 0.00285600 +
                                        (in[j-3][i-3] - in[j-3][i+3] - in[j+3][i-3] + in[j+3][i+3]) * 0.00145161 +
                                        (-in[j-3][i-2] + in[j-3][i+2] - in[j-2][i-3] + in[j-2][i+3] + in[j+2][i-3] - in[j+2][i+3] + in[j+3][i-2] - in[j+3][i+2]) * 0.00762000 +
                                        (in[j-3][i-1] - in[j-3][i+1] + in[j-1][i-3] - in[j-1][i+3] - in[j+1][i-3] + in[j+1][i+3] - in[j+3][i-1] + in[j+3][i+1]) * 0.0304800 +
                                        (in[j-2][i-2] -  in[j-2][i+2] - in[j+2][i-2] + in[j+2][i+2]) * 0.0400000 +
                                        (-in[j-2][i-1] + in[j-2][i+1] - in[j-1][i-2] + in[j-1][i+2] + in[j+1][i-2] - in[j+1][i+2] + in[j+2][i-1] - in[j+2][i+1]) * 0.160000 +
                                        (in[j-1][i-1] - in[j-1][i+1] - in[j+1][i-1] + in[j+1][i+1]) * 0.640000;
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

	j2d64pt<<<gridconfig, blockconfig>>> (in, out, N);
	cudaMemcpy (h_out, out, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

	cudaFree (in); 
	cudaFree (out);
}
