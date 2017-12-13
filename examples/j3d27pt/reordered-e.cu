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
double _t_2_;
double _t_4_;
double outkc0jc0ic0;
double _t_3_;
double _t_8_;
double _t_1_;
double _t_0_;
double _t_5_;
double _t_7_;
double _t_10_;
double outkc0jp1ic0;
double _t_6_;
double outkc0jp3ic0;
double _t_11_;
double outkc0jp2ic0;
double _t_9_;

_t_2_ = in[k-1][j-1][i];
_t_2_ += in[k][j-1][i-1];
_t_2_ += in[k][j-1][i+1];
_t_2_ += in[k+1][j-1][i];
_t_2_ += in[k-1][j][i-1];
_t_4_ = in[k-1][j][i-1];
_t_2_ += in[k-1][j][i+1];
_t_4_ += in[k-1][j][i+1];
_t_2_ += in[k+1][j][i-1];
_t_4_ += in[k+1][j][i-1];
_t_2_ += in[k+1][j][i+1];
_t_4_ += in[k+1][j][i+1];
_t_2_ += in[k-1][j+1][i];
_t_2_ += in[k][j+1][i-1];
_t_2_ += in[k][j+1][i+1];
_t_2_ += in[k+1][j+1][i];
outkc0jc0ic0 = 1.031 * _t_2_;
outkc0jc0ic0 += 0.125 * in[k][j][i];
_t_3_ = in[k-1][j+1][i];
_t_3_ += in[k+1][j+1][i];
_t_3_ += in[k][j][i];
_t_3_ += in[k][j+1][i-1];
_t_3_ += in[k][j+1][i+1];
_t_8_ = in[k-1][j+1][i];
_t_8_ += in[k][j+1][i-1];
_t_8_ += in[k][j+1][i+1];
_t_8_ += in[k+1][j+1][i];
_t_1_ = in[k-1][j-1][i-1];
_t_1_ += in[k-1][j-1][i+1];
_t_1_ += in[k+1][j-1][i-1];
_t_1_ += in[k+1][j-1][i+1];
_t_1_ += in[k-1][j+1][i-1];
_t_1_ += in[k-1][j+1][i+1];
_t_1_ += in[k+1][j+1][i-1];
_t_1_ += in[k+1][j+1][i+1];
outkc0jc0ic0 += 0.75 * _t_1_;
_t_0_ = in[k][j-1][i];
_t_0_ += in[k-1][j][i];
_t_5_ = in[k-1][j][i];
_t_5_ += in[k-1][j+1][i-1];
_t_5_ += in[k-1][j+1][i+1];
_t_5_ += in[k+1][j+1][i-1];
_t_5_ += in[k+1][j+1][i+1];
_t_7_ = in[k-1][j+1][i-1];
_t_7_ += in[k-1][j+1][i+1];
_t_7_ += in[k+1][j+1][i-1];
_t_7_ += in[k+1][j+1][i+1];
_t_0_ += in[k][j][i-1];
_t_5_ += in[k][j][i-1];
_t_0_ += in[k][j][i+1];
_t_5_ += in[k][j][i+1];
_t_0_ += in[k+1][j][i];
_t_5_ += in[k+1][j][i];
_t_0_ += in[k][j+1][i];
outkc0jc0ic0 += 1.14 * _t_0_;
out[k][j][i] = outkc0jc0ic0;
_t_4_ += in[k-1][j+2][i-1];
_t_8_ += in[k-1][j+2][i-1];
_t_10_ = in[k-1][j+2][i-1];
_t_4_ += in[k-1][j+2][i+1];
_t_8_ += in[k-1][j+2][i+1];
_t_10_ += in[k-1][j+2][i+1];
_t_4_ += in[k+1][j+2][i-1];
_t_8_ += in[k+1][j+2][i-1];
_t_10_ += in[k+1][j+2][i-1];
_t_4_ += in[k+1][j+2][i+1];
_t_8_ += in[k+1][j+2][i+1];
_t_10_ += in[k+1][j+2][i+1];
outkc0jp1ic0 = 0.125 * in[k][j+1][i];
outkc0jp1ic0 += 0.75 * _t_4_;
_t_6_ = in[k][j+1][i];
_t_10_ += in[k-1][j+4][i-1];
_t_10_ += in[k-1][j+4][i+1];
_t_10_ += in[k+1][j+4][i-1];
_t_10_ += in[k+1][j+4][i+1];
outkc0jp3ic0 = 0.75 * _t_10_;
_t_6_ += in[k][j+3][i];
outkc0jp3ic0 += 0.125 * in[k][j+3][i];
_t_5_ += in[k-1][j+2][i];
_t_6_ += in[k-1][j+2][i];
_t_11_ = in[k-1][j+2][i];
_t_5_ += in[k][j+2][i-1];
_t_6_ += in[k][j+2][i-1];
_t_11_ += in[k][j+2][i-1];
_t_5_ += in[k][j+2][i+1];
_t_6_ += in[k][j+2][i+1];
_t_11_ += in[k][j+2][i+1];
_t_5_ += in[k+1][j+2][i];
outkc0jp1ic0 += 1.031 * _t_5_;
_t_6_ += in[k+1][j+2][i];
_t_11_ += in[k+1][j+2][i];
outkc0jp2ic0 = 1.14 * _t_6_;
_t_7_ += in[k-1][j+3][i-1];
_t_11_ += in[k-1][j+3][i-1];
_t_7_ += in[k-1][j+3][i+1];
_t_11_ += in[k-1][j+3][i+1];
_t_7_ += in[k+1][j+3][i-1];
_t_11_ += in[k+1][j+3][i-1];
_t_7_ += in[k+1][j+3][i+1];
outkc0jp2ic0 += 0.75 * _t_7_;
_t_11_ += in[k+1][j+3][i+1];
_t_11_ += in[k-1][j+4][i];
_t_11_ += in[k][j+4][i-1];
_t_11_ += in[k][j+4][i+1];
_t_11_ += in[k+1][j+4][i];
outkc0jp3ic0 += 1.031 * _t_11_;
_t_3_ += in[k][j+2][i];
outkc0jp1ic0 += 1.14 * _t_3_;
out[k][j+1][i] = outkc0jp1ic0;
outkc0jp2ic0 += 0.125 * in[k][j+2][i];
_t_9_ = in[k][j+2][i];
_t_8_ += in[k-1][j+3][i];
_t_9_ += in[k-1][j+3][i];
_t_8_ += in[k][j+3][i-1];
_t_9_ += in[k][j+3][i-1];
_t_8_ += in[k][j+3][i+1];
_t_9_ += in[k][j+3][i+1];
_t_8_ += in[k+1][j+3][i];
outkc0jp2ic0 += 1.031 * _t_8_;
out[k][j+2][i] = outkc0jp2ic0;
_t_9_ += in[k+1][j+3][i];
_t_9_ += in[k][j+4][i];
outkc0jp3ic0 += 1.14 * _t_9_;
out[k][j+3][i] = outkc0jp3ic0;
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
