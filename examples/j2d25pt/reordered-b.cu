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
		double _t_2_ = in[j-2][i];
		_t_2_ += in[j+2][i];
		double outjc0ic0 = 0.3 * _t_2_;
		double _t_0_ = in[j-2][i-2];
		_t_0_ += in[j-2][i+2];
		_t_0_ += in[j+2][i-2];
		_t_0_ += in[j+2][i+2];
		outjc0ic0 += 0.1 * _t_0_;
		double _t_1_ = in[j-2][i-1];
		_t_1_ += in[j-2][i+1];
		_t_1_ += in[j+2][i-1];
		_t_1_ += in[j+2][i+1];
		outjc0ic0 += 0.2 * _t_1_;
		double _t_5_ = in[j-1][i];
		double _t_10_ = in[j-1][i];
		_t_5_ += in[j+1][i];
		outjc0ic0 += 1.3 * _t_5_;
		double _t_6_ = in[j][i-2];
		double _t_11_ = in[j][i-2];
		double _t_16_ = in[j][i-2];
		_t_6_ += in[j][i+2];
		_t_11_ += in[j][i+2];
		_t_16_ += in[j][i+2];
		outjc0ic0 += 2.1 * _t_6_;
		double _t_7_ = in[j][i-1];
		double _t_12_ = in[j][i-1];
		double _t_17_ = in[j][i-1];
		_t_7_ += in[j][i+1];
		_t_12_ += in[j][i+1];
		_t_17_ += in[j][i+1];
		outjc0ic0 += 2.2 * _t_7_;
		double _t_3_ = in[j-1][i-2];
		double _t_8_ = in[j-1][i-2];
		_t_3_ += in[j-1][i+2];
		_t_8_ += in[j-1][i+2];
		_t_3_ += in[j+1][i-2];
		_t_3_ += in[j+1][i+2];
		outjc0ic0 += 1.1 * _t_3_;
		double _t_4_ = in[j-1][i-1];
		double _t_9_ = in[j-1][i-1];
		_t_4_ += in[j-1][i+1];
		_t_9_ += in[j-1][i+1];
		_t_4_ += in[j+1][i-1];
		_t_4_ += in[j+1][i+1];
		outjc0ic0 += 1.2 * _t_4_;
		outjc0ic0 += 2.3 * in[j][i];
		double _t_13_ = in[j][i];
		double _t_18_ = in[j][i];

		_t_8_ += in[j+3][i-2];
		_t_8_ += in[j+3][i+2];
		double outjp1ic0 = 0.1 * _t_8_;
		_t_9_ += in[j+3][i-1];
		_t_9_ += in[j+3][i+1];
		outjp1ic0 += 0.2 * _t_9_;
		_t_10_ += in[j+3][i];
		outjp1ic0 += 0.3 * _t_10_;
		_t_11_ += in[j+2][i-2];
		_t_11_ += in[j+2][i+2];
		outjp1ic0 += 1.1 * _t_11_;
		_t_12_ += in[j+2][i-1];
		_t_12_ += in[j+2][i+1];
		outjp1ic0 += 1.2 * _t_12_;
		_t_13_ += in[j+2][i];
		outjp1ic0 += 1.3 * _t_13_;
		double _t_14_ = in[j+1][i-2];
		_t_14_ += in[j+1][i+2];
		outjp1ic0 += 2.1 * _t_14_;
		double _t_15_ = in[j+1][i-1];
		_t_15_ += in[j+1][i+1];
		outjp1ic0 += 2.2 * _t_15_;
		outjp1ic0 += 2.3 * in[j+1][i];

		_t_16_ += in[j+4][i-2];
		double _t_27_ = in[j+4][i-2];
		_t_16_ += in[j+4][i+2];
		_t_27_ += in[j+4][i+2];
		double outjp2ic0 = 0.1 * _t_16_;
		_t_17_ += in[j+4][i-1];
		double _t_28_ = in[j+4][i-1];
		_t_17_ += in[j+4][i+1];
		_t_28_ += in[j+4][i+1];
		outjp2ic0 += 0.2 * _t_17_;
		_t_18_ += in[j+4][i];
		double _t_29_ = in[j+4][i];
		outjp2ic0 += 0.3 * _t_18_;
		double _t_19_ = in[j+1][i-2];
		double _t_24_ = in[j+1][i-2];
		_t_19_ += in[j+1][i+2];
		_t_24_ += in[j+1][i+2];
		_t_19_ += in[j+3][i-2];
		double _t_30_ = in[j+3][i-2];
		_t_19_ += in[j+3][i+2];
		_t_30_ += in[j+3][i+2];
		outjp2ic0 += 1.1 * _t_19_;
		double _t_20_ = in[j+1][i-1];
		double _t_25_ = in[j+1][i-1];
		_t_20_ += in[j+1][i+1];
		_t_25_ += in[j+1][i+1];
		_t_20_ += in[j+3][i-1];
		double _t_31_ = in[j+3][i-1];
		_t_20_ += in[j+3][i+1];
		_t_31_ += in[j+3][i+1];
		outjp2ic0 += 1.2 * _t_20_;
		double _t_21_ = in[j+1][i];
		double _t_26_ = in[j+1][i];
		_t_21_ += in[j+3][i];
		double outjp3ic0 = 2.3 * in[j+3][i];
		outjp2ic0 += 1.3 * _t_21_;
		double _t_22_ = in[j+2][i-2];
		_t_27_ += in[j+2][i-2];
		_t_22_ += in[j+2][i+2];
		_t_27_ += in[j+2][i+2];
		outjp2ic0 += 2.1 * _t_22_;
		double _t_23_ = in[j+2][i-1];
		_t_28_ += in[j+2][i-1];
		_t_23_ += in[j+2][i+1];
		_t_28_ += in[j+2][i+1];
		outjp2ic0 += 2.2 * _t_23_;
		outjp2ic0 += 2.3 * in[j+2][i];
		_t_29_ += in[j+2][i];

		outjp3ic0 += 1.1 * _t_27_;
		outjp3ic0 += 1.2 * _t_28_;
		_t_24_ += in[j+5][i-2];
		_t_24_ += in[j+5][i+2];
		outjp3ic0 += 0.1 * _t_24_;
		_t_25_ += in[j+5][i-1];
		_t_25_ += in[j+5][i+1];
		outjp3ic0 += 0.2 * _t_25_;
		outjp3ic0 += 1.3 * _t_29_;
		outjp3ic0 += 2.1 * _t_30_;
		outjp3ic0 += 2.2 * _t_31_;
		_t_26_ += in[j+5][i];
		outjp3ic0 += 0.3 * _t_26_;

		out[j][i] = outjc0ic0;
		out[j+1][i] = outjp1ic0;
		out[j+2][i] = outjp2ic0;
		out[j+3][i] = outjp3ic0;
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
