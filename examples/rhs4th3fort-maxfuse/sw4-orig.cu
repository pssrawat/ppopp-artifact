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

__global__ void sw4 (double * __restrict__ uacc_0, double * __restrict__ uacc_1, double * __restrict__ uacc_2, double * __restrict__ u_0, double * __restrict__ u_1, double * __restrict__ u_2, double * __restrict__ mu, double * __restrict__ la, double * __restrict__ strx, double * __restrict__ stry, double * __restrict__ strz, int N) {
	//Determing the block's indices
	int blockdim_i= (int)(blockDim.x);
	int i0 = (int)(blockIdx.x)*(blockdim_i);
	int i = max (i0, 0) + (int)(threadIdx.x);
	int blockdim_j= (int)(blockDim.y);
	int j0 = (int)(blockIdx.y)*(blockdim_j);
	int j = max (j0, 0) + (int)(threadIdx.y);
	int blockdim_k= (int)(blockDim.z);
	int k0 = (int)(blockIdx.z)*(blockdim_k);
	int k = max (k0, 0) + (int)(threadIdx.z);

	// Assumptions 
	int a1 = 1;
	double h = 3.7;
	double cof = 1e0 / ( h *  h);

	if (i>=2 & j>=2 & k>=2 & i<=N-3 & j<=N-3 & k<=N-3) {
			/* 28 * 3 = 84 flops */
			double mux1 = mu[k*N*N+j*N+i-1] * strx[i-1] - 3e0 / 4 * (mu[k*N*N+j*N+i] * strx[i] + mu[k*N*N+j*N+i-2] * strx[i-2]);
			double mux2 = mu[k*N*N+j*N+i-2] * strx[i-2] + mu[k*N*N+j*N+i+1] * strx[i+1] + 3 * (mu[k*N*N+j*N+i] * strx[i] + mu[k*N*N+j*N+i-1] * strx[i-1]);
			double mux3 = mu[k*N*N+j*N+i-1] * strx[i-1] + mu[k*N*N+j*N+i+2] * strx[i+2] + 3 * (mu[k*N*N+j*N+i+1] * strx[i+1] + mu[k*N*N+j*N+i] * strx[i]);
			double mux4 = mu[k*N*N+j*N+i+1] * strx[i+1] - 3e0 / 4 * (mu[k*N*N+j*N+i] * strx[i] + mu[k*N*N+j*N+i+2] * strx[i+2]);
			double muy1 = mu[k*N*N+(j-1)*N+i] * stry[j-1] - 3e0 / 4 * (mu[k*N*N+j*N+i] * stry[j] + mu[k*N*N+(j-2)*N+i] * stry[j-2]);
			double muy2 = mu[k*N*N+(j-2)*N+i] * stry[j-2] + mu[k*N*N+(j+1)*N+i] * stry[j+1] + 3 * (mu[k*N*N+j*N+i] * stry[j] + mu[k*N*N+(j-1)*N+i] * stry[j-1]);
			double muy3 = mu[k*N*N+(j-1)*N+i] * stry[j-1] + mu[k*N*N+(j+2)*N+i] * stry[j+2] + 3 * (mu[k*N*N+(j+1)*N+i] * stry[j+1] + mu[k*N*N+j*N+i] * stry[j]);
			double muy4 = mu[k*N*N+(j+1)*N+i] * stry[j+1] - 3e0 / 4 * (mu[k*N*N+j*N+i] * stry[j] + mu[k*N*N+(j+2)*N+i] * stry[j+2]);
			double muz1 = mu[(k-1)*N*N+j*N+i] * strz[k-1] - 3e0 / 4 * (mu[k*N*N+j*N+i] * strz[k] + mu[(k-2)*N*N+j*N+i] * strz[k-2]);
			double muz2 = mu[(k-2)*N*N+j*N+i] * strz[k-2] + mu[(k+1)*N*N+j*N+i] * strz[k+1] + 3 * (mu[k*N*N+j*N+i] * strz[k] + mu[(k-1)*N*N+j*N+i] * strz[k-1]);
			double muz3 = mu[(k-1)*N*N+j*N+i] * strz[k-1] + mu[(k+2)*N*N+j*N+i] * strz[k+2] + 3 * (mu[(k+1)*N*N+j*N+i] * strz[k+1] + mu[k*N*N+j*N+i] * strz[k]);
			double muz4 = mu[(k+1)*N*N+j*N+i] * strz[k+1] - 3e0 / 4 * (mu[k*N*N+j*N+i] * strz[k] + mu[(k+2)*N*N+j*N+i] * strz[k+2]);

			/* 78 * 3 = 234 flops */
			double r1 = 1e0 / 6 * (strx[i] * ((2 * mux1 + la[k*N*N+j*N+i-1] * strx[i-1] - 3e0 / 4 * (la[k*N*N+j*N+i] * strx[i] + la[k*N*N+j*N+i-2] * strx[i-2])) * (u_0[k*N*N+j*N+i-2] - u_0[k*N*N+j*N+i]) + (2 * mux2 + la[k*N*N+j*N+i-2] * strx[i-2] + la[k*N*N+j*N+i+1] * strx[i+1] + 3 * (la[k*N*N+j*N+i] * strx[i] + la[k*N*N+j*N+i-1] * strx[i-1])) * (u_0[k*N*N+j*N+i-1] - u_0[k*N*N+j*N+i]) + (2 * mux3 + la[k*N*N+j*N+i-1] * strx[i-1] + la[k*N*N+j*N+i+2] * strx[i+2] + 3 * (la[k*N*N+j*N+i+1] * strx[i+1] + la[k*N*N+j*N+i] * strx[i])) * (u_0[k*N*N+j*N+i+1] - u_0[k*N*N+j*N+i]) + (2 * mux4 + la[k*N*N+j*N+i+1] * strx[i+1] - 3e0 / 4 * (la[k*N*N+j*N+i] * strx[i] + la[k*N*N+j*N+i+2] * strx[i+2])) * (u_0[k*N*N+j*N+i+2] - u_0[k*N*N+j*N+i])) + stry[j] * (muy1 * (u_0[k*N*N+(j-2)*N+i] - u_0[k*N*N+j*N+i]) + muy2 * (u_0[k*N*N+(j-1)*N+i] - u_0[k*N*N+j*N+i]) + muy3 * (u_0[k*N*N+(j+1)*N+i] - u_0[k*N*N+j*N+i]) + muy4 * (u_0[k*N*N+(j+2)*N+i] - u_0[k*N*N+j*N+i])) + strz[k] * (muz1 * (u_0[(k-2)*N*N+j*N+i] - u_0[k*N*N+j*N+i]) + muz2 * (u_0[(k-1)*N*N+j*N+i] - u_0[k*N*N+j*N+i]) + muz3 * (u_0[(k+1)*N*N+j*N+i] - u_0[k*N*N+j*N+i]) + muz4 * (u_0[(k+2)*N*N+j*N+i] - u_0[k*N*N+j*N+i])));
			double r2 = 1e0 / 6 * (strx[i] * (mux1 * (u_1[k*N*N+j*N+i-2] - u_1[k*N*N+j*N+i]) + mux2 * (u_1[k*N*N+j*N+i-1] - u_1[k*N*N+j*N+i]) + mux3 * (u_1[k*N*N+j*N+i+1] - u_1[k*N*N+j*N+i]) + mux4 * (u_1[k*N*N+j*N+i+2] - u_1[k*N*N+j*N+i])) + stry[j] * ((2 * muy1 + la[k*N*N+(j-1)*N+i] * stry[j-1] - 3e0 / 4 * (la[k*N*N+j*N+i] * stry[j] + la[k*N*N+(j-2)*N+i] * stry[j-2])) * (u_1[k*N*N+(j-2)*N+i] - u_1[k*N*N+j*N+i]) + (2 * muy2 + la[k*N*N+(j-2)*N+i] * stry[j-2] + la[k*N*N+(j+1)*N+i] * stry[j+1] + 3 * (la[k*N*N+j*N+i] * stry[j] + la[k*N*N+(j-1)*N+i] * stry[j-1])) * (u_1[k*N*N+(j-1)*N+i] - u_1[k*N*N+j*N+i]) + (2 * muy3 + la[k*N*N+(j-1)*N+i] * stry[j-1] + la[k*N*N+(j+2)*N+i] * stry[j+2] + 3 * (la[k*N*N+(j+1)*N+i] * stry[j+1] + la[k*N*N+j*N+i] * stry[j])) * (u_1[k*N*N+(j+1)*N+i] - u_1[k*N*N+j*N+i]) + (2 * muy4 + la[k*N*N+(j+1)*N+i] * stry[j+1] - 3e0 / 4 * (la[k*N*N+j*N+i] * stry[j] + la[k*N*N+(j+2)*N+i] * stry[j+2])) * (u_1[k*N*N+(j+2)*N+i] - u_1[k*N*N+j*N+i])) + strz[k] * (muz1 * (u_1[(k-2)*N*N+j*N+i] - u_1[k*N*N+j*N+i]) + muz2 * (u_1[(k-1)*N*N+j*N+i] - u_1[k*N*N+j*N+i]) + muz3 * (u_1[(k+1)*N*N+j*N+i] - u_1[k*N*N+j*N+i]) + muz4 * (u_1[(k+2)*N*N+j*N+i] - u_1[k*N*N+j*N+i])));
			double r3 = 1e0 / 6 * (strx[i] * (mux1 * (u_2[k*N*N+j*N+i-2] - u_2[k*N*N+j*N+i]) + mux2 * (u_2[k*N*N+j*N+i-1] - u_2[k*N*N+j*N+i]) + mux3 * (u_2[k*N*N+j*N+i+1] - u_2[k*N*N+j*N+i]) + mux4 * (u_2[k*N*N+j*N+i+2] - u_2[k*N*N+j*N+i])) + stry[j] * (muy1 * (u_2[k*N*N+(j-2)*N+i] - u_2[k*N*N+j*N+i]) + muy2 * (u_2[k*N*N+(j-1)*N+i] - u_2[k*N*N+j*N+i]) + muy3 * (u_2[k*N*N+(j+1)*N+i] - u_2[k*N*N+j*N+i]) + muy4 * (u_2[k*N*N+(j+2)*N+i] - u_2[k*N*N+j*N+i])) + strz[k] * ((2 * muz1 + la[(k-1)*N*N+j*N+i] * strz[k-1] - 3e0 / 4 * (la[k*N*N+j*N+i] * strz[k] + la[(k-2)*N*N+j*N+i] * strz[k-2])) * (u_2[(k-2)*N*N+j*N+i] - u_2[k*N*N+j*N+i]) + (2 * muz2 + la[(k-2)*N*N+j*N+i] * strz[k-2] + la[(k+1)*N*N+j*N+i] * strz[k+1] + 3 * (la[k*N*N+j*N+i] * strz[k] + la[(k-1)*N*N+j*N+i] * strz[k-1])) * (u_2[(k-1)*N*N+j*N+i] - u_2[k*N*N+j*N+i]) + (2 * muz3 + la[(k-1)*N*N+j*N+i] * strz[k-1] + la[(k+2)*N*N+j*N+i] * strz[k+2] + 3 * (la[(k+1)*N*N+j*N+i] * strz[k+1] + la[k*N*N+j*N+i] * strz[k])) * (u_2[(k+1)*N*N+j*N+i] - u_2[k*N*N+j*N+i]) + (2 * muz4 + la[(k+1)*N*N+j*N+i] * strz[k+1] - 3e0 / 4 * (la[k*N*N+j*N+i] * strz[k] + la[(k+2)*N*N+j*N+i] * strz[k+2])) * (u_2[(k+2)*N*N+j*N+i] - u_2[k*N*N+j*N+i])));

			/* 120 * 3 = 360 flops */
			r1 = r1 + strx[i] * stry[j] * (1e0 / 144) * (la[k*N*N+j*N+i-2] * (u_1[k*N*N+(j-2)*N+i-2] - u_1[k*N*N+(j+2)*N+i-2] + 8 * (-u_1[k*N*N+(j-1)*N+i-2] + u_1[k*N*N+(j+1)*N+i-2])) - 8 * (la[k*N*N+j*N+i-1] * (u_1[k*N*N+(j-2)*N+i-1] - u_1[k*N*N+(j+2)*N+i-1] + 8 * (-u_1[k*N*N+(j-1)*N+i-1] + u_1[k*N*N+(j+1)*N+i-1]))) + 8 * (la[k*N*N+j*N+i+1] * (u_1[k*N*N+(j-2)*N+i+1] - u_1[k*N*N+(j+2)*N+i+1] + 8 * (-u_1[k*N*N+(j-1)*N+i+1] + u_1[k*N*N+(j+1)*N+i+1]))) - (la[k*N*N+j*N+i+2] * (u_1[k*N*N+(j-2)*N+i+2] - u_1[k*N*N+(j+2)*N+i+2] + 8 * (-u_1[k*N*N+(j-1)*N+i+2] + u_1[k*N*N+(j+1)*N+i+2])))) + strx[i] * strz[k] * (1e0 / 144) * (la[k*N*N+j*N+i-2] * (u_2[(k-2)*N*N+j*N+i-2] - u_2[(k+2)*N*N+j*N+i-2] + 8 * (-u_2[(k-1)*N*N+j*N+i-2] + u_2[(k+1)*N*N+j*N+i-2])) - 8 * (la[k*N*N+j*N+i-1] * (u_2[(k-2)*N*N+j*N+i-1] - u_2[(k+2)*N*N+j*N+i-1] + 8 * (-u_2[(k-1)*N*N+j*N+i-1] + u_2[(k+1)*N*N+j*N+i-1]))) + 8 * (la[k*N*N+j*N+i+1] * (u_2[(k-2)*N*N+j*N+i+1] - u_2[(k+2)*N*N+j*N+i+1] + 8 * (-u_2[(k-1)*N*N+j*N+i+1] + u_2[(k+1)*N*N+j*N+i+1]))) - (la[k*N*N+j*N+i+2] * (u_2[(k-2)*N*N+j*N+i+2] - u_2[(k+2)*N*N+j*N+i+2] + 8 * (-u_2[(k-1)*N*N+j*N+i+2] + u_2[(k+1)*N*N+j*N+i+2])))) + strx[i] * stry[j] * (1e0 / 144) * (mu[k*N*N+(j-2)*N+i] * (u_1[k*N*N+(j-2)*N+i-2] - u_1[k*N*N+(j-2)*N+i+2] + 8 * (-u_1[k*N*N+(j-2)*N+i-1] + u_1[k*N*N+(j-2)*N+i+1])) - 8 * (mu[k*N*N+(j-1)*N+i] * (u_1[k*N*N+(j-1)*N+i-2] - u_1[k*N*N+(j-1)*N+i+2] + 8 * (-u_1[k*N*N+(j-1)*N+i-1] + u_1[k*N*N+(j-1)*N+i+1]))) + 8 * (mu[k*N*N+(j+1)*N+i] * (u_1[k*N*N+(j+1)*N+i-2] - u_1[k*N*N+(j+1)*N+i+2] + 8 * (-u_1[k*N*N+(j+1)*N+i-1] + u_1[k*N*N+(j+1)*N+i+1]))) - (mu[k*N*N+(j+2)*N+i] * (u_1[k*N*N+(j+2)*N+i-2] - u_1[k*N*N+(j+2)*N+i+2] + 8 * (-u_1[k*N*N+(j+2)*N+i-1] + u_1[k*N*N+(j+2)*N+i+1])))) + strx[i] * strz[k] * (1e0 / 144) * (mu[(k-2)*N*N+j*N+i] * (u_2[(k-2)*N*N+j*N+i-2] - u_2[(k-2)*N*N+j*N+i+2] + 8 * (-u_2[(k-2)*N*N+j*N+i-1] + u_2[(k-2)*N*N+j*N+i+1])) - 8 * (mu[(k-1)*N*N+j*N+i] * (u_2[(k-1)*N*N+j*N+i-2] - u_2[(k-1)*N*N+j*N+i+2] + 8 * (-u_2[(k-1)*N*N+j*N+i-1] + u_2[(k-1)*N*N+j*N+i+1]))) + 8 * (mu[(k+1)*N*N+j*N+i] * (u_2[(k+1)*N*N+j*N+i-2] - u_2[(k+1)*N*N+j*N+i+2] + 8 * (-u_2[(k+1)*N*N+j*N+i-1] + u_2[(k+1)*N*N+j*N+i+1]))) - (mu[(k+2)*N*N+j*N+i] * (u_2[(k+2)*N*N+j*N+i-2] - u_2[(k+2)*N*N+j*N+i+2] + 8 * (-u_2[(k+2)*N*N+j*N+i-1] + u_2[(k+2)*N*N+j*N+i+1])))); 
			r2 = r2 + strx[i] * stry[j] * (1e0 / 144) * (mu[k*N*N+j*N+i-2] * (u_0[k*N*N+(j-2)*N+i-2] - u_0[k*N*N+(j+2)*N+i-2] + 8 * (-u_0[k*N*N+(j-1)*N+i-2] + u_0[k*N*N+(j+1)*N+i-2])) - 8 * (mu[k*N*N+j*N+i-1] * (u_0[k*N*N+(j-2)*N+i-1] - u_0[k*N*N+(j+2)*N+i-1] + 8 * (-u_0[k*N*N+(j-1)*N+i-1] + u_0[k*N*N+(j+1)*N+i-1]))) + 8 * (mu[k*N*N+j*N+i+1] * (u_0[k*N*N+(j-2)*N+i+1] - u_0[k*N*N+(j+2)*N+i+1] + 8 * (-u_0[k*N*N+(j-1)*N+i+1] + u_0[k*N*N+(j+1)*N+i+1]))) - (mu[k*N*N+j*N+i+2] * (u_0[k*N*N+(j-2)*N+i+2] - u_0[k*N*N+(j+2)*N+i+2] + 8 * (-u_0[k*N*N+(j-1)*N+i+2] + u_0[k*N*N+(j+1)*N+i+2])))) + strx[i] * stry[j] * (1e0 / 144) * (la[k*N*N+(j-2)*N+i] * (u_0[k*N*N+(j-2)*N+i-2] - u_0[k*N*N+(j-2)*N+i+2] + 8 * (-u_0[k*N*N+(j-2)*N+i-1] + u_0[k*N*N+(j-2)*N+i+1])) - 8 * (la[k*N*N+(j-1)*N+i] * (u_0[k*N*N+(j-1)*N+i-2] - u_0[k*N*N+(j-1)*N+i+2] + 8 * (-u_0[k*N*N+(j-1)*N+i-1] + u_0[k*N*N+(j-1)*N+i+1]))) + 8 * (la[k*N*N+(j+1)*N+i] * (u_0[k*N*N+(j+1)*N+i-2] - u_0[k*N*N+(j+1)*N+i+2] + 8 * (-u_0[k*N*N+(j+1)*N+i-1] + u_0[k*N*N+(j+1)*N+i+1]))) - (la[k*N*N+(j+2)*N+i] * (u_0[k*N*N+(j+2)*N+i-2] - u_0[k*N*N+(j+2)*N+i+2] + 8 * (-u_0[k*N*N+(j+2)*N+i-1] + u_0[k*N*N+(j+2)*N+i+1])))) + stry[j] * strz[k] * (1e0 / 144) * (la[k*N*N+(j-2)*N+i] * (u_2[(k-2)*N*N+(j-2)*N+i] - u_2[(k+2)*N*N+(j-2)*N+i] + 8 * (-u_2[(k-1)*N*N+(j-2)*N+i] + u_2[(k+1)*N*N+(j-2)*N+i])) - 8 * (la[k*N*N+(j-1)*N+i] * (u_2[(k-2)*N*N+(j-1)*N+i] - u_2[(k+2)*N*N+(j-1)*N+i] + 8 * (-u_2[(k-1)*N*N+(j-1)*N+i] + u_2[(k+1)*N*N+(j-1)*N+i]))) + 8 * (la[k*N*N+(j+1)*N+i] * (u_2[(k-2)*N*N+(j+1)*N+i] - u_2[(k+2)*N*N+(j+1)*N+i] + 8 * (-u_2[(k-1)*N*N+(j+1)*N+i] + u_2[(k+1)*N*N+(j+1)*N+i]))) - (la[k*N*N+(j+2)*N+i] * (u_2[(k-2)*N*N+(j+2)*N+i] - u_2[(k+2)*N*N+(j+2)*N+i] + 8 * (-u_2[(k-1)*N*N+(j+2)*N+i] + u_2[(k+1)*N*N+(j+2)*N+i])))) + stry[j] * strz[k] * (1e0 / 144) * (mu[(k-2)*N*N+j*N+i] * (u_2[(k-2)*N*N+(j-2)*N+i] - u_2[(k-2)*N*N+(j+2)*N+i] + 8 * (-u_2[(k-2)*N*N+(j-1)*N+i] + u_2[(k-2)*N*N+(j+1)*N+i])) - 8 * (mu[(k-1)*N*N+j*N+i] * (u_2[(k-1)*N*N+(j-2)*N+i] - u_2[(k-1)*N*N+(j+2)*N+i] + 8 * (-u_2[(k-1)*N*N+(j-1)*N+i] + u_2[(k-1)*N*N+(j+1)*N+i]))) + 8 * (mu[(k+1)*N*N+j*N+i] * (u_2[(k+1)*N*N+(j-2)*N+i] - u_2[(k+1)*N*N+(j+2)*N+i] + 8 * (-u_2[(k+1)*N*N+(j-1)*N+i] + u_2[(k+1)*N*N+(j+1)*N+i]))) - (mu[(k+2)*N*N+j*N+i] * (u_2[(k+2)*N*N+(j-2)*N+i] - u_2[(k+2)*N*N+(j+2)*N+i] + 8 * (-u_2[(k+2)*N*N+(j-1)*N+i] + u_2[(k+2)*N*N+(j+1)*N+i])))); 
			r3 = r3 + strx[i] * strz[k] * (1e0 / 144) * (mu[k*N*N+j*N+i-2] * (u_0[(k-2)*N*N+j*N+i-2] - u_0[(k+2)*N*N+j*N+i-2] + 8 * (-u_0[(k-1)*N*N+j*N+i-2] + u_0[(k+1)*N*N+j*N+i-2])) - 8 * (mu[k*N*N+j*N+i-1] * (u_0[(k-2)*N*N+j*N+i-1] - u_0[(k+2)*N*N+j*N+i-1] + 8 * (-u_0[(k-1)*N*N+j*N+i-1] + u_0[(k+1)*N*N+j*N+i-1]))) + 8 * (mu[k*N*N+j*N+i+1] * (u_0[(k-2)*N*N+j*N+i+1] - u_0[(k+2)*N*N+j*N+i+1] + 8 * (-u_0[(k-1)*N*N+j*N+i+1] + u_0[(k+1)*N*N+j*N+i+1]))) - (mu[k*N*N+j*N+i+2] * (u_0[(k-2)*N*N+j*N+i+2] - u_0[(k+2)*N*N+j*N+i+2] + 8 * (-u_0[(k-1)*N*N+j*N+i+2] + u_0[(k+1)*N*N+j*N+i+2])))) + stry[j] * strz[k] * (1e0 / 144) * (mu[k*N*N+(j-2)*N+i] * (u_1[(k-2)*N*N+(j-2)*N+i] - u_1[(k+2)*N*N+(j-2)*N+i] + 8 * (-u_1[(k-1)*N*N+(j-2)*N+i] + u_1[(k+1)*N*N+(j-2)*N+i])) - 8 * (mu[k*N*N+(j-1)*N+i] * (u_1[(k-2)*N*N+(j-1)*N+i] - u_1[(k+2)*N*N+(j-1)*N+i] + 8 * (-u_1[(k-1)*N*N+(j-1)*N+i] + u_1[(k+1)*N*N+(j-1)*N+i]))) + 8 * (mu[k*N*N+(j+1)*N+i] * (u_1[(k-2)*N*N+(j+1)*N+i] - u_1[(k+2)*N*N+(j+1)*N+i] + 8 * (-u_1[(k-1)*N*N+(j+1)*N+i] + u_1[(k+1)*N*N+(j+1)*N+i]))) - (mu[k*N*N+(j+2)*N+i] * (u_1[(k-2)*N*N+(j+2)*N+i] - u_1[(k+2)*N*N+(j+2)*N+i] + 8 * (-u_1[(k-1)*N*N+(j+2)*N+i] + u_1[(k+1)*N*N+(j+2)*N+i])))) + strx[i] * strz[k] * (1e0 / 144) * (la[(k-2)*N*N+j*N+i] * (u_0[(k-2)*N*N+j*N+i-2] - u_0[(k-2)*N*N+j*N+i+2] + 8 * (-u_0[(k-2)*N*N+j*N+i-1] + u_0[(k-2)*N*N+j*N+i+1])) - 8 * (la[(k-1)*N*N+j*N+i] * (u_0[(k-1)*N*N+j*N+i-2] - u_0[(k-1)*N*N+j*N+i+2] + 8 * (-u_0[(k-1)*N*N+j*N+i-1] + u_0[(k-1)*N*N+j*N+i+1]))) + 8 * (la[(k+1)*N*N+j*N+i] * (u_0[(k+1)*N*N+j*N+i-2] - u_0[(k+1)*N*N+j*N+i+2] + 8 * (-u_0[(k+1)*N*N+j*N+i-1] + u_0[(k+1)*N*N+j*N+i+1]))) - (la[(k+2)*N*N+j*N+i] * (u_0[(k+2)*N*N+j*N+i-2] - u_0[(k+2)*N*N+j*N+i+2] + 8 * (-u_0[(k+2)*N*N+j*N+i-1] + u_0[(k+2)*N*N+j*N+i+1])))) + stry[j] * strz[k] * (1e0 / 144) * (la[(k-2)*N*N+j*N+i] * (u_1[(k-2)*N*N+(j-2)*N+i] - u_1[(k-2)*N*N+(j+2)*N+i] + 8 * (-u_1[(k-2)*N*N+(j-1)*N+i] + u_1[(k-2)*N*N+(j+1)*N+i])) - 8 * (la[(k-1)*N*N+j*N+i] * (u_1[(k-1)*N*N+(j-2)*N+i] - u_1[(k-1)*N*N+(j+2)*N+i] + 8 * (-u_1[(k-1)*N*N+(j-1)*N+i] + u_1[(k-1)*N*N+(j+1)*N+i]))) + 8 * (la[(k+1)*N*N+j*N+i] * (u_1[(k+1)*N*N+(j-2)*N+i] - u_1[(k+1)*N*N+(j+2)*N+i] + 8 * (-u_1[(k+1)*N*N+(j-1)*N+i] + u_1[(k+1)*N*N+(j+1)*N+i]))) - (la[(k+2)*N*N+j*N+i] * (u_1[(k+2)*N*N+(j-2)*N+i] - u_1[(k+2)*N*N+(j+2)*N+i] + 8 * (-u_1[(k+2)*N*N+(j-1)*N+i] + u_1[(k+2)*N*N+(j+1)*N+i]))));

			/* 3 * 3 = 9 flops */
			uacc_0[k*N*N+j*N+i] = a1 * uacc_0[k*N*N+j*N+i] + cof * r1;
			uacc_1[k*N*N+j*N+i] = a1 * uacc_1[k*N*N+j*N+i] + cof * r2;
			uacc_2[k*N*N+j*N+i] = a1 * uacc_2[k*N*N+j*N+i] + cof * r3;
		} 
}

extern "C" void host_code (double *h_uacc_0, double *h_uacc_1, double *h_uacc_2, double *h_u_0, double *h_u_1, double *h_u_2, double *h_mu, double *h_la, double *h_strx, double *h_stry, double *h_strz, int N) {
	double *uacc_0;
	cudaMalloc (&uacc_0, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for uacc_0\n");
	cudaMemcpy (uacc_0, h_uacc_0, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *uacc_1;
	cudaMalloc (&uacc_1, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for uacc_1\n");
	cudaMemcpy (uacc_1, h_uacc_1, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *uacc_2;
	cudaMalloc (&uacc_2, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for uacc_2\n");
	cudaMemcpy (uacc_2, h_uacc_2, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *u_0;
	cudaMalloc (&u_0, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for u_0\n");
	cudaMemcpy (u_0, h_u_0, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *u_1;
	cudaMalloc (&u_1, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for u_1\n");
	cudaMemcpy (u_1, h_u_1, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *u_2;
	cudaMalloc (&u_2, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for u_2\n");
	cudaMemcpy (u_2, h_u_2, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *mu;
	cudaMalloc (&mu, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for mu\n");
	cudaMemcpy (mu, h_mu, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *la;
	cudaMalloc (&la, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for la\n");
	cudaMemcpy (la, h_la, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *strx;
	cudaMalloc (&strx, sizeof(double)*N);
	check_error ("Failed to allocate device memory for strx\n");
	cudaMemcpy (strx, h_strx, sizeof(double)*N, cudaMemcpyHostToDevice);
	double *stry;
	cudaMalloc (&stry, sizeof(double)*N);
	check_error ("Failed to allocate device memory for stry\n");
	cudaMemcpy (stry, h_stry, sizeof(double)*N, cudaMemcpyHostToDevice);
	double *strz;
	cudaMalloc (&strz, sizeof(double)*N);
	check_error ("Failed to allocate device memory for strz\n");
	cudaMemcpy (strz, h_strz, sizeof(double)*N, cudaMemcpyHostToDevice);

	dim3 blockconfig (16, 4, 4);
	dim3 gridconfig (ceil(N, blockconfig.x), ceil(N, blockconfig.y), ceil(N, blockconfig.z));

	sw4 <<<gridconfig, blockconfig>>> (uacc_0, uacc_1, uacc_2, u_0, u_1, u_2, mu, la, strx, stry, strz, N);

	cudaMemcpy (h_uacc_0, uacc_0, sizeof(double)*N*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy (h_uacc_1, uacc_1, sizeof(double)*N*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy (h_uacc_2, uacc_2, sizeof(double)*N*N*N, cudaMemcpyDeviceToHost);

	cudaFree (uacc_0); 
	cudaFree (uacc_1);
	cudaFree (uacc_2);
	cudaFree (u_0);
	cudaFree (u_1);
	cudaFree (u_2);
	cudaFree (mu);
	cudaFree (la);
	cudaFree (strx);
	cudaFree (stry);
	cudaFree (strz);
}
