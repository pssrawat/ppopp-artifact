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

__global__ void __launch_bounds__ (128,2) sw4_1 (double * uacc_in_0, double * uacc_in_1, double * uacc_in_2, double * __restrict__ u_in_0, double * __restrict__ u_in_1, double * __restrict__ u_in_2, double * __restrict__ mu_in, double * __restrict__ la_in, double * strx, double * stry, double * strz, int N) {
	//Determing the block's indices
	int blockdim_i= (int)(blockDim.x);
	int i0 = (int)(blockIdx.x)*(blockdim_i);
	int i = max (i0, 0) + (int)(threadIdx.x);
	int blockdim_j= (int)(blockDim.y);
	int j0 = (int)(blockIdx.y)*(blockdim_j);
	int j = max (j0, 0) + (int)(threadIdx.y);
	// Assumptions 
	int a1 = 1;
	double h = 3.7;
	double cof = 1e0 / ( h *  h);

	double (*uacc_0)[304][304] = (double (*)[304][304])uacc_in_0;
	double (*uacc_1)[304][304] = (double (*)[304][304])uacc_in_1;
	double (*uacc_2)[304][304] = (double (*)[304][304])uacc_in_2;
	double (*u_0)[304][304] = (double (*)[304][304])u_in_0;
	double (*u_1)[304][304] = (double (*)[304][304])u_in_1;
	double (*u_2)[304][304] = (double (*)[304][304])u_in_2;
	double (*mu)[304][304] = (double (*)[304][304])mu_in;
	double (*la)[304][304] = (double (*)[304][304])la_in;

	double a_mux1, a_mux2, a_mux3, a_mux4, a_muy1, a_muy2, a_muy3, a_muy4, a_muz1, a_muz2, a_muz3, a_muz4;
	double b_mux1, b_mux2, b_mux3, b_mux4, b_muy1, b_muy2, b_muy3, b_muy4, b_muz1, b_muz2, b_muz3, b_muz4;
	double a_r1, b_r1;

	if (i>=2 & j>=2 & i<=N-3 & j<=N-3) {
#pragma unroll 3 
		for (int k=2; k<=N-3; k+=2) {
			a_mux1 = mu[k][j][i-1] * strx[i-1] - 3e0 / 4 * mu[k][j][i] * strx[i] - 3e0 / 4 * mu[k][j][i-2] * strx[i-2];
			a_mux2 = mu[k][j][i-2] * strx[i-2] + mu[k][j][i+1] * strx[i+1] + 3.0 * mu[k][j][i] * strx[i] + 3.0 * mu[k][j][i-1] * strx[i-1];
			a_mux3 = mu[k][j][i-1] * strx[i-1] + mu[k][j][i+2] * strx[i+2] + 3.0 * mu[k][j][i+1] * strx[i+1] + 3.0 * mu[k][j][i] * strx[i];
			a_mux4 = mu[k][j][i+1] * strx[i+1] - 3e0 / 4 * mu[k][j][i] * strx[i] - 3e0 / 4 *  mu[k][j][i+2] * strx[i+2];
			a_muy1 = mu[k][j-1][i] * stry[j-1] - 3e0 / 4 * mu[k][j][i] * stry[j] -3e0 / 4 * mu[k][j-2][i] * stry[j-2];
			a_muy2 = mu[k][j-2][i] * stry[j-2] + mu[k][j+1][i] * stry[j+1] + 3.0 * mu[k][j][i] * stry[j] +     3.0 * mu[k][j-1][i] * stry[j-1];
			a_muy3 = mu[k][j-1][i] * stry[j-1] + mu[k][j+2][i] * stry[j+2] + 3.0 * mu[k][j+1][i] * stry[j+1] + 3.0 * mu[k][j][i] * stry[j];
			a_muy4 = mu[k][j+1][i] * stry[j+1] - 3e0 / 4 * mu[k][j][i] * stry[j] - 3e0 / 4 * mu[k][j+2][i] * stry[j+2];
			a_muz1 = mu[k-1][j][i] * strz[k-1] - 3e0 / 4 * mu[k][j][i] * strz[k] - 3e0 / 4 * mu[k-2][j][i] * strz[k-2];
			a_muz2 = mu[k-2][j][i] * strz[k-2] + mu[k+1][j][i] * strz[k+1] + 3.0 * mu[k][j][i] * strz[k] + 3.0 * mu[k-1][j][i] * strz[k-1];
			a_muz3 = mu[k-1][j][i] * strz[k-1] + mu[k+2][j][i] * strz[k+2] + 3.0 * mu[k+1][j][i] * strz[k+1] + 3.0 * mu[k][j][i] * strz[k];
			a_muz4 = mu[k+1][j][i] * strz[k+1] - 3e0 / 4 * mu[k][j][i] * strz[k] - 3e0 /4  * mu[k+2][j][i] * strz[k+2];

			a_r1 = 1e0 / 6 * (strx[i] * ((2 * a_mux1 + la[k][j][i-1] * strx[i-1] - 3e0 / 4 * la[k][j][i] * strx[i] - 3e0 / 4 * la[k][j][i-2] * strx[i-2]) * (u_0[k][j][i-2] - u_0[k][j][i]) + 
						(2 * a_mux2 + la[k][j][i-2] * strx[i-2] + la[k][j][i+1] * strx[i+1] + 3 * la[k][j][i] * strx[i] + 3 * la[k][j][i-1] * strx[i-1]) * (u_0[k][j][i-1] - u_0[k][j][i]) + 
						(2 * a_mux3 + la[k][j][i-1] * strx[i-1] + la[k][j][i+2] * strx[i+2] + 3 * la[k][j][i+1] * strx[i+1] + 3 * la[k][j][i] * strx[i]) * (u_0[k][j][i+1] - u_0[k][j][i]) + 
						(2 * a_mux4 + la[k][j][i+1] * strx[i+1] - 3e0 / 4 * la[k][j][i] * strx[i] - 3e0 / 4 * la[k][j][i+2] * strx[i+2]) * (u_0[k][j][i+2] - u_0[k][j][i]))
					+ stry[j] * (a_muy1 * (u_0[k][j-2][i] - u_0[k][j][i]) + a_muy2 * (u_0[k][j-1][i] - u_0[k][j][i]) + a_muy3 * (u_0[k][j+1][i] - u_0[k][j][i]) + a_muy4 * (u_0[k][j+2][i] - u_0[k][j][i])) + strz[k] * (a_muz1 * (u_0[k-2][j][i] - u_0[k][j][i]) + a_muz2 * (u_0[k-1][j][i] - u_0[k][j][i]) + a_muz3 * (u_0[k+1][j][i] - u_0[k][j][i]) + a_muz4 * (u_0[k+2][j][i] - u_0[k][j][i])));

			a_r1 += strx[i] * stry[j] * (1e0 / 144) * (la[k][j][i-2] * (u_1[k][j-2][i-2] - u_1[k][j+2][i-2] + 8 * (-u_1[k][j-1][i-2] + u_1[k][j+1][i-2])) - 8 * (la[k][j][i-1] * (u_1[k][j-2][i-1] - u_1[k][j+2][i-1] + 8 * (-u_1[k][j-1][i-1] + u_1[k][j+1][i-1]))) + 8 * (la[k][j][i+1] * (u_1[k][j-2][i+1] - u_1[k][j+2][i+1] + 8 * (-u_1[k][j-1][i+1] + u_1[k][j+1][i+1]))) - (la[k][j][i+2] * (u_1[k][j-2][i+2] - u_1[k][j+2][i+2] + 8 * (-u_1[k][j-1][i+2] + u_1[k][j+1][i+2]))));
			a_r1 += strx[i] * strz[k] * (1e0 / 144) * (la[k][j][i-2] * (u_2[k-2][j][i-2] - u_2[k+2][j][i-2] + 8 * (-u_2[k-1][j][i-2] + u_2[k+1][j][i-2])) - 8 * (la[k][j][i-1] * (u_2[k-2][j][i-1] - u_2[k+2][j][i-1] + 8 * (-u_2[k-1][j][i-1] + u_2[k+1][j][i-1]))) + 8 * (la[k][j][i+1] * (u_2[k-2][j][i+1] - u_2[k+2][j][i+1] + 8 * (-u_2[k-1][j][i+1] + u_2[k+1][j][i+1]))) - (la[k][j][i+2] * (u_2[k-2][j][i+2] - u_2[k+2][j][i+2] + 8 * (-u_2[k-1][j][i+2] + u_2[k+1][j][i+2]))));
			a_r1 += strx[i] * stry[j] * (1e0 / 144) * (mu[k][j-2][i] * (u_1[k][j-2][i-2] - u_1[k][j-2][i+2] + 8 * (-u_1[k][j-2][i-1] + u_1[k][j-2][i+1])) - 8 * (mu[k][j-1][i] * (u_1[k][j-1][i-2] - u_1[k][j-1][i+2] + 8 * (-u_1[k][j-1][i-1] + u_1[k][j-1][i+1]))) + 8 * (mu[k][j+1][i] * (u_1[k][j+1][i-2] - u_1[k][j+1][i+2] + 8 * (-u_1[k][j+1][i-1] + u_1[k][j+1][i+1]))) - (mu[k][j+2][i] * (u_1[k][j+2][i-2] - u_1[k][j+2][i+2] + 8 * (-u_1[k][j+2][i-1] + u_1[k][j+2][i+1]))));
			a_r1 += strx[i] * strz[k] * (1e0 / 144) * (mu[k-2][j][i] * (u_2[k-2][j][i-2] - u_2[k-2][j][i+2] + 8 * (-u_2[k-2][j][i-1] + u_2[k-2][j][i+1])) - 8 * (mu[k-1][j][i] * (u_2[k-1][j][i-2] - u_2[k-1][j][i+2] + 8 * (-u_2[k-1][j][i-1] + u_2[k-1][j][i+1]))) + 8 * (mu[k+1][j][i] * (u_2[k+1][j][i-2] - u_2[k+1][j][i+2] + 8 * (-u_2[k+1][j][i-1] + u_2[k+1][j][i+1]))) - (mu[k+2][j][i] * (u_2[k+2][j][i-2] - u_2[k+2][j][i+2] + 8 * (-u_2[k+2][j][i-1] + u_2[k+2][j][i+1]))));
			uacc_0[k][j][i] = a1 * uacc_0[k][j][i] + cof * a_r1;

			b_mux1 = mu[k+1][j][i-1] * strx[i-1] - 3e0 / 4 * mu[k+1][j][i] * strx[i] - 3e0 / 4 * mu[k+1][j][i-2] * strx[i-2];
			b_mux2 = mu[k+1][j][i-2] * strx[i-2] + mu[k+1][j][i+1] * strx[i+1] + 3.0 * mu[k+1][j][i] * strx[i] + 3.0 * mu[k+1][j][i-1] * strx[i-1];
			b_mux3 = mu[k+1][j][i-1] * strx[i-1] + mu[k+1][j][i+2] * strx[i+2] + 3.0 * mu[k+1][j][i+1] * strx[i+1] + 3.0 * mu[k+1][j][i] * strx[i];
			b_mux4 = mu[k+1][j][i+1] * strx[i+1] - 3e0 / 4 * mu[k+1][j][i] * strx[i] - 3e0 / 4 *  mu[k+1][j][i+2] * strx[i+2];
			b_muy1 = mu[k+1][j-1][i] * stry[j-1] - 3e0 / 4 * mu[k+1][j][i] * stry[j] -3e0 / 4 * mu[k+1][j-2][i] * stry[j-2];
			b_muy2 = mu[k+1][j-2][i] * stry[j-2] + mu[k+1][j+1][i] * stry[j+1] + 3.0 * mu[k+1][j][i] * stry[j] +     3.0 * mu[k+1][j-1][i] * stry[j-1];
			b_muy3 = mu[k+1][j-1][i] * stry[j-1] + mu[k+1][j+2][i] * stry[j+2] + 3.0 * mu[k+1][j+1][i] * stry[j+1] + 3.0 * mu[k+1][j][i] * stry[j];
			b_muy4 = mu[k+1][j+1][i] * stry[j+1] - 3e0 / 4 * mu[k+1][j][i] * stry[j] - 3e0 / 4 * mu[k+1][j+2][i] * stry[j+2];
			b_muz1 = mu[k+1-1][j][i] * strz[k+1-1] - 3e0 / 4 * mu[k+1][j][i] * strz[k+1] - 3e0 / 4 * mu[k+1-2][j][i] * strz[k+1-2];
			b_muz2 = mu[k+1-2][j][i] * strz[k+1-2] + mu[k+1+1][j][i] * strz[k+1+1] + 3.0 * mu[k+1][j][i] * strz[k+1] + 3.0 * mu[k+1-1][j][i] * strz[k+1-1];
			b_muz3 = mu[k+1-1][j][i] * strz[k+1-1] + mu[k+1+2][j][i] * strz[k+1+2] + 3.0 * mu[k+1+1][j][i] * strz[k+1+1] + 3.0 * mu[k+1][j][i] * strz[k+1];
			b_muz4 = mu[k+1+1][j][i] * strz[k+1+1] - 3e0 / 4 * mu[k+1][j][i] * strz[k+1] - 3e0 /4  * mu[k+1+2][j][i] * strz[k+1+2];

			b_r1 = 1e0 / 6 * (strx[i] * ((2 * b_mux1 + la[k+1][j][i-1] * strx[i-1] - 3e0 / 4 * la[k+1][j][i] * strx[i] - 3e0 / 4 * la[k+1][j][i-2] * strx[i-2]) * (u_0[k+1][j][i-2] - u_0[k+1][j][i]) + 
						(2 * b_mux2 + la[k+1][j][i-2] * strx[i-2] + la[k+1][j][i+1] * strx[i+1] + 3 * la[k+1][j][i] * strx[i] + 3 * la[k+1][j][i-1] * strx[i-1]) * (u_0[k+1][j][i-1] - u_0[k+1][j][i]) + 
						(2 * b_mux3 + la[k+1][j][i-1] * strx[i-1] + la[k+1][j][i+2] * strx[i+2] + 3 * la[k+1][j][i+1] * strx[i+1] + 3 * la[k+1][j][i] * strx[i]) * (u_0[k+1][j][i+1] - u_0[k+1][j][i]) + 
						(2 * b_mux4 + la[k+1][j][i+1] * strx[i+1] - 3e0 / 4 * la[k+1][j][i] * strx[i] - 3e0 / 4 * la[k+1][j][i+2] * strx[i+2]) * (u_0[k+1][j][i+2] - u_0[k+1][j][i]))
					+ stry[j] * (b_muy1 * (u_0[k+1][j-2][i] - u_0[k+1][j][i]) + b_muy2 * (u_0[k+1][j-1][i] - u_0[k+1][j][i]) + b_muy3 * (u_0[k+1][j+1][i] - u_0[k+1][j][i]) + b_muy4 * (u_0[k+1][j+2][i] - u_0[k+1][j][i])) + strz[k+1] * (b_muz1 * (u_0[k+1-2][j][i] - u_0[k+1][j][i]) + b_muz2 * (u_0[k+1-1][j][i] - u_0[k+1][j][i]) + b_muz3 * (u_0[k+1+1][j][i] - u_0[k+1][j][i]) + b_muz4 * (u_0[k+1+2][j][i] - u_0[k+1][j][i])));


			b_r1 += strx[i] * stry[j] * (1e0 / 144) * (la[k+1][j][i-2] * (u_1[k+1][j-2][i-2] - u_1[k+1][j+2][i-2] + 8 * (-u_1[k+1][j-1][i-2] + u_1[k+1][j+1][i-2])) - 8 * (la[k+1][j][i-1] * (u_1[k+1][j-2][i-1] - u_1[k+1][j+2][i-1] + 8 * (-u_1[k+1][j-1][i-1] + u_1[k+1][j+1][i-1]))) + 8 * (la[k+1][j][i+1] * (u_1[k+1][j-2][i+1] - u_1[k+1][j+2][i+1] + 8 * (-u_1[k+1][j-1][i+1] + u_1[k+1][j+1][i+1]))) - (la[k+1][j][i+2] * (u_1[k+1][j-2][i+2] - u_1[k+1][j+2][i+2] + 8 * (-u_1[k+1][j-1][i+2] + u_1[k+1][j+1][i+2]))));
			b_r1 += strx[i] * strz[k+1] * (1e0 / 144) * (la[k+1][j][i-2] * (u_2[k+1-2][j][i-2] - u_2[k+1+2][j][i-2] + 8 * (-u_2[k+1-1][j][i-2] + u_2[k+1+1][j][i-2])) - 8 * (la[k+1][j][i-1] * (u_2[k+1-2][j][i-1] - u_2[k+1+2][j][i-1] + 8 * (-u_2[k+1-1][j][i-1] + u_2[k+1+1][j][i-1]))) + 8 * (la[k+1][j][i+1] * (u_2[k+1-2][j][i+1] - u_2[k+1+2][j][i+1] + 8 * (-u_2[k+1-1][j][i+1] + u_2[k+1+1][j][i+1]))) - (la[k+1][j][i+2] * (u_2[k+1-2][j][i+2] - u_2[k+1+2][j][i+2] + 8 * (-u_2[k+1-1][j][i+2] + u_2[k+1+1][j][i+2]))));
			b_r1 += strx[i] * stry[j] * (1e0 / 144) * (mu[k+1][j-2][i] * (u_1[k+1][j-2][i-2] - u_1[k+1][j-2][i+2] + 8 * (-u_1[k+1][j-2][i-1] + u_1[k+1][j-2][i+1])) - 8 * (mu[k+1][j-1][i] * (u_1[k+1][j-1][i-2] - u_1[k+1][j-1][i+2] + 8 * (-u_1[k+1][j-1][i-1] + u_1[k+1][j-1][i+1]))) + 8 * (mu[k+1][j+1][i] * (u_1[k+1][j+1][i-2] - u_1[k+1][j+1][i+2] + 8 * (-u_1[k+1][j+1][i-1] + u_1[k+1][j+1][i+1]))) - (mu[k+1][j+2][i] * (u_1[k+1][j+2][i-2] - u_1[k+1][j+2][i+2] + 8 * (-u_1[k+1][j+2][i-1] + u_1[k+1][j+2][i+1]))));
			b_r1 += strx[i] * strz[k+1] * (1e0 / 144) * (mu[k+1-2][j][i] * (u_2[k+1-2][j][i-2] - u_2[k+1-2][j][i+2] + 8 * (-u_2[k+1-2][j][i-1] + u_2[k+1-2][j][i+1])) - 8 * (mu[k+1-1][j][i] * (u_2[k+1-1][j][i-2] - u_2[k+1-1][j][i+2] + 8 * (-u_2[k+1-1][j][i-1] + u_2[k+1-1][j][i+1]))) + 8 * (mu[k+1+1][j][i] * (u_2[k+1+1][j][i-2] - u_2[k+1+1][j][i+2] + 8 * (-u_2[k+1+1][j][i-1] + u_2[k+1+1][j][i+1]))) - (mu[k+1+2][j][i] * (u_2[k+1+2][j][i-2] - u_2[k+1+2][j][i+2] + 8 * (-u_2[k+1+2][j][i-1] + u_2[k+1+2][j][i+1]))));
			uacc_0[k+1][j][i] = a1 * uacc_0[k+1][j][i] + cof * b_r1;
		}
	} 
}


__global__ void __launch_bounds__ (128,2) sw4_2 (double * uacc_in_0, double * uacc_in_1, double * uacc_in_2, double * __restrict__ u_in_0, double * __restrict__ u_in_1, double * __restrict__ u_in_2, double * __restrict__ mu_in, double * __restrict__ la_in, double * strx, double * stry, double * strz, int N) {
	//Determing the block's indices
	int blockdim_i= (int)(blockDim.x);
	int i0 = (int)(blockIdx.x)*(blockdim_i);
	int i = max (i0, 0) + (int)(threadIdx.x);
	int blockdim_j= (int)(blockDim.y);
	int j0 = (int)(blockIdx.y)*(blockdim_j);
	int j = max (j0, 0) + (int)(threadIdx.y);

	// Assumptions 
	int a1 = 1;
	double h = 3.7;
	double cof = 1e0 / ( h *  h);

	double (*uacc_0)[304][304] = (double (*)[304][304])uacc_in_0;
	double (*uacc_1)[304][304] = (double (*)[304][304])uacc_in_1;
	double (*uacc_2)[304][304] = (double (*)[304][304])uacc_in_2;
	double (*u_0)[304][304] = (double (*)[304][304])u_in_0;
	double (*u_1)[304][304] = (double (*)[304][304])u_in_1;
	double (*u_2)[304][304] = (double (*)[304][304])u_in_2;
	double (*mu)[304][304] = (double (*)[304][304])mu_in;
	double (*la)[304][304] = (double (*)[304][304])la_in;

	double a_mux1, a_mux2, a_mux3, a_mux4, a_muy1, a_muy2, a_muy3, a_muy4, a_muz1, a_muz2, a_muz3, a_muz4;
	double b_mux1, b_mux2, b_mux3, b_mux4, b_muy1, b_muy2, b_muy3, b_muy4, b_muz1, b_muz2, b_muz3, b_muz4;
	double a_r2, b_r2;
	if (i>=2 & j>=2 & i<=N-3 & j<=N-3) {
#pragma unroll 3 
		for (int k=2; k<=N-3; k+=2) {
double a_mux1;
double a_mux2;
double a_mux3;
double a_mux4;
double a_muy1;
double a_muy2;
double a_muy3;
double a_muy4;
double a_muz1;
double a_muz2;
double a_muz3;
double a_muz4;
double _t_1_;
double a_r2;
double _t_7_;
double _t_8_;
double _t_6_;
double _t_9_;
double _t_10_;
double _t_11_;
double _t_12_;
double _t_13_;
double _t_14_;
double _t_15_;
double _t_21_;
double _t_22_;
double _t_25_;
double _t_27_;
double _t_30_;
double _t_33_;
double _t_20_;
double _t_34_;
double _t_35_;
double _t_38_;
double _t_40_;
double _t_43_;
double _t_46_;
double _t_47_;
double _t_48_;
double _t_51_;
double _t_53_;
double _t_56_;
double _t_59_;
double _t_60_;
double _t_61_;
double _t_64_;
double _t_66_;
double _t_69_;
double _t_72_;
double uacc_1kc0jc0ic0;
double b_mux1;
double b_mux2;
double b_mux3;
double b_mux4;
double b_muy1;
double b_muy2;
double b_muy3;
double b_muy4;
double b_muz1;
double b_muz2;
double b_muz3;
double b_muz4;
double _t_74_;
double b_r2;
double _t_80_;
double _t_81_;
double _t_79_;
double _t_82_;
double _t_83_;
double _t_84_;
double _t_85_;
double _t_86_;
double _t_87_;
double _t_88_;
double _t_94_;
double _t_95_;
double _t_98_;
double _t_100_;
double _t_103_;
double _t_106_;
double _t_93_;
double _t_107_;
double _t_108_;
double _t_111_;
double _t_113_;
double _t_116_;
double _t_119_;
double _t_120_;
double _t_121_;
double _t_124_;
double _t_126_;
double _t_129_;
double _t_132_;
double _t_133_;
double _t_134_;
double _t_137_;
double _t_139_;
double _t_142_;
double _t_145_;
double uacc_1kp1jc0ic0;

a_mux1 = mu[k][j][i-1] * strx[i-1];
a_mux1 -= 3.0 / 4.0 * mu[k][j][i] * strx[i];
a_mux1 -= 3.0 / 4.0 * mu[k][j][i-2] * strx[i-2];
a_mux2 = mu[k][j][i-2] * strx[i-2];
a_mux2 += mu[k][j][i+1] * strx[i+1];
a_mux2 += 3.0 * mu[k][j][i] * strx[i];
a_mux2 += 3.0 * mu[k][j][i-1] * strx[i-1];
a_mux3 = mu[k][j][i-1] * strx[i-1];
a_mux3 += mu[k][j][i+2] * strx[i+2];
a_mux3 += 3.0 * mu[k][j][i+1] * strx[i+1];
a_mux3 += 3.0 * mu[k][j][i] * strx[i];
a_mux4 = mu[k][j][i+1] * strx[i+1];
a_mux4 -= 3.0 / 4.0 * mu[k][j][i] * strx[i];
a_mux4 -= 3.0 / 4.0 * mu[k][j][i+2] * strx[i+2];
a_muy1 = mu[k][j-1][i] * stry[j-1];
a_muy1 -= 3.0 / 4.0 * mu[k][j][i] * stry[j];
a_muy1 -= 3.0 / 4.0 * mu[k][j-2][i] * stry[j-2];
a_muy2 = mu[k][j-2][i] * stry[j-2];
a_muy2 += mu[k][j+1][i] * stry[j+1];
a_muy2 += 3.0 * mu[k][j][i] * stry[j];
a_muy2 += 3.0 * mu[k][j-1][i] * stry[j-1];
a_muy3 = mu[k][j-1][i] * stry[j-1];
a_muy3 += mu[k][j+2][i] * stry[j+2];
a_muy3 += 3.0 * mu[k][j+1][i] * stry[j+1];
a_muy3 += 3.0 * mu[k][j][i] * stry[j];
a_muy4 = mu[k][j+1][i] * stry[j+1];
a_muy4 -= 3.0 / 4.0 * mu[k][j][i] * stry[j];
a_muy4 -= 3.0 / 4.0 * mu[k][j+2][i] * stry[j+2];
a_muz1 = mu[k-1][j][i] * strz[k-1];
a_muz1 -= 3.0 / 4.0 * mu[k][j][i] * strz[k];
a_muz1 -= 3.0 / 4.0 * mu[k-2][j][i] * strz[k-2];
a_muz2 = mu[k-2][j][i] * strz[k-2];
a_muz2 += mu[k+1][j][i] * strz[k+1];
a_muz2 += 3.0 * mu[k][j][i] * strz[k];
a_muz2 += 3.0 * mu[k-1][j][i] * strz[k-1];
a_muz3 = mu[k-1][j][i] * strz[k-1];
a_muz3 += mu[k+2][j][i] * strz[k+2];
a_muz3 += 3.0 * mu[k+1][j][i] * strz[k+1];
a_muz3 += 3.0 * mu[k][j][i] * strz[k];
a_muz4 = mu[k+1][j][i] * strz[k+1];
a_muz4 -= 3.0 / 4.0 * mu[k][j][i] * strz[k];
a_muz4 -= 3.0 / 4.0 * mu[k+2][j][i] * strz[k+2];
_t_1_ = a_mux1 * u_1[k][j][i-2];
_t_1_ -= a_mux1 * u_1[k][j][i];
_t_1_ += a_mux2 * u_1[k][j][i-1];
_t_1_ -= a_mux2 * u_1[k][j][i];
_t_1_ += a_mux3 * u_1[k][j][i+1];
_t_1_ -= a_mux3 * u_1[k][j][i];
_t_1_ += a_mux4 * u_1[k][j][i+2];
_t_1_ -= a_mux4 * u_1[k][j][i];
a_r2 = 1.0 / 6.0 * strx[i] * _t_1_;
_t_7_ = 2.0 * a_muy1;
_t_7_ += la[k][j-1][i] * stry[j-1];
_t_7_ -= 3.0 / 4.0 * la[k][j][i] * stry[j];
_t_7_ -= 3.0 / 4.0 * la[k][j-2][i] * stry[j-2];
_t_8_ = u_1[k][j-2][i];
_t_8_ -= u_1[k][j][i];
_t_6_ = _t_7_ * _t_8_;
_t_9_ = 2.0 * a_muy2;
_t_9_ += la[k][j-2][i] * stry[j-2];
_t_9_ += la[k][j+1][i] * stry[j+1];
_t_9_ += 3.0 * la[k][j][i] * stry[j];
_t_9_ += 3.0 * la[k][j-1][i] * stry[j-1];
_t_10_ = u_1[k][j-1][i];
_t_10_ -= u_1[k][j][i];
_t_6_ += _t_9_ * _t_10_;
_t_11_ = 2.0 * a_muy3;
_t_11_ += la[k][j-1][i] * stry[j-1];
_t_11_ += la[k][j+2][i] * stry[j+2];
_t_11_ += 3.0 * la[k][j+1][i] * stry[j+1];
_t_11_ += 3.0 * la[k][j][i] * stry[j];
_t_12_ = u_1[k][j+1][i];
_t_12_ -= u_1[k][j][i];
_t_6_ += _t_11_ * _t_12_;
_t_13_ = 2.0 * a_muy4;
_t_13_ += la[k][j+1][i] * stry[j+1];
_t_13_ -= 3.0 / 4.0 * la[k][j][i] * stry[j];
_t_13_ -= 3.0 / 4.0 * la[k][j+2][i] * stry[j+2];
_t_14_ = u_1[k][j+2][i];
_t_14_ -= u_1[k][j][i];
_t_6_ += _t_13_ * _t_14_;
a_r2 += 1.0 / 6.0 * stry[j] * _t_6_;
_t_15_ = a_muz1 * u_1[k-2][j][i];
_t_15_ -= a_muz1 * u_1[k][j][i];
_t_15_ += a_muz2 * u_1[k-1][j][i];
_t_15_ -= a_muz2 * u_1[k][j][i];
_t_15_ += a_muz3 * u_1[k+1][j][i];
_t_15_ -= a_muz3 * u_1[k][j][i];
_t_15_ += a_muz4 * u_1[k+2][j][i];
_t_15_ -= a_muz4 * u_1[k][j][i];
a_r2 += 1.0 / 6.0 * strz[k] * _t_15_;
_t_21_ = 1.0 / 144.0 * strx[i] * stry[j];
_t_22_ = mu[k][j][i-2] * u_0[k][j-2][i-2];
_t_22_ -= mu[k][j][i-2] * u_0[k][j+2][i-2];
_t_25_ = -u_0[k][j-1][i-2];
_t_25_ += u_0[k][j+1][i-2];
_t_22_ += mu[k][j][i-2] * 8.0 * _t_25_;
_t_27_ = u_0[k][j-2][i-1];
_t_27_ -= u_0[k][j+2][i-1];
_t_27_ += 8.0 * -u_0[k][j-1][i-1];
_t_27_ += 8.0 * u_0[k][j+1][i-1];
_t_22_ -= 8.0 * mu[k][j][i-1] * _t_27_;
_t_30_ = u_0[k][j-2][i+1];
_t_30_ -= u_0[k][j+2][i+1];
_t_30_ += 8.0 * -u_0[k][j-1][i+1];
_t_30_ += 8.0 * u_0[k][j+1][i+1];
_t_22_ += 8.0 * mu[k][j][i+1] * _t_30_;
_t_22_ -= mu[k][j][i+2] * u_0[k][j-2][i+2];
_t_22_ += mu[k][j][i+2] * u_0[k][j+2][i+2];
_t_33_ = -u_0[k][j-1][i+2];
_t_33_ += u_0[k][j+1][i+2];
_t_22_ -= mu[k][j][i+2] * 8.0 * _t_33_;
_t_20_ = _t_21_ * _t_22_;
_t_34_ = 1.0 / 144.0 * strx[i] * stry[j];
_t_35_ = la[k][j-2][i] * u_0[k][j-2][i-2];
_t_35_ -= la[k][j-2][i] * u_0[k][j-2][i+2];
_t_38_ = -u_0[k][j-2][i-1];
_t_38_ += u_0[k][j-2][i+1];
_t_35_ += la[k][j-2][i] * 8.0 * _t_38_;
_t_40_ = u_0[k][j-1][i-2];
_t_40_ -= u_0[k][j-1][i+2];
_t_40_ += 8.0 * -u_0[k][j-1][i-1];
_t_40_ += 8.0 * u_0[k][j-1][i+1];
_t_35_ -= 8.0 * la[k][j-1][i] * _t_40_;
_t_43_ = u_0[k][j+1][i-2];
_t_43_ -= u_0[k][j+1][i+2];
_t_43_ += 8.0 * -u_0[k][j+1][i-1];
_t_43_ += 8.0 * u_0[k][j+1][i+1];
_t_35_ += 8.0 * la[k][j+1][i] * _t_43_;
_t_35_ -= la[k][j+2][i] * u_0[k][j+2][i-2];
_t_35_ += la[k][j+2][i] * u_0[k][j+2][i+2];
_t_46_ = -u_0[k][j+2][i-1];
_t_46_ += u_0[k][j+2][i+1];
_t_35_ -= la[k][j+2][i] * 8.0 * _t_46_;
_t_20_ += _t_34_ * _t_35_;
_t_47_ = 1.0 / 144.0 * stry[j] * strz[k];
_t_48_ = la[k][j-2][i] * u_2[k-2][j-2][i];
_t_48_ -= la[k][j-2][i] * u_2[k+2][j-2][i];
_t_51_ = -u_2[k-1][j-2][i];
_t_51_ += u_2[k+1][j-2][i];
_t_48_ += la[k][j-2][i] * 8.0 * _t_51_;
_t_53_ = u_2[k-2][j-1][i];
_t_53_ -= u_2[k+2][j-1][i];
_t_53_ += 8.0 * -u_2[k-1][j-1][i];
_t_53_ += 8.0 * u_2[k+1][j-1][i];
_t_48_ -= 8.0 * la[k][j-1][i] * _t_53_;
_t_56_ = u_2[k-2][j+1][i];
_t_56_ -= u_2[k+2][j+1][i];
_t_56_ += 8.0 * -u_2[k-1][j+1][i];
_t_56_ += 8.0 * u_2[k+1][j+1][i];
_t_48_ += 8.0 * la[k][j+1][i] * _t_56_;
_t_48_ -= la[k][j+2][i] * u_2[k-2][j+2][i];
_t_48_ += la[k][j+2][i] * u_2[k+2][j+2][i];
_t_59_ = -u_2[k-1][j+2][i];
_t_59_ += u_2[k+1][j+2][i];
_t_48_ -= la[k][j+2][i] * 8.0 * _t_59_;
_t_20_ += _t_47_ * _t_48_;
_t_60_ = 1.0 / 144.0 * stry[j] * strz[k];
_t_61_ = mu[k-2][j][i] * u_2[k-2][j-2][i];
_t_61_ -= mu[k-2][j][i] * u_2[k-2][j+2][i];
_t_64_ = -u_2[k-2][j-1][i];
_t_64_ += u_2[k-2][j+1][i];
_t_61_ += mu[k-2][j][i] * 8.0 * _t_64_;
_t_66_ = u_2[k-1][j-2][i];
_t_66_ -= u_2[k-1][j+2][i];
_t_66_ += 8.0 * -u_2[k-1][j-1][i];
_t_66_ += 8.0 * u_2[k-1][j+1][i];
_t_61_ -= 8.0 * mu[k-1][j][i] * _t_66_;
_t_69_ = u_2[k+1][j-2][i];
_t_69_ -= u_2[k+1][j+2][i];
_t_69_ += 8.0 * -u_2[k+1][j-1][i];
_t_69_ += 8.0 * u_2[k+1][j+1][i];
_t_61_ += 8.0 * mu[k+1][j][i] * _t_69_;
_t_61_ -= mu[k+2][j][i] * u_2[k+2][j-2][i];
_t_61_ += mu[k+2][j][i] * u_2[k+2][j+2][i];
_t_72_ = -u_2[k+2][j-1][i];
_t_72_ += u_2[k+2][j+1][i];
_t_61_ -= mu[k+2][j][i] * 8.0 * _t_72_;
_t_20_ += _t_60_ * _t_61_;
a_r2 += _t_20_;
uacc_1kc0jc0ic0 = a1 * uacc_1[k][j][i];
uacc_1kc0jc0ic0 += cof * a_r2;
uacc_1[k][j][i] = uacc_1kc0jc0ic0;
b_mux1 = mu[k+1][j][i-1] * strx[i-1];
b_mux1 -= 3.0 / 4.0 * mu[k+1][j][i] * strx[i];
b_mux1 -= 3.0 / 4.0 * mu[k+1][j][i-2] * strx[i-2];
b_mux2 = mu[k+1][j][i-2] * strx[i-2];
b_mux2 += mu[k+1][j][i+1] * strx[i+1];
b_mux2 += 3.0 * mu[k+1][j][i] * strx[i];
b_mux2 += 3.0 * mu[k+1][j][i-1] * strx[i-1];
b_mux3 = mu[k+1][j][i-1] * strx[i-1];
b_mux3 += mu[k+1][j][i+2] * strx[i+2];
b_mux3 += 3.0 * mu[k+1][j][i+1] * strx[i+1];
b_mux3 += 3.0 * mu[k+1][j][i] * strx[i];
b_mux4 = mu[k+1][j][i+1] * strx[i+1];
b_mux4 -= 3.0 / 4.0 * mu[k+1][j][i] * strx[i];
b_mux4 -= 3.0 / 4.0 * mu[k+1][j][i+2] * strx[i+2];
b_muy1 = mu[k+1][j-1][i] * stry[j-1];
b_muy1 -= 3.0 / 4.0 * mu[k+1][j][i] * stry[j];
b_muy1 -= 3.0 / 4.0 * mu[k+1][j-2][i] * stry[j-2];
b_muy2 = mu[k+1][j-2][i] * stry[j-2];
b_muy2 += mu[k+1][j+1][i] * stry[j+1];
b_muy2 += 3.0 * mu[k+1][j][i] * stry[j];
b_muy2 += 3.0 * mu[k+1][j-1][i] * stry[j-1];
b_muy3 = mu[k+1][j-1][i] * stry[j-1];
b_muy3 += mu[k+1][j+2][i] * stry[j+2];
b_muy3 += 3.0 * mu[k+1][j+1][i] * stry[j+1];
b_muy3 += 3.0 * mu[k+1][j][i] * stry[j];
b_muy4 = mu[k+1][j+1][i] * stry[j+1];
b_muy4 -= 3.0 / 4.0 * mu[k+1][j][i] * stry[j];
b_muy4 -= 3.0 / 4.0 * mu[k+1][j+2][i] * stry[j+2];
b_muz1 = mu[k][j][i] * strz[k];
b_muz1 -= 3.0 / 4.0 * mu[k+1][j][i] * strz[k+1];
b_muz1 -= 3.0 / 4.0 * mu[k-1][j][i] * strz[k-1];
b_muz2 = mu[k-1][j][i] * strz[k-1];
b_muz2 += mu[k+2][j][i] * strz[k+2];
b_muz2 += 3.0 * mu[k+1][j][i] * strz[k+1];
b_muz2 += 3.0 * mu[k][j][i] * strz[k];
b_muz3 = mu[k][j][i] * strz[k];
b_muz3 += mu[k+3][j][i] * strz[k+3];
b_muz3 += 3.0 * mu[k+2][j][i] * strz[k+2];
b_muz3 += 3.0 * mu[k+1][j][i] * strz[k+1];
b_muz4 = mu[k+2][j][i] * strz[k+2];
b_muz4 -= 3.0 / 4.0 * mu[k+1][j][i] * strz[k+1];
b_muz4 -= 3.0 / 4.0 * mu[k+3][j][i] * strz[k+3];
_t_74_ = b_mux1 * u_1[k+1][j][i-2];
_t_74_ -= b_mux1 * u_1[k+1][j][i];
_t_74_ += b_mux2 * u_1[k+1][j][i-1];
_t_74_ -= b_mux2 * u_1[k+1][j][i];
_t_74_ += b_mux3 * u_1[k+1][j][i+1];
_t_74_ -= b_mux3 * u_1[k+1][j][i];
_t_74_ += b_mux4 * u_1[k+1][j][i+2];
_t_74_ -= b_mux4 * u_1[k+1][j][i];
b_r2 = 1.0 / 6.0 * strx[i] * _t_74_;
_t_80_ = 2.0 * b_muy1;
_t_80_ += la[k+1][j-1][i] * stry[j-1];
_t_80_ -= 3.0 / 4.0 * la[k+1][j][i] * stry[j];
_t_80_ -= 3.0 / 4.0 * la[k+1][j-2][i] * stry[j-2];
_t_81_ = u_1[k+1][j-2][i];
_t_81_ -= u_1[k+1][j][i];
_t_79_ = _t_80_ * _t_81_;
_t_82_ = 2.0 * b_muy2;
_t_82_ += la[k+1][j-2][i] * stry[j-2];
_t_82_ += la[k+1][j+1][i] * stry[j+1];
_t_82_ += 3.0 * la[k+1][j][i] * stry[j];
_t_82_ += 3.0 * la[k+1][j-1][i] * stry[j-1];
_t_83_ = u_1[k+1][j-1][i];
_t_83_ -= u_1[k+1][j][i];
_t_79_ += _t_82_ * _t_83_;
_t_84_ = 2.0 * b_muy3;
_t_84_ += la[k+1][j-1][i] * stry[j-1];
_t_84_ += la[k+1][j+2][i] * stry[j+2];
_t_84_ += 3.0 * la[k+1][j+1][i] * stry[j+1];
_t_84_ += 3.0 * la[k+1][j][i] * stry[j];
_t_85_ = u_1[k+1][j+1][i];
_t_85_ -= u_1[k+1][j][i];
_t_79_ += _t_84_ * _t_85_;
_t_86_ = 2.0 * b_muy4;
_t_86_ += la[k+1][j+1][i] * stry[j+1];
_t_86_ -= 3.0 / 4.0 * la[k+1][j][i] * stry[j];
_t_86_ -= 3.0 / 4.0 * la[k+1][j+2][i] * stry[j+2];
_t_87_ = u_1[k+1][j+2][i];
_t_87_ -= u_1[k+1][j][i];
_t_79_ += _t_86_ * _t_87_;
b_r2 += 1.0 / 6.0 * stry[j] * _t_79_;
_t_88_ = b_muz1 * u_1[k-1][j][i];
_t_88_ -= b_muz1 * u_1[k+1][j][i];
_t_88_ += b_muz2 * u_1[k][j][i];
_t_88_ -= b_muz2 * u_1[k+1][j][i];
_t_88_ += b_muz3 * u_1[k+2][j][i];
_t_88_ -= b_muz3 * u_1[k+1][j][i];
_t_88_ += b_muz4 * u_1[k+3][j][i];
_t_88_ -= b_muz4 * u_1[k+1][j][i];
b_r2 += 1.0 / 6.0 * strz[k+1] * _t_88_;
_t_94_ = 1.0 / 144.0 * strx[i] * stry[j];
_t_95_ = mu[k+1][j][i-2] * u_0[k+1][j-2][i-2];
_t_95_ -= mu[k+1][j][i-2] * u_0[k+1][j+2][i-2];
_t_98_ = -u_0[k+1][j-1][i-2];
_t_98_ += u_0[k+1][j+1][i-2];
_t_95_ += mu[k+1][j][i-2] * 8.0 * _t_98_;
_t_100_ = u_0[k+1][j-2][i-1];
_t_100_ -= u_0[k+1][j+2][i-1];
_t_100_ += 8.0 * -u_0[k+1][j-1][i-1];
_t_100_ += 8.0 * u_0[k+1][j+1][i-1];
_t_95_ -= 8.0 * mu[k+1][j][i-1] * _t_100_;
_t_103_ = u_0[k+1][j-2][i+1];
_t_103_ -= u_0[k+1][j+2][i+1];
_t_103_ += 8.0 * -u_0[k+1][j-1][i+1];
_t_103_ += 8.0 * u_0[k+1][j+1][i+1];
_t_95_ += 8.0 * mu[k+1][j][i+1] * _t_103_;
_t_95_ -= mu[k+1][j][i+2] * u_0[k+1][j-2][i+2];
_t_95_ += mu[k+1][j][i+2] * u_0[k+1][j+2][i+2];
_t_106_ = -u_0[k+1][j-1][i+2];
_t_106_ += u_0[k+1][j+1][i+2];
_t_95_ -= mu[k+1][j][i+2] * 8.0 * _t_106_;
_t_93_ = _t_94_ * _t_95_;
_t_107_ = 1.0 / 144.0 * strx[i] * stry[j];
_t_108_ = la[k+1][j-2][i] * u_0[k+1][j-2][i-2];
_t_108_ -= la[k+1][j-2][i] * u_0[k+1][j-2][i+2];
_t_111_ = -u_0[k+1][j-2][i-1];
_t_111_ += u_0[k+1][j-2][i+1];
_t_108_ += la[k+1][j-2][i] * 8.0 * _t_111_;
_t_113_ = u_0[k+1][j-1][i-2];
_t_113_ -= u_0[k+1][j-1][i+2];
_t_113_ += 8.0 * -u_0[k+1][j-1][i-1];
_t_113_ += 8.0 * u_0[k+1][j-1][i+1];
_t_108_ -= 8.0 * la[k+1][j-1][i] * _t_113_;
_t_116_ = u_0[k+1][j+1][i-2];
_t_116_ -= u_0[k+1][j+1][i+2];
_t_116_ += 8.0 * -u_0[k+1][j+1][i-1];
_t_116_ += 8.0 * u_0[k+1][j+1][i+1];
_t_108_ += 8.0 * la[k+1][j+1][i] * _t_116_;
_t_108_ -= la[k+1][j+2][i] * u_0[k+1][j+2][i-2];
_t_108_ += la[k+1][j+2][i] * u_0[k+1][j+2][i+2];
_t_119_ = -u_0[k+1][j+2][i-1];
_t_119_ += u_0[k+1][j+2][i+1];
_t_108_ -= la[k+1][j+2][i] * 8.0 * _t_119_;
_t_93_ += _t_107_ * _t_108_;
_t_120_ = 1.0 / 144.0 * stry[j] * strz[k+1];
_t_121_ = la[k+1][j-2][i] * u_2[k-1][j-2][i];
_t_121_ -= la[k+1][j-2][i] * u_2[k+3][j-2][i];
_t_124_ = -u_2[k][j-2][i];
_t_124_ += u_2[k+2][j-2][i];
_t_121_ += la[k+1][j-2][i] * 8.0 * _t_124_;
_t_126_ = u_2[k-1][j-1][i];
_t_126_ -= u_2[k+3][j-1][i];
_t_126_ += 8.0 * -u_2[k][j-1][i];
_t_126_ += 8.0 * u_2[k+2][j-1][i];
_t_121_ -= 8.0 * la[k+1][j-1][i] * _t_126_;
_t_129_ = u_2[k-1][j+1][i];
_t_129_ -= u_2[k+3][j+1][i];
_t_129_ += 8.0 * -u_2[k][j+1][i];
_t_129_ += 8.0 * u_2[k+2][j+1][i];
_t_121_ += 8.0 * la[k+1][j+1][i] * _t_129_;
_t_121_ -= la[k+1][j+2][i] * u_2[k-1][j+2][i];
_t_121_ += la[k+1][j+2][i] * u_2[k+3][j+2][i];
_t_132_ = -u_2[k][j+2][i];
_t_132_ += u_2[k+2][j+2][i];
_t_121_ -= la[k+1][j+2][i] * 8.0 * _t_132_;
_t_93_ += _t_120_ * _t_121_;
_t_133_ = 1.0 / 144.0 * stry[j] * strz[k+1];
_t_134_ = mu[k-1][j][i] * u_2[k-1][j-2][i];
_t_134_ -= mu[k-1][j][i] * u_2[k-1][j+2][i];
_t_137_ = -u_2[k-1][j-1][i];
_t_137_ += u_2[k-1][j+1][i];
_t_134_ += mu[k-1][j][i] * 8.0 * _t_137_;
_t_139_ = u_2[k][j-2][i];
_t_139_ -= u_2[k][j+2][i];
_t_139_ += 8.0 * -u_2[k][j-1][i];
_t_139_ += 8.0 * u_2[k][j+1][i];
_t_134_ -= 8.0 * mu[k][j][i] * _t_139_;
_t_142_ = u_2[k+2][j-2][i];
_t_142_ -= u_2[k+2][j+2][i];
_t_142_ += 8.0 * -u_2[k+2][j-1][i];
_t_142_ += 8.0 * u_2[k+2][j+1][i];
_t_134_ += 8.0 * mu[k+2][j][i] * _t_142_;
_t_134_ -= mu[k+3][j][i] * u_2[k+3][j-2][i];
_t_134_ += mu[k+3][j][i] * u_2[k+3][j+2][i];
_t_145_ = -u_2[k+3][j-1][i];
_t_145_ += u_2[k+3][j+1][i];
_t_134_ -= mu[k+3][j][i] * 8.0 * _t_145_;
_t_93_ += _t_133_ * _t_134_;
b_r2 += _t_93_;
uacc_1kp1jc0ic0 = a1 * uacc_1[k+1][j][i];
uacc_1kp1jc0ic0 += cof * b_r2;
uacc_1[k+1][j][i] = uacc_1kp1jc0ic0;
		}
	} 
}

__global__ void __launch_bounds__ (128,2) sw4_3 (double * uacc_in_0, double * uacc_in_1, double * uacc_in_2, double * __restrict__ u_in_0, double * __restrict__ u_in_1, double * __restrict__ u_in_2, double * __restrict__ mu_in, double * __restrict__ la_in, double * strx, double * stry, double * strz, int N) {
	//Determing the block's indices
	int blockdim_i= (int)(blockDim.x);
	int i0 = (int)(blockIdx.x)*(blockdim_i);
	int i = max (i0, 0) + (int)(threadIdx.x);
	int blockdim_j= (int)(blockDim.y);
	int j0 = (int)(blockIdx.y)*(blockdim_j);
	int j = max (j0, 0) + (int)(threadIdx.y);

	// Assumptions 
	int a1 = 1;
	double h = 3.7;
	double cof = 1e0 / ( h *  h);

	double (*uacc_0)[304][304] = (double (*)[304][304])uacc_in_0;
	double (*uacc_1)[304][304] = (double (*)[304][304])uacc_in_1;
	double (*uacc_2)[304][304] = (double (*)[304][304])uacc_in_2;
	double (*u_0)[304][304] = (double (*)[304][304])u_in_0;
	double (*u_1)[304][304] = (double (*)[304][304])u_in_1;
	double (*u_2)[304][304] = (double (*)[304][304])u_in_2;
	double (*mu)[304][304] = (double (*)[304][304])mu_in;
	double (*la)[304][304] = (double (*)[304][304])la_in;

	double mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
	double r1, r2, r3;
	if (i>=2 & j>=2 & i<=N-3 & j<=N-3) {
#pragma unroll 10 
		for (int k=2; k<=N-3; k++) {
			mux1 = mu[k][j][i-1] * strx[i-1] - 3e0 / 4 * mu[k][j][i] * strx[i] - 3e0 / 4 * mu[k][j][i-2] * strx[i-2];
			mux2 = mu[k][j][i-2] * strx[i-2] + mu[k][j][i+1] * strx[i+1] + 3.0 * mu[k][j][i] * strx[i] + 3.0 * mu[k][j][i-1] * strx[i-1];
			mux3 = mu[k][j][i-1] * strx[i-1] + mu[k][j][i+2] * strx[i+2] + 3.0 * mu[k][j][i+1] * strx[i+1] + 3.0 * mu[k][j][i] * strx[i];
			mux4 = mu[k][j][i+1] * strx[i+1] - 3e0 / 4 * mu[k][j][i] * strx[i] - 3e0 / 4 *  mu[k][j][i+2] * strx[i+2];

			muy1 = mu[k][j-1][i] * stry[j-1] - 3e0 / 4 * mu[k][j][i] * stry[j] -3e0 / 4 * mu[k][j-2][i] * stry[j-2];
			muy2 = mu[k][j-2][i] * stry[j-2] + mu[k][j+1][i] * stry[j+1] + 3.0 * mu[k][j][i] * stry[j] +     3.0 * mu[k][j-1][i] * stry[j-1];
			muy3 = mu[k][j-1][i] * stry[j-1] + mu[k][j+2][i] * stry[j+2] + 3.0 * mu[k][j+1][i] * stry[j+1] + 3.0 * mu[k][j][i] * stry[j];

			muy4 = mu[k][j+1][i] * stry[j+1] - 3e0 / 4 * mu[k][j][i] * stry[j] - 3e0 / 4 * mu[k][j+2][i] * stry[j+2];
			muz1 = mu[k-1][j][i] * strz[k-1] - 3e0 / 4 * mu[k][j][i] * strz[k] - 3e0 / 4 * mu[k-2][j][i] * strz[k-2];
			muz2 = mu[k-2][j][i] * strz[k-2] + mu[k+1][j][i] * strz[k+1] + 3.0 * mu[k][j][i] * strz[k] + 3.0 * mu[k-1][j][i] * strz[k-1];
			muz3 = mu[k-1][j][i] * strz[k-1] + mu[k+2][j][i] * strz[k+2] + 3.0 * mu[k+1][j][i] * strz[k+1] + 3.0 * mu[k][j][i] * strz[k];
			muz4 = mu[k+1][j][i] * strz[k+1] - 3e0 / 4 * mu[k][j][i] * strz[k] - 3e0 /4  * mu[k+2][j][i] * strz[k+2];

			r3 = 1e0 / 6 * (strx[i] * (mux1 * (u_2[k][j][i-2] - u_2[k][j][i]) + mux2 * (u_2[k][j][i-1] - u_2[k][j][i]) + mux3 * (u_2[k][j][i+1] - u_2[k][j][i]) + mux4 * (u_2[k][j][i+2] - u_2[k][j][i])) + 
					stry[j] * (muy1 * (u_2[k][j-2][i] - u_2[k][j][i]) + muy2 * (u_2[k][j-1][i] - u_2[k][j][i]) + muy3 * (u_2[k][j+1][i] - u_2[k][j][i]) + muy4 * (u_2[k][j+2][i] - u_2[k][j][i])) + 
					strz[k] * ((2 * muz1 + la[k-1][j][i] * strz[k-1] - 3e0 / 4 * la[k][j][i] * strz[k] - 3e0 / 4 * la[k-2][j][i] * strz[k-2]) * (u_2[k-2][j][i] - u_2[k][j][i]) + 
						(2 * muz2 + la[k-2][j][i] * strz[k-2] + la[k+1][j][i] * strz[k+1] + 3 * la[k][j][i] * strz[k] + 3 * la[k-1][j][i] * strz[k-1]) * (u_2[k-1][j][i] - u_2[k][j][i]) + 
						(2 * muz3 + la[k-1][j][i] * strz[k-1] + la[k+2][j][i] * strz[k+2] + 3 * la[k+1][j][i] * strz[k+1] + 3 * la[k][j][i] * strz[k]) * (u_2[k+1][j][i] - u_2[k][j][i]) + 
						(2 * muz4 + la[k+1][j][i] * strz[k+1] - 3e0 / 4 * la[k][j][i] * strz[k] - 3e0 / 4 * la[k+2][j][i] * strz[k+2]) * (u_2[k+2][j][i] - u_2[k][j][i])));

			r3 += strx[i] * strz[k] * (1e0 / 144) * (mu[k][j][i-2] * (u_0[k-2][j][i-2] - u_0[k+2][j][i-2] + 8 * (-u_0[k-1][j][i-2] + u_0[k+1][j][i-2])) - 8 * (mu[k][j][i-1] * (u_0[k-2][j][i-1] - u_0[k+2][j][i-1] + 8 * (-u_0[k-1][j][i-1] + u_0[k+1][j][i-1]))) + 8 * (mu[k][j][i+1] * (u_0[k-2][j][i+1] - u_0[k+2][j][i+1] + 8 * (-u_0[k-1][j][i+1] + u_0[k+1][j][i+1]))) - (mu[k][j][i+2] * (u_0[k-2][j][i+2] - u_0[k+2][j][i+2] + 8 * (-u_0[k-1][j][i+2] + u_0[k+1][j][i+2]))));
			r3 += stry[j] * strz[k] * (1e0 / 144) * (mu[k][j-2][i] * (u_1[k-2][j-2][i] - u_1[k+2][j-2][i] + 8 * (-u_1[k-1][j-2][i] + u_1[k+1][j-2][i])) - 8 * (mu[k][j-1][i] * (u_1[k-2][j-1][i] - u_1[k+2][j-1][i] + 8 * (-u_1[k-1][j-1][i] + u_1[k+1][j-1][i]))) + 8 * (mu[k][j+1][i] * (u_1[k-2][j+1][i] - u_1[k+2][j+1][i] + 8 * (-u_1[k-1][j+1][i] + u_1[k+1][j+1][i]))) - (mu[k][j+2][i] * (u_1[k-2][j+2][i] - u_1[k+2][j+2][i] + 8 * (-u_1[k-1][j+2][i] + u_1[k+1][j+2][i]))));
			r3 += strx[i] * strz[k] * (1e0 / 144) * (la[k-2][j][i] * (u_0[k-2][j][i-2] - u_0[k-2][j][i+2] + 8 * (-u_0[k-2][j][i-1] + u_0[k-2][j][i+1])) - 8 * (la[k-1][j][i] * (u_0[k-1][j][i-2] - u_0[k-1][j][i+2] + 8 * (-u_0[k-1][j][i-1] + u_0[k-1][j][i+1]))) + 8 * (la[k+1][j][i] * (u_0[k+1][j][i-2] - u_0[k+1][j][i+2] + 8 * (-u_0[k+1][j][i-1] + u_0[k+1][j][i+1]))) - (la[k+2][j][i] * (u_0[k+2][j][i-2] - u_0[k+2][j][i+2] + 8 * (-u_0[k+2][j][i-1] + u_0[k+2][j][i+1]))));
			r3 += stry[j] * strz[k] * (1e0 / 144) * (la[k-2][j][i] * (u_1[k-2][j-2][i] - u_1[k-2][j+2][i] + 8 * (-u_1[k-2][j-1][i] + u_1[k-2][j+1][i])) - 8 * (la[k-1][j][i] * (u_1[k-1][j-2][i] - u_1[k-1][j+2][i] + 8 * (-u_1[k-1][j-1][i] + u_1[k-1][j+1][i]))) + 8 * (la[k+1][j][i] * (u_1[k+1][j-2][i] - u_1[k+1][j+2][i] + 8 * (-u_1[k+1][j-1][i] + u_1[k+1][j+1][i]))) - (la[k+2][j][i] * (u_1[k+2][j-2][i] - u_1[k+2][j+2][i] + 8 * (-u_1[k+2][j-1][i] + u_1[k+2][j+1][i]))));

			uacc_2[k][j][i] = a1 * uacc_2[k][j][i] + cof * r3;
		}
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

	dim3 blockconfig (16, 8);
	dim3 gridconfig (ceil(N, blockconfig.x), ceil(N, blockconfig.y), 1);

	sw4_1 <<<gridconfig, blockconfig>>> (uacc_0, uacc_1, uacc_2, u_0, u_1, u_2, mu, la, strx, stry, strz, N);
	sw4_2 <<<gridconfig, blockconfig>>> (uacc_0, uacc_1, uacc_2, u_0, u_1, u_2, mu, la, strx, stry, strz, N);
	sw4_3 <<<gridconfig, blockconfig>>> (uacc_0, uacc_1, uacc_2, u_0, u_1, u_2, mu, la, strx, stry, strz, N);

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
