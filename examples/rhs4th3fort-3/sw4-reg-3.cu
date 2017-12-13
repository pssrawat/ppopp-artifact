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
#pragma begin stencil2 unroll k=1,j=1,i=1
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

			a_r2 = 1e0 / 6 * (strx[i] * (a_mux1 * (u_1[k][j][i-2] - u_1[k][j][i]) + a_mux2 * (u_1[k][j][i-1] - u_1[k][j][i]) + a_mux3 * (u_1[k][j][i+1] - u_1[k][j][i]) + a_mux4 * (u_1[k][j][i+2] - u_1[k][j][i])) + 
					stry[j] * ((2 * a_muy1 + la[k][j-1][i] * stry[j-1] - 3e0 / 4 * la[k][j][i] * stry[j] - 3e0 / 4 * la[k][j-2][i] * stry[j-2]) * (u_1[k][j-2][i] - u_1[k][j][i]) + 
						(2 * a_muy2 + la[k][j-2][i] * stry[j-2] + la[k][j+1][i] * stry[j+1] + 3 * la[k][j][i] * stry[j] + 3 * la[k][j-1][i] * stry[j-1]) * (u_1[k][j-1][i] - u_1[k][j][i]) + 
						(2 * a_muy3 + la[k][j-1][i] * stry[j-1] + la[k][j+2][i] * stry[j+2] + 3 * la[k][j+1][i] * stry[j+1] + 3 * la[k][j][i] * stry[j]) * (u_1[k][j+1][i] - u_1[k][j][i]) + 
						(2 * a_muy4 + la[k][j+1][i] * stry[j+1] - 3e0 / 4 * la[k][j][i] * stry[j] - 3e0 / 4 * la[k][j+2][i] * stry[j+2]) * (u_1[k][j+2][i] - u_1[k][j][i])) + 
					strz[k] * (a_muz1 * (u_1[k-2][j][i] - u_1[k][j][i]) + a_muz2 * (u_1[k-1][j][i] - u_1[k][j][i]) + a_muz3 * (u_1[k+1][j][i] - u_1[k][j][i]) + a_muz4 * (u_1[k+2][j][i] - u_1[k][j][i])));

			a_r2 += strx[i] * stry[j] * (1e0 / 144) * (mu[k][j][i-2] * (u_0[k][j-2][i-2] - u_0[k][j+2][i-2] + 8 * (-u_0[k][j-1][i-2] + u_0[k][j+1][i-2])) - 8 * (mu[k][j][i-1] * (u_0[k][j-2][i-1] - u_0[k][j+2][i-1] + 8 * (-u_0[k][j-1][i-1] + u_0[k][j+1][i-1]))) + 8 * (mu[k][j][i+1] * (u_0[k][j-2][i+1] - u_0[k][j+2][i+1] + 8 * (-u_0[k][j-1][i+1] + u_0[k][j+1][i+1]))) - (mu[k][j][i+2] * (u_0[k][j-2][i+2] - u_0[k][j+2][i+2] + 8 * (-u_0[k][j-1][i+2] + u_0[k][j+1][i+2])))) + strx[i] * stry[j] * (1e0 / 144) * (la[k][j-2][i] * (u_0[k][j-2][i-2] - u_0[k][j-2][i+2] + 8 * (-u_0[k][j-2][i-1] + u_0[k][j-2][i+1])) - 8 * (la[k][j-1][i] * (u_0[k][j-1][i-2] - u_0[k][j-1][i+2] + 8 * (-u_0[k][j-1][i-1] + u_0[k][j-1][i+1]))) + 8 * (la[k][j+1][i] * (u_0[k][j+1][i-2] - u_0[k][j+1][i+2] + 8 * (-u_0[k][j+1][i-1] + u_0[k][j+1][i+1]))) - (la[k][j+2][i] * (u_0[k][j+2][i-2] - u_0[k][j+2][i+2] + 8 * (-u_0[k][j+2][i-1] + u_0[k][j+2][i+1])))) + stry[j] * strz[k] * (1e0 / 144) * (la[k][j-2][i] * (u_2[k-2][j-2][i] - u_2[k+2][j-2][i] + 8 * (-u_2[k-1][j-2][i] + u_2[k+1][j-2][i])) - 8 * (la[k][j-1][i] * (u_2[k-2][j-1][i] - u_2[k+2][j-1][i] + 8 * (-u_2[k-1][j-1][i] + u_2[k+1][j-1][i]))) + 8 * (la[k][j+1][i] * (u_2[k-2][j+1][i] - u_2[k+2][j+1][i] + 8 * (-u_2[k-1][j+1][i] + u_2[k+1][j+1][i]))) - (la[k][j+2][i] * (u_2[k-2][j+2][i] - u_2[k+2][j+2][i] + 8 * (-u_2[k-1][j+2][i] + u_2[k+1][j+2][i])))) + stry[j] * strz[k] * (1e0 / 144) * (mu[k-2][j][i] * (u_2[k-2][j-2][i] - u_2[k-2][j+2][i] + 8 * (-u_2[k-2][j-1][i] + u_2[k-2][j+1][i])) - 8 * (mu[k-1][j][i] * (u_2[k-1][j-2][i] - u_2[k-1][j+2][i] + 8 * (-u_2[k-1][j-1][i] + u_2[k-1][j+1][i]))) + 8 * (mu[k+1][j][i] * (u_2[k+1][j-2][i] - u_2[k+1][j+2][i] + 8 * (-u_2[k+1][j-1][i] + u_2[k+1][j+1][i]))) - (mu[k+2][j][i] * (u_2[k+2][j-2][i] - u_2[k+2][j+2][i] + 8 * (-u_2[k+2][j-1][i] + u_2[k+2][j+1][i]))));

			uacc_1[k][j][i] = a1 * uacc_1[k][j][i] + cof * a_r2;


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

			b_r2 = 1e0 / 6 * (strx[i] * (b_mux1 * (u_1[k+1][j][i-2] - u_1[k+1][j][i]) + b_mux2 * (u_1[k+1][j][i-1] - u_1[k+1][j][i]) + b_mux3 * (u_1[k+1][j][i+1] - u_1[k+1][j][i]) + b_mux4 * (u_1[k+1][j][i+2] - u_1[k+1][j][i])) + 
					stry[j] * ((2 * b_muy1 + la[k+1][j-1][i] * stry[j-1] - 3e0 / 4 * la[k+1][j][i] * stry[j] - 3e0 / 4 * la[k+1][j-2][i] * stry[j-2]) * (u_1[k+1][j-2][i] - u_1[k+1][j][i]) + 
						(2 * b_muy2 + la[k+1][j-2][i] * stry[j-2] + la[k+1][j+1][i] * stry[j+1] + 3 * la[k+1][j][i] * stry[j] + 3 * la[k+1][j-1][i] * stry[j-1]) * (u_1[k+1][j-1][i] - u_1[k+1][j][i]) + 
						(2 * b_muy3 + la[k+1][j-1][i] * stry[j-1] + la[k+1][j+2][i] * stry[j+2] + 3 * la[k+1][j+1][i] * stry[j+1] + 3 * la[k+1][j][i] * stry[j]) * (u_1[k+1][j+1][i] - u_1[k+1][j][i]) + 
						(2 * b_muy4 + la[k+1][j+1][i] * stry[j+1] - 3e0 / 4 * la[k+1][j][i] * stry[j] - 3e0 / 4 * la[k+1][j+2][i] * stry[j+2]) * (u_1[k+1][j+2][i] - u_1[k+1][j][i])) + 
					strz[k+1] * (b_muz1 * (u_1[k+1-2][j][i] - u_1[k+1][j][i]) + b_muz2 * (u_1[k+1-1][j][i] - u_1[k+1][j][i]) + b_muz3 * (u_1[k+1+1][j][i] - u_1[k+1][j][i]) + b_muz4 * (u_1[k+1+2][j][i] - u_1[k+1][j][i])));


			b_r2 += strx[i] * stry[j] * (1e0 / 144) * (mu[k+1][j][i-2] * (u_0[k+1][j-2][i-2] - u_0[k+1][j+2][i-2] + 8 * (-u_0[k+1][j-1][i-2] + u_0[k+1][j+1][i-2])) - 8 * (mu[k+1][j][i-1] * (u_0[k+1][j-2][i-1] - u_0[k+1][j+2][i-1] + 8 * (-u_0[k+1][j-1][i-1] + u_0[k+1][j+1][i-1]))) + 8 * (mu[k+1][j][i+1] * (u_0[k+1][j-2][i+1] - u_0[k+1][j+2][i+1] + 8 * (-u_0[k+1][j-1][i+1] + u_0[k+1][j+1][i+1]))) - (mu[k+1][j][i+2] * (u_0[k+1][j-2][i+2] - u_0[k+1][j+2][i+2] + 8 * (-u_0[k+1][j-1][i+2] + u_0[k+1][j+1][i+2])))) + strx[i] * stry[j] * (1e0 / 144) * (la[k+1][j-2][i] * (u_0[k+1][j-2][i-2] - u_0[k+1][j-2][i+2] + 8 * (-u_0[k+1][j-2][i-1] + u_0[k+1][j-2][i+1])) - 8 * (la[k+1][j-1][i] * (u_0[k+1][j-1][i-2] - u_0[k+1][j-1][i+2] + 8 * (-u_0[k+1][j-1][i-1] + u_0[k+1][j-1][i+1]))) + 8 * (la[k+1][j+1][i] * (u_0[k+1][j+1][i-2] - u_0[k+1][j+1][i+2] + 8 * (-u_0[k+1][j+1][i-1] + u_0[k+1][j+1][i+1]))) - (la[k+1][j+2][i] * (u_0[k+1][j+2][i-2] - u_0[k+1][j+2][i+2] + 8 * (-u_0[k+1][j+2][i-1] + u_0[k+1][j+2][i+1])))) + stry[j] * strz[k+1] * (1e0 / 144) * (la[k+1][j-2][i] * (u_2[k+1-2][j-2][i] - u_2[k+1+2][j-2][i] + 8 * (-u_2[k+1-1][j-2][i] + u_2[k+1+1][j-2][i])) - 8 * (la[k+1][j-1][i] * (u_2[k+1-2][j-1][i] - u_2[k+1+2][j-1][i] + 8 * (-u_2[k+1-1][j-1][i] + u_2[k+1+1][j-1][i]))) + 8 * (la[k+1][j+1][i] * (u_2[k+1-2][j+1][i] - u_2[k+1+2][j+1][i] + 8 * (-u_2[k+1-1][j+1][i] + u_2[k+1+1][j+1][i]))) - (la[k+1][j+2][i] * (u_2[k+1-2][j+2][i] - u_2[k+1+2][j+2][i] + 8 * (-u_2[k+1-1][j+2][i] + u_2[k+1+1][j+2][i])))) + stry[j] * strz[k+1] * (1e0 / 144) * (mu[k+1-2][j][i] * (u_2[k+1-2][j-2][i] - u_2[k+1-2][j+2][i] + 8 * (-u_2[k+1-2][j-1][i] + u_2[k+1-2][j+1][i])) - 8 * (mu[k+1-1][j][i] * (u_2[k+1-1][j-2][i] - u_2[k+1-1][j+2][i] + 8 * (-u_2[k+1-1][j-1][i] + u_2[k+1-1][j+1][i]))) + 8 * (mu[k+1+1][j][i] * (u_2[k+1+1][j-2][i] - u_2[k+1+1][j+2][i] + 8 * (-u_2[k+1+1][j-1][i] + u_2[k+1+1][j+1][i]))) - (mu[k+1+2][j][i] * (u_2[k+1+2][j-2][i] - u_2[k+1+2][j+2][i] + 8 * (-u_2[k+1+2][j-1][i] + u_2[k+1+2][j+1][i]))));

			uacc_1[k+1][j][i] = a1 * uacc_1[k+1][j][i] + cof * b_r2;
#pragma end stencil2
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
