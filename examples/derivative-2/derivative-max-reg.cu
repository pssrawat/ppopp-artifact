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

__global__ void curvi (double * __restrict__ in_r1, double *__restrict__ in_u1, double * __restrict__ in_u2, double *__restrict__ in_u3, double * __restrict__ in_mu, double * __restrict__ in_la, double * __restrict__ in_met1, double * __restrict__ in_met2, double * __restrict__ in_met3, double * __restrict__ in_met4, double * strx, double * stry, double c1, double c2, int N) {
	//Determing the block's indices
	int blockdim_k= (int)(blockDim.x);
	int k0 = (int)(blockIdx.x)*(blockdim_k);
	int k = max (k0, 0) + (int)(threadIdx.x);
	int blockdim_j= (int)(blockDim.y);
	int j0 = (int)(blockIdx.y)*(blockdim_j);
	int j = max (j0, 0) + (int)(threadIdx.y);

	double (*u1)[304][304] = (double (*)[304][304])in_u1;
	double (*u2)[304][304] = (double (*)[304][304])in_u2;
	double (*u3)[304][304] = (double (*)[304][304])in_u3;
	double (*mu)[304][304] = (double (*)[304][304])in_mu;
	double (*la)[304][304] = (double (*)[304][304])in_la;
	double (*r1)[304][304] = (double (*)[304][304])in_r1;
	double (*met1)[304][304] = (double (*)[304][304])in_met1;
	double (*met2)[304][304] = (double (*)[304][304])in_met2;
	double (*met3)[304][304] = (double (*)[304][304])in_met3;
	double (*met4)[304][304] = (double (*)[304][304])in_met4;

	if (j>=2 & k>=2 & j<=N-3 & k<=N-3) {
		for (int i=2; i<=N-3; i++) {
#pragma begin stencil1 unroll i=1,j=1,k=1

			r1[i][j][k] += c2*(
					(2*mu[i][j][k+2]+la[i][j][k+2])*met2[i][j][k+2]*met1[i][j][k+2]*(
						c2*(u1[i+2][j][k+2]-u1[i-2][j][k+2]) +
						c1*(u1[i+1][j][k+2]-u1[i-1][j][k+2])   )*strx[i]*stry[j]
					+ mu[i][j][k+2]*met3[i][j][k+2]*met1[i][j][k+2]*(
						c2*(u2[i+2][j][k+2]-u2[i-2][j][k+2]) +
						c1*(u2[i+1][j][k+2]-u2[i-1][j][k+2])  )
					+ mu[i][j][k+2]*met4[i][j][k+2]*met1[i][j][k+2]*(
						c2*(u3[i+2][j][k+2]-u3[i-2][j][k+2]) +
						c1*(u3[i+1][j][k+2]-u3[i-1][j][k+2])  )*stry[j]
					+ ((2*mu[i][j][k-2]+la[i][j][k-2])*met2[i][j][k-2]*met1[i][j][k-2]*(
							c2*(u1[i+2][j][k-2]-u1[i-2][j][k-2]) +
							c1*(u1[i+1][j][k-2]-u1[i-1][j][k-2])  )*strx[i]*stry[j]
						+ mu[i][j][k-2]*met3[i][j][k-2]*met1[i][j][k-2]*(
							c2*(u2[i+2][j][k-2]-u2[i-2][j][k-2]) +
							c1*(u2[i+1][j][k-2]-u2[i-1][j][k-2])   )
						+ mu[i][j][k-2]*met4[i][j][k-2]*met1[i][j][k-2]*(
							c2*(u3[i+2][j][k-2]-u3[i-2][j][k-2]) +
							c1*(u3[i+1][j][k-2]-u3[i-1][j][k-2])   )*stry[j] )
					) + c1*(
						(2*mu[i][j][k+1]+la[i][j][k+1])*met2[i][j][k+1]*met1[i][j][k+1]*(
							c2*(u1[i+2][j][k+1]-u1[i-2][j][k+1]) +
							c1*(u1[i+1][j][k+1]-u1[i-1][j][k+1]) )*strx[i+2]*stry[j]
						+ mu[i][j][k+1]*met3[i][j][k+1]*met1[i][j][k+1]*(
							c2*(u2[i+2][j][k+1]-u2[i-2][j][k+1]) +
							c1*(u2[i+1][j][k+1]-u2[i-1][j][k+1]) )
						+ mu[i][j][k+1]*met4[i][j][k+1]*met1[i][j][k+1]*(
							c2*(u3[i+2][j][k+1]-u3[i-2][j][k+1]) +
							c1*(u3[i+1][j][k+1]-u3[i-1][j][k+1])  )*stry[j]
						+ ((2*mu[i][j][k-1]+la[i][j][k-1])*met2[i][j][k-1]*met1[i][j][k-1]*(
								c2*(u1[i+2][j][k-1]-u1[i-2][j][k-1]) +
								c1*(u1[i+1][j][k-1]-u1[i-1][j][k-1]) )*strx[i-2]*stry[j]
							+ mu[i][j][k-1]*met3[i][j][k-1]*met1[i][j][k-1]*(
								c2*(u2[i+2][j][k-1]-u2[i-2][j][k-1]) +
								c1*(u2[i+1][j][k-1]-u2[i-1][j][k-1]) )
							+ mu[i][j][k-1]*met4[i][j][k-1]*met1[i][j][k-1]*(
								c2*(u3[i+2][j][k-1]-u3[i-2][j][k-1]) +
								c1*(u3[i+1][j][k-1]-u3[i-1][j][k-1])   )*stry[j]  ) );

			r1[i][j][k] += ( c2*(
						(2*mu[i+2][j][k]+la[i+2][j][k])*met2[i+2][j][k]*met1[i+2][j][k]*(
							c2*(u1[i+2][j][k+2]-u1[i+2][j][k-2]) +
							c1*(u1[i+2][j][k+1]-u1[i+2][j][k-1])   )*strx[i]
						+ la[i+2][j][k]*met3[i+2][j][k]*met1[i+2][j][k]*(
							c2*(u2[i+2][j][k+2]-u2[i+2][j][k-2]) +
							c1*(u2[i+2][j][k+1]-u2[i+2][j][k-1])  )*stry[j]
						+ la[i+2][j][k]*met4[i+2][j][k]*met1[i+2][j][k]*(
							c2*(u3[i+2][j][k+2]-u3[i+2][j][k-2]) +
							c1*(u3[i+2][j][k+1]-u3[i+2][j][k-1])  )
						+ ((2*mu[i-2][j][k]+la[i-2][j][k])*met2[i-2][j][k]*met1[i-2][j][k]*(
								c2*(u1[i-2][j][k+2]-u1[i-2][j][k-2]) +
								c1*(u1[i-2][j][k+1]-u1[i-2][j][k-1])  )*strx[i]
							+ la[i-2][j][k]*met3[i-2][j][k]*met1[i-2][j][k]*(
								c2*(u2[i-2][j][k+2]-u2[i-2][j][k-2]) +
								c1*(u2[i-2][j][k+1]-u2[i-2][j][k-1])   )*stry[j]
							+ la[i-2][j][k]*met4[i-2][j][k]*met1[i-2][j][k]*(
								c2*(u3[i-2][j][k+2]-u3[i-2][j][k-2]) +
								c1*(u3[i-2][j][k+1]-u3[i-2][j][k-1])   ) )
						) + c1*(
							(2*mu[i+1][j][k]+la[i+1][j][k])*met2[i+1][j][k]*met1[i+1][j][k]*(
								c2*(u1[i+1][j][k+2]-u1[i+1][j][k-2]) +
								c1*(u1[i+1][j][k+1]-u1[i+1][j][k-1]) )*strx[i]
							+ la[i+1][j][k]*met3[i+1][j][k]*met1[i+1][j][k]*(
								c2*(u2[i+1][j][k+2]-u2[i+1][j][k-2]) +
								c1*(u2[i+1][j][k+1]-u2[i+1][j][k-1]) )*stry[j]
							+ la[i+1][j][k]*met4[i+1][j][k]*met1[i+1][j][k]*(
								c2*(u3[i+1][j][k+2]-u3[i+1][j][k-2]) +
								c1*(u3[i+1][j][k+1]-u3[i+1][j][k-1])  )
							+ ((2*mu[i-1][j][k]+la[i-1][j][k])*met2[i-1][j][k]*met1[i-1][j][k]*(
									c2*(u1[i-1][j][k+2]-u1[i-1][j][k-2]) +
									c1*(u1[i-1][j][k+1]-u1[i-1][j][k-1]) )*strx[i]
								+ la[i-1][j][k]*met3[i-1][j][k]*met1[i-1][j][k]*(
									c2*(u2[i-1][j][k+2]-u2[i-1][j][k-2]) +
									c1*(u2[i-1][j][k+1]-u2[i-1][j][k-1]) )*stry[j]
								+ la[i-1][j][k]*met4[i-1][j][k]*met1[i-1][j][k]*(
									c2*(u3[i-1][j][k+2]-u3[i-1][j][k-2]) +
									c1*(u3[i-1][j][k+1]-u3[i-1][j][k-1])   )  ) ) )*stry[j];


			r1[i][j][k] += c2*(
					mu[i][j][k+2]*met3[i][j][k+2]*met1[i][j][k+2]*(
						c2*(u1[i][j+2][k+2]-u1[i][j-2][k+2]) +
						c1*(u1[i][j+1][k+2]-u1[i][j-1][k+2])   )*stry[j+2]*strx[i]
					+ la[i][j][k+2]*met2[i][j][k+2]*met1[i][j][k+2]*(
						c2*(u2[i][j+2][k+2]-u2[i][j-2][k+2]) +
						c1*(u2[i][j+1][k+2]-u2[i][j-1][k+2])  )
					+ ( mu[i][j][k-2]*met3[i][j][k-2]*met1[i][j][k-2]*(
							c2*(u1[i][j+2][k-2]-u1[i][j-2][k-2]) +
							c1*(u1[i][j+1][k-2]-u1[i][j-1][k-2])  )*stry[j]*strx[i]
						+ la[i][j][k-2]*met2[i][j][k-2]*met1[i][j][k-2]*(
							c2*(u2[i][j+2][k-2]-u2[i][j-2][k-2]) +
							c1*(u2[i][j+1][k-2]-u2[i][j-1][k-2])   ) )
					) + c1*(
						mu[i][j][k+1]*met3[i][j][k+1]*met1[i][j][k+1]*(
							c2*(u1[i][j+2][k+1]-u1[i][j-2][k+1]) +
							c1*(u1[i][j+1][k+1]-u1[i][j-1][k+1]) )*stry[j-2]*strx[i]
						+ la[i][j][k+1]*met2[i][j][k+1]*met1[i][j][k+1]*(
							c2*(u2[i][j+2][k+1]-u2[i][j-2][k+1]) +
							c1*(u2[i][j+1][k+1]-u2[i][j-1][k+1]) )
						+ ( mu[i][j][k-1]*met3[i][j][k-1]*met1[i][j][k-1]*(
								c2*(u1[i][j+2][k-1]-u1[i][j-2][k-1]) +
								c1*(u1[i][j+1][k-1]-u1[i][j-1][k-1]) )*stry[j]*strx[i]
							+ la[i][j][k-1]*met2[i][j][k-1]*met1[i][j][k-1]*(
								c2*(u2[i][j+2][k-1]-u2[i][j-2][k-1]) +
								c1*(u2[i][j+1][k-1]-u2[i][j-1][k-1]) ) ) );

#pragma end stencil1

			r1[i][j][k] += c2*(
					mu[i][j+2][k]*met3[i][j+2][k]*met1[i][j+2][k]*(
						c2*(u1[i][j+2][k+2]-u1[i][j+2][k-2]) +
						c1*(u1[i][j+2][k+1]-u1[i][j+2][k-1])   )*stry[j+1]*strx[i]
					+ mu[i][j+2][k]*met2[i][j+2][k]*met1[i][j+2][k]*(
						c2*(u2[i][j+2][k+2]-u2[i][j+2][k-2]) +
						c1*(u2[i][j+2][k+1]-u2[i][j+2][k-1])  )
					+ ( mu[i][j-2][k]*met3[i][j-2][k]*met1[i][j-2][k]*(
							c2*(u1[i][j-2][k+2]-u1[i][j-2][k-2]) +
							c1*(u1[i][j-2][k+1]-u1[i][j-2][k-1])  )*stry[j]*strx[i]
						+ mu[i][j-2][k]*met2[i][j-2][k]*met1[i][j-2][k]*(
							c2*(u2[i][j-2][k+2]-u2[i][j-2][k-2]) +
							c1*(u2[i][j-2][k+1]-u2[i][j-2][k-1])   ) )
					) + c1*(
						mu[i][j+1][k]*met3[i][j+1][k]*met1[i][j+1][k]*(
							c2*(u1[i][j+1][k+2]-u1[i][j+1][k-2]) +
							c1*(u1[i][j+1][k+1]-u1[i][j+1][k-1]) )*stry[j-1]*strx[i]
						+ mu[i][j+1][k]*met2[i][j+1][k]*met1[i][j+1][k]*(
							c2*(u2[i][j+1][k+2]-u2[i][j+1][k-2]) +
							c1*(u2[i][j+1][k+1]-u2[i][j+1][k-1]) )
						+ ( mu[i][j-1][k]*met3[i][j-1][k]*met1[i][j-1][k]*(
								c2*(u1[i][j-1][k+2]-u1[i][j-1][k-2]) +
								c1*(u1[i][j-1][k+1]-u1[i][j-1][k-1]) )*stry[j]*strx[i]
							+ mu[i][j-1][k]*met2[i][j-1][k]*met1[i][j-1][k]*(
								c2*(u2[i][j-1][k+2]-u2[i][j-1][k-2]) +
								c1*(u2[i][j-1][k+1]-u2[i][j-1][k-1]) ) ) );

			r1[i][j][k] +=
				c2*(  mu[i][j+2][k]*met1[i][j+2][k]*met1[i][j+2][k]*(
							c2*(u2[i+2][j+2][k]-u2[i-2][j+2][k]) +
							c1*(u2[i+1][j+2][k]-u2[i-1][j+2][k])    )
						+  mu[i][j-2][k]*met1[i][j-2][k]*met1[i][j-2][k]*(
							c2*(u2[i+2][j-2][k]-u2[i-2][j-2][k])+
							c1*(u2[i+1][j-2][k]-u2[i-1][j-2][k])     )
				   ) +
				c1*(  mu[i][j+1][k]*met1[i][j+1][k]*met1[i][j+1][k]*(
							c2*(u2[i+2][j+1][k]-u2[i-2][j+1][k]) +
							c1*(u2[i+1][j+1][k]-u2[i-1][j+1][k])  )
						+ mu[i][j-1][k]*met1[i][j-1][k]*met1[i][j-1][k]*(
							c2*(u2[i+2][j-1][k]-u2[i-2][j-1][k]) +
							c1*(u2[i+1][j-1][k]-u2[i-1][j-1][k])))
				+
				c2*(  la[i+2][j][k]*met1[i+2][j][k]*met1[i+2][j][k]*(
							c2*(u2[i+2][j+2][k]-u2[i+2][j-2][k]) +
							c1*(u2[i+2][j+1][k]-u2[i+2][j-1][k])    )
						+ la[i-2][j][k]*met1[i-2][j][k]*met1[i-2][j][k]*(
							c2*(u2[i-2][j+2][k]-u2[i-2][j-2][k])+
							c1*(u2[i-2][j+1][k]-u2[i-2][j-1][k])     )
				   ) +
				c1*(  la[i+1][j][k]*met1[i+1][j][k]*met1[i+1][j][k]*(
							c2*(u2[i+1][j+2][k]-u2[i+1][j-2][k]) +
							c1*(u2[i+1][j+1][k]-u2[i+1][j-1][k])  )
						+ la[i-1][j][k]*met1[i-1][j][k]*met1[i-1][j][k]*(
							c2*(u2[i-1][j+2][k]-u2[i-1][j-2][k]) +
							c1*(u2[i-1][j+1][k]-u2[i-1][j-1][k])));

		} 
	}
}

extern "C" void host_code (double *h_r1, double *h_u1, double *h_u2, double *h_u3,  double *h_mu, double *h_la, double *h_met1, double *h_met2, double *h_met3, double *h_met4, double *h_strx, double *h_stry, double c1, double c2, int N) {
	double *r1;
	cudaMalloc (&r1, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for r1\n");
	cudaMemcpy (r1, h_r1, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *u1;
	cudaMalloc (&u1, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for u1\n");
	cudaMemcpy (u1, h_u1, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *u2;
	cudaMalloc (&u2, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for u2\n");
	cudaMemcpy (u2, h_u2, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *u3;
	cudaMalloc (&u3, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for u3\n");
	cudaMemcpy (u3, h_u3, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *mu;
	cudaMalloc (&mu, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for mu\n");
	cudaMemcpy (mu, h_mu, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *la;
	cudaMalloc (&la, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for la\n");
	cudaMemcpy (la, h_la, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *met1;
	cudaMalloc (&met1, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for met1\n");
	cudaMemcpy (met1, h_met1, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *met2;
	cudaMalloc (&met2, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for met2\n");
	cudaMemcpy (met2, h_met2, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *met3;
	cudaMalloc (&met3, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for met3\n");
	cudaMemcpy (met3, h_met3, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *met4;
	cudaMalloc (&met4, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for met4\n");
	cudaMemcpy (met4, h_met4, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *strx;
	cudaMalloc (&strx, sizeof(double)*N);
	check_error ("Failed to allocate device memory for strx\n");
	cudaMemcpy (strx, h_strx, sizeof(double)*N, cudaMemcpyHostToDevice);
	double *stry;
	cudaMalloc (&stry, sizeof(double)*N);
	check_error ("Failed to allocate device memory for stry\n");
	cudaMemcpy (stry, h_stry, sizeof(double)*N, cudaMemcpyHostToDevice);

	dim3 blockconfig (16, 8);
	dim3 gridconfig (ceil(N, blockconfig.x), ceil(N, blockconfig.y), 1);

	curvi <<<gridconfig, blockconfig>>> (r1, u1, u2, u3, mu, la, met1, met2, met3, met4, strx, stry, c1, c2, N);
	cudaMemcpy (h_r1, r1, sizeof(double)*N*N*N, cudaMemcpyDeviceToHost);
}
