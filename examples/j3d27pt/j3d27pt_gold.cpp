extern "C" void j3d27pt_gold (const double* l_in, double* l_out, int N) {
	const double (*in)[514][514] = (const double (*)[514][514])l_in;
	double (*out)[514][514] = (double (*)[514][514])l_out;

	for (int k = 1; k < N-1; k++) {
		for (int j = 1; j < N-1; j++) {
			for (int i = 1; i < N-1; i++) {
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
			}
		}
	}
}
