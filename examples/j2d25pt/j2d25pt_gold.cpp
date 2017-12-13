extern "C" void j2d25pt_gold (const double* l_in, double* l_out, int N) {
	const double (*in)[8196] = (const double (*)[8196])l_in;
	double (*out)[8196] = (double (*)[8196])l_out;

	for (int j = 2; j < N-2; j++) {
		for (int i = 2; i < N-2; i++) {
                out[j][i] = 0.1*(in[j-2][i-2] + in[j-2][i+2] + in[j+2][i-2] + in[j+2][i+2]) +
                        0.2*(in[j-2][i-1] + in[j-2][i+1] + in[j+2][i-1] + in[j+2][i+1]) +
                        0.3*(in[j-2][i] + in[j+2][i]) +
                        1.1*(in[j-1][i-2] + in[j-1][i+2] + in[j+1][i-2] + in[j+1][i+2]) +
                        1.2*(in[j-1][i-1] + in[j-1][i+1] + in[j+1][i-1] + in[j+1][i+1]) +
                        1.3*(in[j-1][i] + in[j+1][i]) + 
                        2.1*(in[j][i-2] + in[j][i+2]) +
                        2.2*(in[j][i-1] + in[j][i+1]) +
                        2.3*in[j][i]; 
		}

	}
}
