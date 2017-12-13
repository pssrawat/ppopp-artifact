extern "C" void j2d64pt_gold (const double* l_in, double* l_out, int N) {
	const double (*in)[8200] = (const double (*)[8200])l_in;
	double (*out)[8200] = (double (*)[8200])l_out;

	for (int j = 4; j < N-4; j++) {
		for (int i = 4; i < N-4; i++) {
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
}
