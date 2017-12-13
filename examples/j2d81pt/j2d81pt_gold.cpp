extern "C" void j2d81pt_gold (const double* l_in, double* l_out, int N) {
	const double (*in)[8200] = (const double (*)[8200])l_in;
	double (*out)[8200] = (double (*)[8200])l_out;

	for (int j = 0; j < N-8; j++) {
		for (int i = 0; i < N-8; i++) {
			out[j][i] =
				(in[j][i] + in[j][i+8] + in[j+8][i] + in[j+8][i+8]) * 3.1862206 +
				(in[j][i+1] + in[j][i+7] + in[j+1][i] + in[j+1][i+8] + in[j+7][i] + in[j+7][i+8] + in[j+8][i+1] + in[j+8][i+7]) * 4.5339005 +
				(in[j][i+2] + in[j][i+6] + in[j+2][i] + in[j+2][i+8] + in[j+6][i] + in[j+6][i+8] + in[j+8][i+2] + in[j+8][i+6]) * -0.000357000 +
				(in[j][i+3] + in[j][i+5] + in[j+3][i] + in[j+3][i+8] + in[j+5][i] + in[j+5][i+8] + in[j+8][i+3] + in[j+8][i+5]) * 0.00285600 +
				(in[j][i+4] + in[j+4][i+8] + in[j+4][i] + in[j+8][i+4]) * -0.00508225 +
				(in[j+1][i+1] + in[j+1][i+7] + in[j+7][i+1] + in[j+7][i+7]) * 0.000645160 +
				(in[j+1][i+2] + in[j+1][i+6] + in[j+2][i+1] + in[j+2][i+7] + in[j+6][i+1] + in[j+6][i+7] + in[j+7][i+2] + in[j+7][i+6]) * -0.00508000 +
				(in[j+1][i+3] + in[j+1][i+5] + in[j+3][i+1] + in[j+3][i+7] + in[j+5][i+1] + in[j+5][i+7] + in[j+7][i+3] + in[j+7][i+5]) * 0.0406400 +
				(in[j+1][i+4] + in[j+4][i+1] + in[j+4][i+7] + in[j+7][i+4]) * -0.0723189 +
				(in[j+2][i+2] + in[j+2][i+6] + in[j+6][i+2] + in[j+6][i+6]) * 0.0400000 +
				(in[j+2][i+3] + in[j+2][i+5] + in[j+3][i+2] + in[j+3][i+6] + in[j+5][i+2] + in[j+5][i+6] + in[j+6][i+3] + in[j+6][i+5]) * -0.320000 +
				(in[j+2][i+4] + in[j+4][i+2] + in[j+4][i+6] + in[j+6][i+4]) * 0.569440 +
				(in[j+3][i+3] + in[j+3][i+5] + in[j+5][i+3] + in[j+5][i+5]) * 2.56000 +
				(in[j+3][i+4] + in[j+4][i+3] + in[j+4][i+5] + in[j+5][i+4]) * -4.55552 +
				in[j+4][i+4] * 8.10655;
		}

	}
}
