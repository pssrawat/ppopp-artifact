extern "C" void j3d125pt_gold (const double* l_in, double* l_out, int N) {
  const double (*in)[516][516] = (const double (*)[516][516])l_in;
  double (*out)[516][516] = (double (*)[516][516])l_out;

  for (int k = 2; k < N-2; k++) {
    for (int j = 2; j < N-2; j++) {
      for (int i = 2; i < N-2; i++) {
		out[k][j][i] =
			0.75 * (in[k-2][j-2][i-2] + in[k-2][j-2][i+2] + in[k-2][j+2][i-2] + in[k-2][j+2][i+2] + 
			in[k-1][j-1][i-1] + in[k-1][j-1][i+1] + in[k-1][j+1][i-1] + in[k-1][j+1][i+1] + 
			in[k][j-1][i] + in[k][j][i-1] +  in[k][j][i+1] + in[k][j+1][i] + 
			in[k+1][j-1][i-1] + in[k+1][j-1][i+1] + in[k+1][j+1][i-1] + in[k+1][j+1][i+1] + 
			in[k+2][j-2][i-2] + in[k+2][j-2][i+2] + in[k+2][j+2][i-2] + in[k+2][j+2][i+2]) + 

			1.132 * (in[k-2][j-2][i-1] + in[k-2][j-2][i+1] + in[k-2][j-1][i-2] + in[k-2][j-1][i+2] + 
			in[k-2][j][i] + in[k-2][j+1][i-2] + in[k-2][j+1][i+2] + in[k-2][j+2][i-1] +  in[k-2][j+2][i+1] +
			in[k-1][j-2][i-2] + in[k-1][j-2][i+2] + in[k-1][j+2][i-2] + in[k-1][j+2][i+2] + 
			in[k][j-2][i] + in[k][j][i-2] + in[k][j][i+2] + in[k][j+2][i] +  
			in[k+1][j-2][i-2] + in[k+1][j-2][i+2] + in[k+1][j+2][i-2] + in[k+1][j+2][i+2] + 
			in[k+2][j-2][i-1] + in[k+2][j-2][i+1] + in[k+2][j-1][i-2] + in[k+2][j-1][i+2] + in[k+2][j][i] + 
			in[k+2][j+1][i-2] + in[k+2][j+1][i+2] + in[k+2][j+2][i-1] +  in[k+2][j+2][i+1]) +

			0.217 * (in[k-2][j-2][i] + in[k-2][j][i-2] + in[k-2][j][i+2] + in[k-2][j+2][i] + 
 			in[k-1][j-1][i] + in[k-1][j][i-1] +  in[k-1][j][i+1] + in[k-1][j+1][i] + 
			in[k][j-2][i-2] + in[k][j-2][i+2] + in[k][j+2][i-2] + in[k][j+2][i+2] + 
			in[k+1][j-1][i] + in[k+1][j][i-1] +  in[k+1][j][i+1] + in[k+1][j+1][i] + 
			in[k+2][j-2][i] + in[k+2][j][i-2] + in[k+2][j][i+2] + in[k+2][j+2][i]) +  

			2.13 * (in[k-2][j-1][i] + in[k-2][j][i-1] +  in[k-2][j][i+1] + in[k-2][j+1][i] +
 			in[k-1][j-2][i] + in[k-1][j][i-2] + in[k-1][j][i+2] + in[k-1][j+2][i] +  
			in[k][j-2][i-1] + in[k][j-2][i+1] + in[k][j-1][i-2] + in[k][j-1][i+2] + 
			in[k][j][i] + in[k][j+1][i-2] + in[k][j+1][i+2] + in[k][j+2][i-1] +  in[k][j+2][i+1] +
			in[k+1][j-2][i] + in[k+1][j][i-2] + in[k+1][j][i+2] + in[k+1][j+2][i] +  
			in[k+2][j-1][i] + in[k+2][j][i-1] +  in[k+2][j][i+1] + in[k+2][j+1][i]) + 

			0.331 * (in[k-2][j-1][i-1] + in[k-2][j-1][i+1] + in[k-2][j+1][i-1] + in[k-2][j+1][i+1] + 
			in[k-1][j-2][i-1] + in[k-1][j-2][i+1] + in[k-1][j-1][i-2] + in[k-1][j-1][i+2] + in[k-1][j][i] + 
			in[k-1][j+1][i-2] + in[k-1][j+1][i+2] + in[k-1][j+2][i-1] +  in[k-1][j+2][i+1] +
			in[k][j-1][i-1] + in[k][j-1][i+1] + in[k][j+1][i-1] + in[k][j+1][i+1] + 
			in[k+1][j-2][i-1] + in[k+1][j-2][i+1] + in[k+1][j-1][i-2] + in[k+1][j-1][i+2] + in[k+1][j][i] + 
			in[k+1][j+1][i-2] + in[k+1][j+1][i+2] + in[k+1][j+2][i-1] +  in[k+1][j+2][i+1] +
			in[k+2][j-1][i-1] + in[k+2][j-1][i+1] + in[k+2][j+1][i-1] + in[k+2][j+1][i+1]); 
      }
    }
  }
}
