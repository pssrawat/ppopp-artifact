#include "common/common.hpp"
#include <cassert>
#include <cstdio>

extern "C" void derivative_gold (double *h_r1, double *h_u1, double *h_u2, double *h_u3, double *h_mu, double *h_la, double *h_met1, double *h_met2, double *h_met3, double *h_met4, double *, double *, double c1, double c2, int N); 
extern "C" void host_code (double *h_r1, double *h_u1, double *h_u2, double *h_u3, double *h_mu, double *h_la, double *h_met1, double *h_met2, double *h_met3, double *h_met4, double *, double *, double c1, double c2, int N); 

int main(int argc, char** argv) {
  int N = 304; 

  double (*r_gold_0)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*mu)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*la)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*met1)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*met2)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*met3)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*met4)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*u1)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*u2)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*u3)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*r_0)[304][304] = (double (*)[304][304]) getZero3DArray<double>(304, 304, 304);
  memcpy(r_0, r_gold_0, sizeof(double)*304*304*304);
  double *strx = (double *) getRandom1DArray<double>(304);
  double *stry = (double *) getRandom1DArray<double>(304);
 
  double c1 = 0.32;
  double c2 = 0.43;
  derivative_gold ((double*)r_gold_0, (double *)u1, (double *)u2, (double *)u3, (double*)mu, (double*)la, (double*)met1, (double*)met2, (double *)met3, (double*)met4, strx, stry, c1, c2, N);
  host_code ((double*)r_0, (double *)u1, (double *)u2, (double *)u3, (double*)mu, (double*)la, (double*)met1, (double*)met2, (double *)met3, (double*)met4, strx, stry, c1, c2, N);
  double error_0 = checkError3D<double> (N, N, (double*)r_0, (double*)r_gold_0, 2, N-2, 2, N-2, 2, N-2);
  printf("[Test] RMS Error : %e\n",error_0);
  if (error_0 > TOLERANCE)
    return -1;
}
