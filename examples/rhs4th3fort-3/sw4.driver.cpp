#include "common/common.hpp"
#include <cassert>
#include <cstdio>

extern "C" void sw4_gold (double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, int);
extern "C" void host_code (double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, int);

int main(int argc, char** argv) {
  int N = 304; 

  double (*u_0)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*u_1)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*u_2)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*mu)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*la)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double *strx = (double *) getRandom1DArray<double>(304);
  double *stry = (double *) getRandom1DArray<double>(304);
  double *strz = (double *) getRandom1DArray<double>(304);
  double (*uacc_gold_0)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*uacc_gold_1)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*uacc_gold_2)[304][304] = (double (*)[304][304]) getRandom3DArray<double>(304, 304, 304);
  double (*uacc_0)[304][304] = (double (*)[304][304]) getZero3DArray<double>(304, 304, 304);
  double (*uacc_1)[304][304] = (double (*)[304][304]) getZero3DArray<double>(304, 304, 304);
  double (*uacc_2)[304][304] = (double (*)[304][304]) getZero3DArray<double>(304, 304, 304);
  memcpy(uacc_0, uacc_gold_0, sizeof(double)*N*N*N);
  memcpy(uacc_1, uacc_gold_1, sizeof(double)*N*N*N);
  memcpy(uacc_2, uacc_gold_2, sizeof(double)*N*N*N);

  sw4_gold ((double*)uacc_gold_0, (double*)uacc_gold_1, (double*)uacc_gold_2, (double*)u_0, (double*)u_1, (double*)u_2, (double*)mu, (double*)la, (double*)strx, (double*)stry, (double*)strz, N);
  host_code ((double*)uacc_0, (double*)uacc_1, (double*)uacc_2, (double*)u_0, (double*)u_1, (double*)u_2, (double*)mu, (double*)la, (double*)strx, (double*)stry, (double*)strz, N);

  double error_0 = checkError3D<double> (N, N, (double*)uacc_0, (double*)uacc_gold_0, 2, N-2, 2, N-2, 2, N-2);
  printf("[Test] RMS Error : %e\n",error_0);
  if (error_0 > TOLERANCE)
    return -1;
  double error_1 = checkError3D<double> (N, N, (double*)uacc_1, (double*)uacc_gold_1, 2, N-2, 2, N-2, 2, N-2);
  printf("[Test] RMS Error : %e\n",error_1);
  if (error_1 > TOLERANCE)
    return -1;
  double error_2 = checkError3D<double> (N, N, (double*)uacc_2, (double*)uacc_gold_2, 2, N-2, 2, N-2, 2, N-2);
  printf("[Test] RMS Error : %e\n",error_2);
  if (error_2 > TOLERANCE)
    return -1;

  delete[] strx;
  delete[] stry;
  delete[] strz;
  delete[] u_0;
  delete[] u_1;
  delete[] u_2;
  delete[] mu;
  delete[] la;
  delete[] uacc_0;
  delete[] uacc_1;
  delete[] uacc_2;
  delete[] uacc_gold_0;
  delete[] uacc_gold_1;
  delete[] uacc_gold_2;
}
