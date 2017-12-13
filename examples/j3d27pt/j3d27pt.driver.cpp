#include "common/common.hpp"
#include <cassert>
#include <cstdio>

extern "C" void host_code (double*, double*, int);
extern "C" void j3d27pt_gold (double*, double*, int);

int main(int argc, char** argv) {
  int N = 514;

  double (*input)[514][514] = (double (*)[514][514]) getRandom3DArray<double>(514, 514, 514);
  double (*output)[514][514] = (double (*)[514][514]) getZero3DArray<double>(514, 514, 514);
  double (*output_gold)[514][514] = (double (*)[514][514]) getZero3DArray<double>(514, 514, 514);

  host_code ((double*)input, (double*)output, N);
  j3d27pt_gold((double*)input, (double*)output_gold, N);

  double error = checkError3D<double> (N, N, (double*)output, (double*) output_gold, 2, N-2, 2, N-2, 2, N-2);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] input;
  delete[] output;
  delete[] output_gold;
}
