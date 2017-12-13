#include "common/common.hpp"
#include <cassert>
#include <cstdio>

extern "C" void host_code (double*, double*, int);
extern "C" void j2d64pt_gold (double*, double*, int);

int main(int argc, char** argv) {
  int N = 8200;

  double (*input)[8200] = (double (*)[8200]) getRandom2DArray<double>(8200, 8200);
  double (*output)[8200] = (double (*)[8200]) getZero2DArray<double>(8200, 8200);
  double (*output_gold)[8200] = (double (*)[8200]) getZero2DArray<double>(8200, 8200);

  host_code ((double*)input, (double*)output, N);
  j2d64pt_gold((double*)input, (double*)output_gold, N);

  double error = checkError2D<double> (N, (double*)output, (double*) output_gold, 4, N-4, 4, N-4);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] input;
  delete[] output;
  delete[] output_gold;
}
