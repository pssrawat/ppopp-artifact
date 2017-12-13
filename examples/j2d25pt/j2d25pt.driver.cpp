#include "common/common.hpp"
#include <cassert>
#include <cstdio>

extern "C" void host_code (double*, double*, int);
extern "C" void j2d25pt_gold (double*, double*, int);

int main(int argc, char** argv) {
  int N = 8196;

  double (*input)[8196] = (double (*)[8196]) getRandom2DArray<double>(8196, 8196);
  double (*output)[8196] = (double (*)[8196]) getZero2DArray<double>(8196, 8196);
  double (*output_gold)[8196] = (double (*)[8196]) getZero2DArray<double>(8196, 8196);

  host_code ((double*)input, (double*)output, N);
  j2d25pt_gold((double*)input, (double*)output_gold, N);

  double error = checkError2D<double> (N, (double*)output, (double*) output_gold, 2, N-2, 2, N-2);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] input;
  delete[] output;
  delete[] output_gold;
}
