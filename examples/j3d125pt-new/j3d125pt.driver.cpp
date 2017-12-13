#include "common/common.hpp"
#include <cassert>
#include <cstdio>

extern "C" void host_code (double*, double*, int);
extern "C" void j3d125pt_gold (double*, double*, int);

int main(int argc, char** argv) {
  int N = 516;

  double (*input)[516][516] = (double (*)[516][516]) getRandom3DArray<double>(516, 516, 516);
  double (*output)[516][516] = (double (*)[516][516]) getZero3DArray<double>(516, 516, 516);
  double (*output_gold)[516][516] = (double (*)[516][516]) getZero3DArray<double>(516, 516, 516);

  host_code ((double*)input, (double*)output, N);
  j3d125pt_gold((double*)input, (double*)output_gold, N);

  double error = checkError3D<double> (N, N, (double*)output, (double*) output_gold, 2, N-2, 2, N-2, 2, N-2);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] input;
  delete[] output;
  delete[] output_gold;
}
