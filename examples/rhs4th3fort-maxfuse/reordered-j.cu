#include <stdio.h>
#include "cuda.h"
#define max(x,y)  ((x) > (y)? (x) : (y))
#define min(x,y)  ((x) < (y)? (x) : (y))
#define ceil(a,b) ((a) % (b) == 0 ? (a) / (b) : ((a) / (b)) + 1)

void check_error (const char* message) {
	cudaError_t error = cudaGetLastError ();
	if (error != cudaSuccess) {
		printf ("CUDA error : %s, %s\n", message, cudaGetErrorString (error));
		exit(-1);
	}
}

__global__ void sw4 (double * uacc_in_0, double * uacc_in_1, double * uacc_in_2, double * __restrict__ u_in_0, double * __restrict__ u_in_1, double * __restrict__ u_in_2, double * __restrict__ mu_in, double * __restrict__ la_in, double * strx, double * stry, double * strz, int N) {
	//Determing the block's indices
	int blockdim_i= (int)(blockDim.x);
	int i0 = (int)(blockIdx.x)*(blockdim_i);
	int i = max (i0, 0) + (int)(threadIdx.x);
	int blockdim_j= (int)(blockDim.y);
	int j0 = (int)(blockIdx.y)*(blockdim_j);
	int j = max (j0, 0) + (int)(threadIdx.y);
	int blockdim_k= (int)(blockDim.z);
	int k0 = (int)(blockIdx.z)*(blockdim_k);
	int k = max (k0, 0) + (int)(threadIdx.z);

	// Assumptions 
	int a1 = 1;
	double h = 3.7;
	double cof = 1e0 / ( h *  h);

	double (*uacc_0)[304][304] = (double (*)[304][304])uacc_in_0;
	double (*uacc_1)[304][304] = (double (*)[304][304])uacc_in_1;
	double (*uacc_2)[304][304] = (double (*)[304][304])uacc_in_2;
	double (*u_0)[304][304] = (double (*)[304][304])u_in_0;
	double (*u_1)[304][304] = (double (*)[304][304])u_in_1;
	double (*u_2)[304][304] = (double (*)[304][304])u_in_2;
	double (*mu)[304][304] = (double (*)[304][304])mu_in;
	double (*la)[304][304] = (double (*)[304][304])la_in;

	double mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
	double r1, r2, r3;
	if (i>=2 & j>=2 & k>=2 & i<=N-3 & j<=N-3 & k<=N-3) {
double mux1;
double mux2;
double mux3;
double mux4;
double muy1;
double muy2;
double muy3;
double muy4;
double muz1;
double muz2;
double muz3;
double muz4;
double _t_2_;
double _t_3_;
double _t_1_;
double _t_4_;
double _t_5_;
double _t_6_;
double _t_7_;
double _t_8_;
double _t_9_;
double _t_0_;
double _t_11_;
double _t_10_;
double _t_12_;
double _t_13_;
double _t_14_;
double _t_16_;
double _t_15_;
double _t_17_;
double _t_18_;
double _t_19_;
double r1;
double _t_22_;
double _t_21_;
double _t_23_;
double _t_24_;
double _t_25_;
double _t_20_;
double _t_27_;
double _t_28_;
double _t_26_;
double _t_29_;
double _t_30_;
double _t_31_;
double _t_32_;
double _t_33_;
double _t_34_;
double _t_36_;
double _t_35_;
double _t_37_;
double _t_38_;
double _t_39_;
double r2;
double _t_42_;
double _t_41_;
double _t_43_;
double _t_44_;
double _t_45_;
double _t_40_;
double _t_47_;
double _t_46_;
double _t_48_;
double _t_49_;
double _t_50_;
double _t_52_;
double _t_53_;
double _t_51_;
double _t_54_;
double _t_55_;
double _t_56_;
double _t_57_;
double _t_58_;
double _t_59_;
double r3;
double _t_63_;
double _t_61_;
double _t_64_;
double _t_65_;
double _t_62_;
double _t_67_;
double _t_68_;
double _t_66_;
double _t_70_;
double _t_71_;
double _t_69_;
double _t_72_;
double _t_73_;
double _t_60_;
double _t_76_;
double _t_74_;
double _t_77_;
double _t_78_;
double _t_75_;
double _t_80_;
double _t_81_;
double _t_79_;
double _t_83_;
double _t_84_;
double _t_82_;
double _t_85_;
double _t_86_;
double _t_89_;
double _t_87_;
double _t_90_;
double _t_91_;
double _t_88_;
double _t_93_;
double _t_94_;
double _t_92_;
double _t_96_;
double _t_97_;
double _t_95_;
double _t_98_;
double _t_99_;
double _t_102_;
double _t_100_;
double _t_103_;
double _t_104_;
double _t_101_;
double _t_106_;
double _t_107_;
double _t_105_;
double _t_109_;
double _t_110_;
double _t_108_;
double _t_111_;
double _t_112_;
double _t_116_;
double _t_114_;
double _t_117_;
double _t_118_;
double _t_115_;
double _t_120_;
double _t_121_;
double _t_119_;
double _t_123_;
double _t_124_;
double _t_122_;
double _t_125_;
double _t_126_;
double _t_113_;
double _t_129_;
double _t_127_;
double _t_130_;
double _t_131_;
double _t_128_;
double _t_133_;
double _t_134_;
double _t_132_;
double _t_136_;
double _t_137_;
double _t_135_;
double _t_138_;
double _t_139_;
double _t_142_;
double _t_140_;
double _t_143_;
double _t_144_;
double _t_141_;
double _t_146_;
double _t_147_;
double _t_145_;
double _t_149_;
double _t_150_;
double _t_148_;
double _t_151_;
double _t_152_;
double _t_155_;
double _t_153_;
double _t_156_;
double _t_157_;
double _t_154_;
double _t_159_;
double _t_160_;
double _t_158_;
double _t_162_;
double _t_163_;
double _t_161_;
double _t_164_;
double _t_165_;
double _t_169_;
double _t_167_;
double _t_170_;
double _t_171_;
double _t_168_;
double _t_173_;
double _t_174_;
double _t_172_;
double _t_176_;
double _t_177_;
double _t_175_;
double _t_178_;
double _t_179_;
double _t_166_;
double _t_182_;
double _t_180_;
double _t_183_;
double _t_184_;
double _t_181_;
double _t_186_;
double _t_187_;
double _t_185_;
double _t_189_;
double _t_190_;
double _t_188_;
double _t_191_;
double _t_192_;
double _t_195_;
double _t_193_;
double _t_196_;
double _t_197_;
double _t_194_;
double _t_199_;
double _t_200_;
double _t_198_;
double _t_202_;
double _t_203_;
double _t_201_;
double _t_204_;
double _t_205_;
double _t_208_;
double _t_206_;
double _t_209_;
double _t_210_;
double _t_207_;
double _t_212_;
double _t_213_;
double _t_211_;
double _t_215_;
double _t_216_;
double _t_214_;
double _t_217_;
double _t_218_;
double uacc_0kc0jc0ic0;
double uacc_1kc0jc0ic0;
double uacc_2kc0jc0ic0;

mux1 = mu[k][j][i-1] * strx[i-1];
mux1 -= 3.0 / 4.0 * mu[k][j][i] * strx[i];
mux1 -= 3.0 / 4.0 * mu[k][j][i-2] * strx[i-2];
mux2 = mu[k][j][i-2] * strx[i-2];
mux2 += mu[k][j][i+1] * strx[i+1];
mux2 += 3.0 * mu[k][j][i] * strx[i];
mux2 += 3.0 * mu[k][j][i-1] * strx[i-1];
mux3 = mu[k][j][i-1] * strx[i-1];
mux3 += mu[k][j][i+2] * strx[i+2];
mux3 += 3.0 * mu[k][j][i+1] * strx[i+1];
mux3 += 3.0 * mu[k][j][i] * strx[i];
mux4 = mu[k][j][i+1] * strx[i+1];
mux4 -= 3.0 / 4.0 * mu[k][j][i] * strx[i];
mux4 -= 3.0 / 4.0 * mu[k][j][i+2] * strx[i+2];
muy1 = mu[k][j-1][i] * stry[j-1];
muy1 -= 3.0 / 4.0 * mu[k][j][i] * stry[j];
muy1 -= 3.0 / 4.0 * mu[k][j-2][i] * stry[j-2];
muy2 = mu[k][j-2][i] * stry[j-2];
muy2 += mu[k][j+1][i] * stry[j+1];
muy2 += 3.0 * mu[k][j][i] * stry[j];
muy2 += 3.0 * mu[k][j-1][i] * stry[j-1];
muy3 = mu[k][j-1][i] * stry[j-1];
muy3 += mu[k][j+2][i] * stry[j+2];
muy3 += 3.0 * mu[k][j+1][i] * stry[j+1];
muy3 += 3.0 * mu[k][j][i] * stry[j];
muy4 = mu[k][j+1][i] * stry[j+1];
muy4 -= 3.0 / 4.0 * mu[k][j][i] * stry[j];
muy4 -= 3.0 / 4.0 * mu[k][j+2][i] * stry[j+2];
muz1 = mu[k-1][j][i] * strz[k-1];
muz1 -= 3.0 / 4.0 * mu[k][j][i] * strz[k];
muz1 -= 3.0 / 4.0 * mu[k-2][j][i] * strz[k-2];
muz2 = mu[k-2][j][i] * strz[k-2];
muz2 += mu[k+1][j][i] * strz[k+1];
muz2 += 3.0 * mu[k][j][i] * strz[k];
muz2 += 3.0 * mu[k-1][j][i] * strz[k-1];
muz3 = mu[k-1][j][i] * strz[k-1];
muz3 += mu[k+2][j][i] * strz[k+2];
muz3 += 3.0 * mu[k+1][j][i] * strz[k+1];
muz3 += 3.0 * mu[k][j][i] * strz[k];
muz4 = mu[k+1][j][i] * strz[k+1];
muz4 -= 3.0 / 4.0 * mu[k][j][i] * strz[k];
muz4 -= 3.0 / 4.0 * mu[k+2][j][i] * strz[k+2];
_t_2_ = 2.0 * mux1;
_t_2_ += la[k][j][i-1] * strx[i-1];
_t_2_ -= 3.0 / 4.0 * la[k][j][i] * strx[i];
_t_2_ -= 3.0 / 4.0 * la[k][j][i-2] * strx[i-2];
_t_3_ = u_0[k][j][i-2];
_t_3_ -= u_0[k][j][i];
_t_1_ = _t_2_ * _t_3_;
_t_4_ = 2.0 * mux2;
_t_4_ += la[k][j][i-2] * strx[i-2];
_t_4_ += la[k][j][i+1] * strx[i+1];
_t_4_ += 3.0 * la[k][j][i] * strx[i];
_t_4_ += 3.0 * la[k][j][i-1] * strx[i-1];
_t_5_ = u_0[k][j][i-1];
_t_5_ -= u_0[k][j][i];
_t_1_ += _t_4_ * _t_5_;
_t_6_ = 2.0 * mux3;
_t_6_ += la[k][j][i-1] * strx[i-1];
_t_6_ += la[k][j][i+2] * strx[i+2];
_t_6_ += 3.0 * la[k][j][i+1] * strx[i+1];
_t_6_ += 3.0 * la[k][j][i] * strx[i];
_t_7_ = u_0[k][j][i+1];
_t_7_ -= u_0[k][j][i];
_t_1_ += _t_6_ * _t_7_;
_t_8_ = 2.0 * mux4;
_t_8_ += la[k][j][i+1] * strx[i+1];
_t_8_ -= 3.0 / 4.0 * la[k][j][i] * strx[i];
_t_8_ -= 3.0 / 4.0 * la[k][j][i+2] * strx[i+2];
_t_9_ = u_0[k][j][i+2];
_t_9_ -= u_0[k][j][i];
_t_1_ += _t_8_ * _t_9_;
_t_0_ = strx[i] * _t_1_;
_t_11_ = u_0[k][j-2][i];
_t_11_ -= u_0[k][j][i];
_t_10_ = muy1 * _t_11_;
_t_12_ = u_0[k][j-1][i];
_t_12_ -= u_0[k][j][i];
_t_10_ += muy2 * _t_12_;
_t_13_ = u_0[k][j+1][i];
_t_13_ -= u_0[k][j][i];
_t_10_ += muy3 * _t_13_;
_t_14_ = u_0[k][j+2][i];
_t_14_ -= u_0[k][j][i];
_t_10_ += muy4 * _t_14_;
_t_0_ += stry[j] * _t_10_;
_t_16_ = u_0[k-2][j][i];
_t_16_ -= u_0[k][j][i];
_t_15_ = muz1 * _t_16_;
_t_17_ = u_0[k-1][j][i];
_t_17_ -= u_0[k][j][i];
_t_15_ += muz2 * _t_17_;
_t_18_ = u_0[k+1][j][i];
_t_18_ -= u_0[k][j][i];
_t_15_ += muz3 * _t_18_;
_t_19_ = u_0[k+2][j][i];
_t_19_ -= u_0[k][j][i];
_t_15_ += muz4 * _t_19_;
_t_0_ += strz[k] * _t_15_;
r1 = 1.0 / 6.0 * _t_0_;
_t_22_ = u_1[k][j][i-2];
_t_22_ -= u_1[k][j][i];
_t_21_ = mux1 * _t_22_;
_t_23_ = u_1[k][j][i-1];
_t_23_ -= u_1[k][j][i];
_t_21_ += mux2 * _t_23_;
_t_24_ = u_1[k][j][i+1];
_t_24_ -= u_1[k][j][i];
_t_21_ += mux3 * _t_24_;
_t_25_ = u_1[k][j][i+2];
_t_25_ -= u_1[k][j][i];
_t_21_ += mux4 * _t_25_;
_t_20_ = strx[i] * _t_21_;
_t_27_ = 2.0 * muy1;
_t_27_ += la[k][j-1][i] * stry[j-1];
_t_27_ -= 3.0 / 4.0 * la[k][j][i] * stry[j];
_t_27_ -= 3.0 / 4.0 * la[k][j-2][i] * stry[j-2];
_t_28_ = u_1[k][j-2][i];
_t_28_ -= u_1[k][j][i];
_t_26_ = _t_27_ * _t_28_;
_t_29_ = 2.0 * muy2;
_t_29_ += la[k][j-2][i] * stry[j-2];
_t_29_ += la[k][j+1][i] * stry[j+1];
_t_29_ += 3.0 * la[k][j][i] * stry[j];
_t_29_ += 3.0 * la[k][j-1][i] * stry[j-1];
_t_30_ = u_1[k][j-1][i];
_t_30_ -= u_1[k][j][i];
_t_26_ += _t_29_ * _t_30_;
_t_31_ = 2.0 * muy3;
_t_31_ += la[k][j-1][i] * stry[j-1];
_t_31_ += la[k][j+2][i] * stry[j+2];
_t_31_ += 3.0 * la[k][j+1][i] * stry[j+1];
_t_31_ += 3.0 * la[k][j][i] * stry[j];
_t_32_ = u_1[k][j+1][i];
_t_32_ -= u_1[k][j][i];
_t_26_ += _t_31_ * _t_32_;
_t_33_ = 2.0 * muy4;
_t_33_ += la[k][j+1][i] * stry[j+1];
_t_33_ -= 3.0 / 4.0 * la[k][j][i] * stry[j];
_t_33_ -= 3.0 / 4.0 * la[k][j+2][i] * stry[j+2];
_t_34_ = u_1[k][j+2][i];
_t_34_ -= u_1[k][j][i];
_t_26_ += _t_33_ * _t_34_;
_t_20_ += stry[j] * _t_26_;
_t_36_ = u_1[k-2][j][i];
_t_36_ -= u_1[k][j][i];
_t_35_ = muz1 * _t_36_;
_t_37_ = u_1[k-1][j][i];
_t_37_ -= u_1[k][j][i];
_t_35_ += muz2 * _t_37_;
_t_38_ = u_1[k+1][j][i];
_t_38_ -= u_1[k][j][i];
_t_35_ += muz3 * _t_38_;
_t_39_ = u_1[k+2][j][i];
_t_39_ -= u_1[k][j][i];
_t_35_ += muz4 * _t_39_;
_t_20_ += strz[k] * _t_35_;
r2 = 1.0 / 6.0 * _t_20_;
_t_42_ = u_2[k][j][i-2];
_t_42_ -= u_2[k][j][i];
_t_41_ = mux1 * _t_42_;
_t_43_ = u_2[k][j][i-1];
_t_43_ -= u_2[k][j][i];
_t_41_ += mux2 * _t_43_;
_t_44_ = u_2[k][j][i+1];
_t_44_ -= u_2[k][j][i];
_t_41_ += mux3 * _t_44_;
_t_45_ = u_2[k][j][i+2];
_t_45_ -= u_2[k][j][i];
_t_41_ += mux4 * _t_45_;
_t_40_ = strx[i] * _t_41_;
_t_47_ = u_2[k][j-2][i];
_t_47_ -= u_2[k][j][i];
_t_46_ = muy1 * _t_47_;
_t_48_ = u_2[k][j-1][i];
_t_48_ -= u_2[k][j][i];
_t_46_ += muy2 * _t_48_;
_t_49_ = u_2[k][j+1][i];
_t_49_ -= u_2[k][j][i];
_t_46_ += muy3 * _t_49_;
_t_50_ = u_2[k][j+2][i];
_t_50_ -= u_2[k][j][i];
_t_46_ += muy4 * _t_50_;
_t_40_ += stry[j] * _t_46_;
_t_52_ = 2.0 * muz1;
_t_52_ += la[k-1][j][i] * strz[k-1];
_t_52_ -= 3.0 / 4.0 * la[k][j][i] * strz[k];
_t_52_ -= 3.0 / 4.0 * la[k-2][j][i] * strz[k-2];
_t_53_ = u_2[k-2][j][i];
_t_53_ -= u_2[k][j][i];
_t_51_ = _t_52_ * _t_53_;
_t_54_ = 2.0 * muz2;
_t_54_ += la[k-2][j][i] * strz[k-2];
_t_54_ += la[k+1][j][i] * strz[k+1];
_t_54_ += 3.0 * la[k][j][i] * strz[k];
_t_54_ += 3.0 * la[k-1][j][i] * strz[k-1];
_t_55_ = u_2[k-1][j][i];
_t_55_ -= u_2[k][j][i];
_t_51_ += _t_54_ * _t_55_;
_t_56_ = 2.0 * muz3;
_t_56_ += la[k-1][j][i] * strz[k-1];
_t_56_ += la[k+2][j][i] * strz[k+2];
_t_56_ += 3.0 * la[k+1][j][i] * strz[k+1];
_t_56_ += 3.0 * la[k][j][i] * strz[k];
_t_57_ = u_2[k+1][j][i];
_t_57_ -= u_2[k][j][i];
_t_51_ += _t_56_ * _t_57_;
_t_58_ = 2.0 * muz4;
_t_58_ += la[k+1][j][i] * strz[k+1];
_t_58_ -= 3.0 / 4.0 * la[k][j][i] * strz[k];
_t_58_ -= 3.0 / 4.0 * la[k+2][j][i] * strz[k+2];
_t_59_ = u_2[k+2][j][i];
_t_59_ -= u_2[k][j][i];
_t_51_ += _t_58_ * _t_59_;
_t_40_ += strz[k] * _t_51_;
r3 = 1.0 / 6.0 * _t_40_;
_t_63_ = strx[i] * strz[k];
_t_61_ = _t_63_ * 1.0 / 144.0;
_t_64_ = u_0[k-2][j][i-2];
_t_64_ -= u_0[k+2][j][i-2];
_t_65_ = -u_0[k-1][j][i-2];
_t_65_ += u_0[k+1][j][i-2];
_t_64_ += 8.0 * _t_65_;
_t_62_ = mu[k][j][i-2] * _t_64_;
_t_67_ = u_0[k-2][j][i-1];
_t_67_ -= u_0[k+2][j][i-1];
_t_68_ = -u_0[k-1][j][i-1];
_t_68_ += u_0[k+1][j][i-1];
_t_67_ += 8.0 * _t_68_;
_t_66_ = mu[k][j][i-1] * _t_67_;
_t_62_ -= 8.0 * _t_66_;
_t_70_ = u_0[k-2][j][i+1];
_t_70_ -= u_0[k+2][j][i+1];
_t_71_ = -u_0[k-1][j][i+1];
_t_71_ += u_0[k+1][j][i+1];
_t_70_ += 8.0 * _t_71_;
_t_69_ = mu[k][j][i+1] * _t_70_;
_t_62_ += 8.0 * _t_69_;
_t_72_ = u_0[k-2][j][i+2];
_t_72_ -= u_0[k+2][j][i+2];
_t_73_ = -u_0[k-1][j][i+2];
_t_73_ += u_0[k+1][j][i+2];
_t_72_ += 8.0 * _t_73_;
_t_62_ -= mu[k][j][i+2] * _t_72_;
_t_60_ = _t_61_ * _t_62_;
_t_76_ = stry[j] * strz[k];
_t_74_ = _t_76_ * 1.0 / 144.0;
_t_77_ = u_1[k-2][j-2][i];
_t_77_ -= u_1[k+2][j-2][i];
_t_78_ = -u_1[k-1][j-2][i];
_t_78_ += u_1[k+1][j-2][i];
_t_77_ += 8.0 * _t_78_;
_t_75_ = mu[k][j-2][i] * _t_77_;
_t_80_ = u_1[k-2][j-1][i];
_t_80_ -= u_1[k+2][j-1][i];
_t_81_ = -u_1[k-1][j-1][i];
_t_81_ += u_1[k+1][j-1][i];
_t_80_ += 8.0 * _t_81_;
_t_79_ = mu[k][j-1][i] * _t_80_;
_t_75_ -= 8.0 * _t_79_;
_t_83_ = u_1[k-2][j+1][i];
_t_83_ -= u_1[k+2][j+1][i];
_t_84_ = -u_1[k-1][j+1][i];
_t_84_ += u_1[k+1][j+1][i];
_t_83_ += 8.0 * _t_84_;
_t_82_ = mu[k][j+1][i] * _t_83_;
_t_75_ += 8.0 * _t_82_;
_t_85_ = u_1[k-2][j+2][i];
_t_85_ -= u_1[k+2][j+2][i];
_t_86_ = -u_1[k-1][j+2][i];
_t_86_ += u_1[k+1][j+2][i];
_t_85_ += 8.0 * _t_86_;
_t_75_ -= mu[k][j+2][i] * _t_85_;
_t_60_ += _t_74_ * _t_75_;
_t_89_ = strx[i] * strz[k];
_t_87_ = _t_89_ * 1.0 / 144.0;
_t_90_ = u_0[k-2][j][i-2];
_t_90_ -= u_0[k-2][j][i+2];
_t_91_ = -u_0[k-2][j][i-1];
_t_91_ += u_0[k-2][j][i+1];
_t_90_ += 8.0 * _t_91_;
_t_88_ = la[k-2][j][i] * _t_90_;
_t_93_ = u_0[k-1][j][i-2];
_t_93_ -= u_0[k-1][j][i+2];
_t_94_ = -u_0[k-1][j][i-1];
_t_94_ += u_0[k-1][j][i+1];
_t_93_ += 8.0 * _t_94_;
_t_92_ = la[k-1][j][i] * _t_93_;
_t_88_ -= 8.0 * _t_92_;
_t_96_ = u_0[k+1][j][i-2];
_t_96_ -= u_0[k+1][j][i+2];
_t_97_ = -u_0[k+1][j][i-1];
_t_97_ += u_0[k+1][j][i+1];
_t_96_ += 8.0 * _t_97_;
_t_95_ = la[k+1][j][i] * _t_96_;
_t_88_ += 8.0 * _t_95_;
_t_98_ = u_0[k+2][j][i-2];
_t_98_ -= u_0[k+2][j][i+2];
_t_99_ = -u_0[k+2][j][i-1];
_t_99_ += u_0[k+2][j][i+1];
_t_98_ += 8.0 * _t_99_;
_t_88_ -= la[k+2][j][i] * _t_98_;
_t_60_ += _t_87_ * _t_88_;
_t_102_ = stry[j] * strz[k];
_t_100_ = _t_102_ * 1.0 / 144.0;
_t_103_ = u_1[k-2][j-2][i];
_t_103_ -= u_1[k-2][j+2][i];
_t_104_ = -u_1[k-2][j-1][i];
_t_104_ += u_1[k-2][j+1][i];
_t_103_ += 8.0 * _t_104_;
_t_101_ = la[k-2][j][i] * _t_103_;
_t_106_ = u_1[k-1][j-2][i];
_t_106_ -= u_1[k-1][j+2][i];
_t_107_ = -u_1[k-1][j-1][i];
_t_107_ += u_1[k-1][j+1][i];
_t_106_ += 8.0 * _t_107_;
_t_105_ = la[k-1][j][i] * _t_106_;
_t_101_ -= 8.0 * _t_105_;
_t_109_ = u_1[k+1][j-2][i];
_t_109_ -= u_1[k+1][j+2][i];
_t_110_ = -u_1[k+1][j-1][i];
_t_110_ += u_1[k+1][j+1][i];
_t_109_ += 8.0 * _t_110_;
_t_108_ = la[k+1][j][i] * _t_109_;
_t_101_ += 8.0 * _t_108_;
_t_111_ = u_1[k+2][j-2][i];
_t_111_ -= u_1[k+2][j+2][i];
_t_112_ = -u_1[k+2][j-1][i];
_t_112_ += u_1[k+2][j+1][i];
_t_111_ += 8.0 * _t_112_;
_t_101_ -= la[k+2][j][i] * _t_111_;
_t_60_ += _t_100_ * _t_101_;
r3 += _t_60_;
_t_116_ = strx[i] * stry[j];
_t_114_ = _t_116_ * 1.0 / 144.0;
_t_117_ = u_1[k][j-2][i-2];
_t_117_ -= u_1[k][j+2][i-2];
_t_118_ = -u_1[k][j-1][i-2];
_t_118_ += u_1[k][j+1][i-2];
_t_117_ += 8.0 * _t_118_;
_t_115_ = la[k][j][i-2] * _t_117_;
_t_120_ = u_1[k][j-2][i-1];
_t_120_ -= u_1[k][j+2][i-1];
_t_121_ = -u_1[k][j-1][i-1];
_t_121_ += u_1[k][j+1][i-1];
_t_120_ += 8.0 * _t_121_;
_t_119_ = la[k][j][i-1] * _t_120_;
_t_115_ -= 8.0 * _t_119_;
_t_123_ = u_1[k][j-2][i+1];
_t_123_ -= u_1[k][j+2][i+1];
_t_124_ = -u_1[k][j-1][i+1];
_t_124_ += u_1[k][j+1][i+1];
_t_123_ += 8.0 * _t_124_;
_t_122_ = la[k][j][i+1] * _t_123_;
_t_115_ += 8.0 * _t_122_;
_t_125_ = u_1[k][j-2][i+2];
_t_125_ -= u_1[k][j+2][i+2];
_t_126_ = -u_1[k][j-1][i+2];
_t_126_ += u_1[k][j+1][i+2];
_t_125_ += 8.0 * _t_126_;
_t_115_ -= la[k][j][i+2] * _t_125_;
_t_113_ = _t_114_ * _t_115_;
_t_129_ = strx[i] * strz[k];
_t_127_ = _t_129_ * 1.0 / 144.0;
_t_130_ = u_2[k-2][j][i-2];
_t_130_ -= u_2[k+2][j][i-2];
_t_131_ = -u_2[k-1][j][i-2];
_t_131_ += u_2[k+1][j][i-2];
_t_130_ += 8.0 * _t_131_;
_t_128_ = la[k][j][i-2] * _t_130_;
_t_133_ = u_2[k-2][j][i-1];
_t_133_ -= u_2[k+2][j][i-1];
_t_134_ = -u_2[k-1][j][i-1];
_t_134_ += u_2[k+1][j][i-1];
_t_133_ += 8.0 * _t_134_;
_t_132_ = la[k][j][i-1] * _t_133_;
_t_128_ -= 8.0 * _t_132_;
_t_136_ = u_2[k-2][j][i+1];
_t_136_ -= u_2[k+2][j][i+1];
_t_137_ = -u_2[k-1][j][i+1];
_t_137_ += u_2[k+1][j][i+1];
_t_136_ += 8.0 * _t_137_;
_t_135_ = la[k][j][i+1] * _t_136_;
_t_128_ += 8.0 * _t_135_;
_t_138_ = u_2[k-2][j][i+2];
_t_138_ -= u_2[k+2][j][i+2];
_t_139_ = -u_2[k-1][j][i+2];
_t_139_ += u_2[k+1][j][i+2];
_t_138_ += 8.0 * _t_139_;
_t_128_ -= la[k][j][i+2] * _t_138_;
_t_113_ += _t_127_ * _t_128_;
_t_142_ = strx[i] * stry[j];
_t_140_ = _t_142_ * 1.0 / 144.0;
_t_143_ = u_1[k][j-2][i-2];
_t_143_ -= u_1[k][j-2][i+2];
_t_144_ = -u_1[k][j-2][i-1];
_t_144_ += u_1[k][j-2][i+1];
_t_143_ += 8.0 * _t_144_;
_t_141_ = mu[k][j-2][i] * _t_143_;
_t_146_ = u_1[k][j-1][i-2];
_t_146_ -= u_1[k][j-1][i+2];
_t_147_ = -u_1[k][j-1][i-1];
_t_147_ += u_1[k][j-1][i+1];
_t_146_ += 8.0 * _t_147_;
_t_145_ = mu[k][j-1][i] * _t_146_;
_t_141_ -= 8.0 * _t_145_;
_t_149_ = u_1[k][j+1][i-2];
_t_149_ -= u_1[k][j+1][i+2];
_t_150_ = -u_1[k][j+1][i-1];
_t_150_ += u_1[k][j+1][i+1];
_t_149_ += 8.0 * _t_150_;
_t_148_ = mu[k][j+1][i] * _t_149_;
_t_141_ += 8.0 * _t_148_;
_t_151_ = u_1[k][j+2][i-2];
_t_151_ -= u_1[k][j+2][i+2];
_t_152_ = -u_1[k][j+2][i-1];
_t_152_ += u_1[k][j+2][i+1];
_t_151_ += 8.0 * _t_152_;
_t_141_ -= mu[k][j+2][i] * _t_151_;
_t_113_ += _t_140_ * _t_141_;
_t_155_ = strx[i] * strz[k];
_t_153_ = _t_155_ * 1.0 / 144.0;
_t_156_ = u_2[k-2][j][i-2];
_t_156_ -= u_2[k-2][j][i+2];
_t_157_ = -u_2[k-2][j][i-1];
_t_157_ += u_2[k-2][j][i+1];
_t_156_ += 8.0 * _t_157_;
_t_154_ = mu[k-2][j][i] * _t_156_;
_t_159_ = u_2[k-1][j][i-2];
_t_159_ -= u_2[k-1][j][i+2];
_t_160_ = -u_2[k-1][j][i-1];
_t_160_ += u_2[k-1][j][i+1];
_t_159_ += 8.0 * _t_160_;
_t_158_ = mu[k-1][j][i] * _t_159_;
_t_154_ -= 8.0 * _t_158_;
_t_162_ = u_2[k+1][j][i-2];
_t_162_ -= u_2[k+1][j][i+2];
_t_163_ = -u_2[k+1][j][i-1];
_t_163_ += u_2[k+1][j][i+1];
_t_162_ += 8.0 * _t_163_;
_t_161_ = mu[k+1][j][i] * _t_162_;
_t_154_ += 8.0 * _t_161_;
_t_164_ = u_2[k+2][j][i-2];
_t_164_ -= u_2[k+2][j][i+2];
_t_165_ = -u_2[k+2][j][i-1];
_t_165_ += u_2[k+2][j][i+1];
_t_164_ += 8.0 * _t_165_;
_t_154_ -= mu[k+2][j][i] * _t_164_;
_t_113_ += _t_153_ * _t_154_;
r1 += _t_113_;
_t_169_ = strx[i] * stry[j];
_t_167_ = _t_169_ * 1.0 / 144.0;
_t_170_ = u_0[k][j-2][i-2];
_t_170_ -= u_0[k][j+2][i-2];
_t_171_ = -u_0[k][j-1][i-2];
_t_171_ += u_0[k][j+1][i-2];
_t_170_ += 8.0 * _t_171_;
_t_168_ = mu[k][j][i-2] * _t_170_;
_t_173_ = u_0[k][j-2][i-1];
_t_173_ -= u_0[k][j+2][i-1];
_t_174_ = -u_0[k][j-1][i-1];
_t_174_ += u_0[k][j+1][i-1];
_t_173_ += 8.0 * _t_174_;
_t_172_ = mu[k][j][i-1] * _t_173_;
_t_168_ -= 8.0 * _t_172_;
_t_176_ = u_0[k][j-2][i+1];
_t_176_ -= u_0[k][j+2][i+1];
_t_177_ = -u_0[k][j-1][i+1];
_t_177_ += u_0[k][j+1][i+1];
_t_176_ += 8.0 * _t_177_;
_t_175_ = mu[k][j][i+1] * _t_176_;
_t_168_ += 8.0 * _t_175_;
_t_178_ = u_0[k][j-2][i+2];
_t_178_ -= u_0[k][j+2][i+2];
_t_179_ = -u_0[k][j-1][i+2];
_t_179_ += u_0[k][j+1][i+2];
_t_178_ += 8.0 * _t_179_;
_t_168_ -= mu[k][j][i+2] * _t_178_;
_t_166_ = _t_167_ * _t_168_;
_t_182_ = strx[i] * stry[j];
_t_180_ = _t_182_ * 1.0 / 144.0;
_t_183_ = u_0[k][j-2][i-2];
_t_183_ -= u_0[k][j-2][i+2];
_t_184_ = -u_0[k][j-2][i-1];
_t_184_ += u_0[k][j-2][i+1];
_t_183_ += 8.0 * _t_184_;
_t_181_ = la[k][j-2][i] * _t_183_;
_t_186_ = u_0[k][j-1][i-2];
_t_186_ -= u_0[k][j-1][i+2];
_t_187_ = -u_0[k][j-1][i-1];
_t_187_ += u_0[k][j-1][i+1];
_t_186_ += 8.0 * _t_187_;
_t_185_ = la[k][j-1][i] * _t_186_;
_t_181_ -= 8.0 * _t_185_;
_t_189_ = u_0[k][j+1][i-2];
_t_189_ -= u_0[k][j+1][i+2];
_t_190_ = -u_0[k][j+1][i-1];
_t_190_ += u_0[k][j+1][i+1];
_t_189_ += 8.0 * _t_190_;
_t_188_ = la[k][j+1][i] * _t_189_;
_t_181_ += 8.0 * _t_188_;
_t_191_ = u_0[k][j+2][i-2];
_t_191_ -= u_0[k][j+2][i+2];
_t_192_ = -u_0[k][j+2][i-1];
_t_192_ += u_0[k][j+2][i+1];
_t_191_ += 8.0 * _t_192_;
_t_181_ -= la[k][j+2][i] * _t_191_;
_t_166_ += _t_180_ * _t_181_;
_t_195_ = stry[j] * strz[k];
_t_193_ = _t_195_ * 1.0 / 144.0;
_t_196_ = u_2[k-2][j-2][i];
_t_196_ -= u_2[k+2][j-2][i];
_t_197_ = -u_2[k-1][j-2][i];
_t_197_ += u_2[k+1][j-2][i];
_t_196_ += 8.0 * _t_197_;
_t_194_ = la[k][j-2][i] * _t_196_;
_t_199_ = u_2[k-2][j-1][i];
_t_199_ -= u_2[k+2][j-1][i];
_t_200_ = -u_2[k-1][j-1][i];
_t_200_ += u_2[k+1][j-1][i];
_t_199_ += 8.0 * _t_200_;
_t_198_ = la[k][j-1][i] * _t_199_;
_t_194_ -= 8.0 * _t_198_;
_t_202_ = u_2[k-2][j+1][i];
_t_202_ -= u_2[k+2][j+1][i];
_t_203_ = -u_2[k-1][j+1][i];
_t_203_ += u_2[k+1][j+1][i];
_t_202_ += 8.0 * _t_203_;
_t_201_ = la[k][j+1][i] * _t_202_;
_t_194_ += 8.0 * _t_201_;
_t_204_ = u_2[k-2][j+2][i];
_t_204_ -= u_2[k+2][j+2][i];
_t_205_ = -u_2[k-1][j+2][i];
_t_205_ += u_2[k+1][j+2][i];
_t_204_ += 8.0 * _t_205_;
_t_194_ -= la[k][j+2][i] * _t_204_;
_t_166_ += _t_193_ * _t_194_;
_t_208_ = stry[j] * strz[k];
_t_206_ = _t_208_ * 1.0 / 144.0;
_t_209_ = u_2[k-2][j-2][i];
_t_209_ -= u_2[k-2][j+2][i];
_t_210_ = -u_2[k-2][j-1][i];
_t_210_ += u_2[k-2][j+1][i];
_t_209_ += 8.0 * _t_210_;
_t_207_ = mu[k-2][j][i] * _t_209_;
_t_212_ = u_2[k-1][j-2][i];
_t_212_ -= u_2[k-1][j+2][i];
_t_213_ = -u_2[k-1][j-1][i];
_t_213_ += u_2[k-1][j+1][i];
_t_212_ += 8.0 * _t_213_;
_t_211_ = mu[k-1][j][i] * _t_212_;
_t_207_ -= 8.0 * _t_211_;
_t_215_ = u_2[k+1][j-2][i];
_t_215_ -= u_2[k+1][j+2][i];
_t_216_ = -u_2[k+1][j-1][i];
_t_216_ += u_2[k+1][j+1][i];
_t_215_ += 8.0 * _t_216_;
_t_214_ = mu[k+1][j][i] * _t_215_;
_t_207_ += 8.0 * _t_214_;
_t_217_ = u_2[k+2][j-2][i];
_t_217_ -= u_2[k+2][j+2][i];
_t_218_ = -u_2[k+2][j-1][i];
_t_218_ += u_2[k+2][j+1][i];
_t_217_ += 8.0 * _t_218_;
_t_207_ -= mu[k+2][j][i] * _t_217_;
_t_166_ += _t_206_ * _t_207_;
r2 += _t_166_;
uacc_0kc0jc0ic0 = a1 * uacc_0[k][j][i];
uacc_0kc0jc0ic0 += cof * r1;
uacc_0[k][j][i] = uacc_0kc0jc0ic0;
uacc_1kc0jc0ic0 = a1 * uacc_1[k][j][i];
uacc_1kc0jc0ic0 += cof * r2;
uacc_1[k][j][i] = uacc_1kc0jc0ic0;
uacc_2kc0jc0ic0 = a1 * uacc_2[k][j][i];
uacc_2kc0jc0ic0 += cof * r3;
uacc_2[k][j][i] = uacc_2kc0jc0ic0;
	} 
}

extern "C" void host_code (double *h_uacc_0, double *h_uacc_1, double *h_uacc_2, double *h_u_0, double *h_u_1, double *h_u_2, double *h_mu, double *h_la, double *h_strx, double *h_stry, double *h_strz, int N) {
	double *uacc_0;
	cudaMalloc (&uacc_0, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for uacc_0\n");
	cudaMemcpy (uacc_0, h_uacc_0, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *uacc_1;
	cudaMalloc (&uacc_1, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for uacc_1\n");
	cudaMemcpy (uacc_1, h_uacc_1, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *uacc_2;
	cudaMalloc (&uacc_2, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for uacc_2\n");
	cudaMemcpy (uacc_2, h_uacc_2, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *u_0;
	cudaMalloc (&u_0, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for u_0\n");
	cudaMemcpy (u_0, h_u_0, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *u_1;
	cudaMalloc (&u_1, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for u_1\n");
	cudaMemcpy (u_1, h_u_1, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *u_2;
	cudaMalloc (&u_2, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for u_2\n");
	cudaMemcpy (u_2, h_u_2, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *mu;
	cudaMalloc (&mu, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for mu\n");
	cudaMemcpy (mu, h_mu, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *la;
	cudaMalloc (&la, sizeof(double)*N*N*N);
	check_error ("Failed to allocate device memory for la\n");
	cudaMemcpy (la, h_la, sizeof(double)*N*N*N, cudaMemcpyHostToDevice);
	double *strx;
	cudaMalloc (&strx, sizeof(double)*N);
	check_error ("Failed to allocate device memory for strx\n");
	cudaMemcpy (strx, h_strx, sizeof(double)*N, cudaMemcpyHostToDevice);
	double *stry;
	cudaMalloc (&stry, sizeof(double)*N);
	check_error ("Failed to allocate device memory for stry\n");
	cudaMemcpy (stry, h_stry, sizeof(double)*N, cudaMemcpyHostToDevice);
	double *strz;
	cudaMalloc (&strz, sizeof(double)*N);
	check_error ("Failed to allocate device memory for strz\n");
	cudaMemcpy (strz, h_strz, sizeof(double)*N, cudaMemcpyHostToDevice);

	dim3 blockconfig (16, 2, 2);
	dim3 gridconfig (ceil(N, blockconfig.x), ceil(N, blockconfig.y), ceil(N, blockconfig.z));

	sw4 <<<gridconfig, blockconfig>>> (uacc_0, uacc_1, uacc_2, u_0, u_1, u_2, mu, la, strx, stry, strz, N);

	cudaMemcpy (h_uacc_0, uacc_0, sizeof(double)*N*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy (h_uacc_1, uacc_1, sizeof(double)*N*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy (h_uacc_2, uacc_2, sizeof(double)*N*N*N, cudaMemcpyDeviceToHost);

	cudaFree (uacc_0); 
	cudaFree (uacc_1);
	cudaFree (uacc_2);
	cudaFree (u_0);
	cudaFree (u_1);
	cudaFree (u_2);
	cudaFree (mu);
	cudaFree (la);
	cudaFree (strx);
	cudaFree (stry);
	cudaFree (strz);
}