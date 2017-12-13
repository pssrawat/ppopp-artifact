NVCC=nvcc
NVPROF=nvprof
NOPTFLAGS=-O3 -ccbin=g++ -std=c++11 -Xcompiler "-fPIC -fopenmp -O3 -fno-strict-aliasing" --use_fast_math -Xptxas "-dlcm=ca"
NCOMPUTEFLAGS=-gencode arch=compute_$(CAPABILITY),code=sm_$(CAPABILITY) 
CLANG=clang++
LOPTFLAGS=-O3 -ffp-contract=fast --cuda-path=$(CUDAHOME) -L$(CUDAHOME)/lib64 -L$(CUDAHOME)/nvvm -lcudart
LCOMPUTEFLAGS=--cuda-gpu-arch=sm_$(CAPABILITY)

all:
	#./reorder.sh j2d25pt-reg.cu
	$(NVCC) $(NOPTFLAGS) $(NCOMPUTEFLAGS) -maxrregcount=32 common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp j2d25pt-orig.cu -o nvcc-orig
	$(NVCC) $(NOPTFLAGS) $(NCOMPUTEFLAGS) -maxrregcount=48 common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp j2d25pt-unroll.cu -o nvcc-unroll 
	$(NVCC) $(NOPTFLAGS) $(NCOMPUTEFLAGS) -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-a.cu -o nvcc-reorder-a
	$(NVCC) $(NOPTFLAGS) $(NCOMPUTEFLAGS) -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-b.cu -o nvcc-reorder-b
	$(NVCC) $(NOPTFLAGS) $(NCOMPUTEFLAGS) -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-c.cu -o nvcc-reorder-c
	$(NVCC) $(NOPTFLAGS) $(NCOMPUTEFLAGS) -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-d.cu -o nvcc-reorder-d
	$(NVCC) $(NOPTFLAGS) $(NCOMPUTEFLAGS) -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-e.cu -o nvcc-reorder-e
	$(CLANG) $(LOPTFLAGS) $(LCOMPUTEFLAGS) -Xcuda-ptxas -maxrregcount=48 common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp j2d25pt-orig.cu -o llvm-orig
	$(CLANG) $(LOPTFLAGS) $(LCOMPUTEFLAGS) -Xcuda-ptxas -maxrregcount=128 common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp j2d25pt-unroll.cu -o llvm-unroll
	$(CLANG) $(LOPTFLAGS) $(LCOMPUTEFLAGS) -Xcuda-ptxas -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-a.cu -o llvm-reorder-a
	$(CLANG) $(LOPTFLAGS) $(LCOMPUTEFLAGS) -Xcuda-ptxas -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-b.cu -o llvm-reorder-b
	$(CLANG) $(LOPTFLAGS) $(LCOMPUTEFLAGS) -Xcuda-ptxas -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-c.cu -o llvm-reorder-c 
	$(CLANG) $(LOPTFLAGS) $(LCOMPUTEFLAGS) -Xcuda-ptxas -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-d.cu -o llvm-reorder-d 
	$(CLANG) $(LOPTFLAGS) $(LCOMPUTEFLAGS) -Xcuda-ptxas -maxrregcount=64  common/cuda_header.cu j2d25pt.driver.cpp j2d25pt_gold.cpp reordered-e.cu -o llvm-reorder-e 
	$(NVPROF) --print-gpu-trace ./nvcc-orig > /dev/null 2>nvcc-orig-results
	$(NVPROF) --print-gpu-trace ./nvcc-unroll > /dev/null 2>nvcc-unroll-results
	$(NVPROF) --print-gpu-trace ./nvcc-reorder-a > /dev/null 2>nvcc-reorder-results-a
	$(NVPROF) --print-gpu-trace ./nvcc-reorder-b > /dev/null 2>nvcc-reorder-results-b
	$(NVPROF) --print-gpu-trace ./nvcc-reorder-c > /dev/null 2>nvcc-reorder-results-c
	$(NVPROF) --print-gpu-trace ./nvcc-reorder-d > /dev/null 2>nvcc-reorder-results-d
	$(NVPROF) --print-gpu-trace ./nvcc-reorder-e > /dev/null 2>nvcc-reorder-results-e
	$(NVPROF) --print-gpu-trace ./llvm-orig > /dev/null 2>llvm-orig-results
	$(NVPROF) --print-gpu-trace ./llvm-unroll > /dev/null 2>llvm-unroll-results
	$(NVPROF) --print-gpu-trace ./llvm-reorder-a > /dev/null 2>llvm-reorder-results-a
	$(NVPROF) --print-gpu-trace ./llvm-reorder-b > /dev/null 2>llvm-reorder-results-b
	$(NVPROF) --print-gpu-trace ./llvm-reorder-c > /dev/null 2>llvm-reorder-results-c
	$(NVPROF) --print-gpu-trace ./llvm-reorder-d > /dev/null 2>llvm-reorder-results-d
	$(NVPROF) --print-gpu-trace ./llvm-reorder-e > /dev/null 2>llvm-reorder-results-e
	./common/time.awk

clean:
	rm test nvcc-* llvm-* *.idsl stencils stencilnames unrollfactors 2>/dev/null || true 
