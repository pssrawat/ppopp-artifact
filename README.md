DESCRIPTION

The artifact for the paper "Register Optimizations for Stencils on GPUs" can be downloaded from 
https://github.com/pssrawat/ppopp-artifact.git. The paper is in the main directory (ppopp-18.pdf).

The package contains:
a. the source code for the reordering framework
b. the examples used in the paper in the examples/ directory
c. scripts for code installation and benchmarking




DEPENDENCIES

We tested the framework on ubuntu 16.04 and Red Hat Enterprise Linux Server release 6.7 using a 
Kepler K40c card, with GCC 5.3.0, LLVM 5.0.0, and NVCC 8.0. The following are hardware requirements
for the framework:
1. flex >= 2.6.0 (2.6.0 tested)
2. bison >= 3.0.4 (3.0.4 tested)
3. cmake >= 3.8 (3.8 tested)
4. boost >=1.58 (1.58 tested)
5. GCC version 4 (4.9.2 tested) or 5 (5.3.0 tested)
6. NVCC 8.0
7. LLVM 5.0 (with gpucc) 




STEPS TO INSTALL

1. Set the CUDAHOME variable with 'export CUDAHOME=path-to-cuda'. 
2. Set the CAPABILITY variable to the GPU device's compute capability. For example, we executed 
   'export CAPABILITY=35' for the K40c card we tested the framework on.
 *Some scripts will not run if these two variables are not set*
3. Download and install LLVM. If you cannot get the latest version of LLVM with GPUCC (LLVM 5.0 and above) from
   apt or any other repo, you can download it from http://releases.llvm.org/download.html. The installation 
   steps are in https://llvm.org/docs/GettingStarted.html. 
   I downloaded LLVM into source directory, and created two separate build and install directories. Then, from
   the build directory, I used the following command to configure LLVM:
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/../install ../source/llvm/ -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DGCC_INSTALL_PREFIX=/opt/software/gcc/4.9.2/ -DCMAKE_C_COMPILER=/opt/software/gcc/4.9.2/bin/gcc -DCMAKE_CXX_COMPILER=/opt/software/gcc/4.9.2/bin/g++ -DCMAKE_CXX_LINK_FLAGS="-L/opt/software/gcc/4.9.2/lib64 -Wl,-rpath,/opt/software/gcc/4.9.2/lib64"
    You may need to adjust the paths according to your machine configuration.
4. Simply run 'make all' in the main directory. The makefile will create a 'test' executable.
5. Go to the examples directory, and run the benchmarking script as './run-benchmarks.sh'.
   This will create a file 'output.txt' with all the results. Alternatively, you can go to an independent
   directory, and run 'make', and see the printed results on standard output.




COPYRIGHT

All files in this archive which do not include a prior copyright are by default included in this tool and copyrighted 2017 Ohio State University.




MORE INFORMATION

For more information on how to add a new benchmark, see the docs/ folder or contact me at <rawat.15@osu.edu>
