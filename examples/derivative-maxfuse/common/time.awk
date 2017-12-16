#!/bin/bash
set -e

echo "-------------------- NVCC ---------------------"

time=`grep -E 'float|double' nvcc-orig-results | awk 'BEGIN {time = 0.0} {time += $2} END {print time}'`
awk -v otime=$time 'BEGIN {print "Original GFlops = " (300*300*300*486/10^6/otime)}'
time0=`grep -E 'float|double' nvcc-reorder-results | awk 'BEGIN {time0 = 0.0} {time0 += $2} END {print time0}'`
timea=`grep -E 'float|double' nvcc-reorder-results-a | awk 'BEGIN {timea = 0.0} {timea += $2} END {print timea}'`
timeb=`grep -E 'float|double' nvcc-reorder-results-b | awk 'BEGIN {timeb = 0.0} {timeb += $2} END {print timeb}'`
timec=`grep -E 'float|double' nvcc-reorder-results-c | awk 'BEGIN {timec = 0.0} {timec += $2} END {print timec}'`
timed=`grep -E 'float|double' nvcc-reorder-results-d | awk 'BEGIN {timed = 0.0} {timed += $2} END {print timed}'`
timeg=`grep -E 'float|double' nvcc-reorder-results-g | awk 'BEGIN {timeg = 0.0} {timeg += $2} END {print timeg}'`
timeh=`grep -E 'float|double' nvcc-reorder-results-h | awk 'BEGIN {timeh = 0.0} {timeh += $2} END {print timeh}'`
min0=`awk -v atime=$time0 -v btime=$timea 'BEGIN {print (atime<btime?atime:btime)}'`
min1=`awk -v atime=$min0 -v btime=$timeb 'BEGIN {print (atime<btime?atime:btime)}'`
min2=`awk -v min=$min1 -v ctime=$timec 'BEGIN {print (ctime<min?ctime:min)}'`
min3=`awk -v min=$min2 -v dtime=$timed 'BEGIN {print (dtime<min?dtime:min)}'`
min6=`awk -v min=$min3 -v gtime=$timeg 'BEGIN {print (gtime<min?gtime:min)}'`
awk -v min=$min6 -v htime=$timeh 'BEGIN {print "Reordered GFlops = " (300*300*300*486/10^6/(min<htime?min:htime))}'

echo "-------------------- LLVM ---------------------"

time=`grep -E 'float|double' llvm-orig-results | awk 'BEGIN {time = 0.0} {time += $2} END {print time}'`
awk -v otime=$time 'BEGIN {print "Original GFlops = " (300*300*300*486/10^6/otime)}'
time0=`grep -E 'float|double' llvm-reorder-results | awk 'BEGIN {time0 = 0.0} {time0 += $2} END {print time0}'`
timea=`grep -E 'float|double' llvm-reorder-results-a | awk 'BEGIN {timea = 0.0} {timea += $2} END {print timea}'`
timeb=`grep -E 'float|double' llvm-reorder-results-b | awk 'BEGIN {timeb = 0.0} {timeb += $2} END {print timeb}'`
timec=`grep -E 'float|double' llvm-reorder-results-c | awk 'BEGIN {timec = 0.0} {timec += $2} END {print timec}'`
timed=`grep -E 'float|double' llvm-reorder-results-d | awk 'BEGIN {timed = 0.0} {timed += $2} END {print timed}'`
timeg=`grep -E 'float|double' llvm-reorder-results-g | awk 'BEGIN {timeg = 0.0} {timeg += $2} END {print timeg}'`
timeh=`grep -E 'float|double' llvm-reorder-results-h | awk 'BEGIN {timeh = 0.0} {timeh += $2} END {print timeh}'`
min0=`awk -v atime=$time0 -v btime=$timea 'BEGIN {print (atime<btime?atime:btime)}'`
min1=`awk -v atime=$min0 -v btime=$timeb 'BEGIN {print (atime<btime?atime:btime)}'`
min2=`awk -v min=$min1 -v ctime=$timec 'BEGIN {print (ctime<min?ctime:min)}'`
min3=`awk -v min=$min2 -v dtime=$timed 'BEGIN {print (dtime<min?dtime:min)}'`
min6=`awk -v min=$min3 -v gtime=$timeg 'BEGIN {print (gtime<min?gtime:min)}'`
awk -v min=$min6 -v htime=$timeh 'BEGIN {print "Reordered GFlops = " (300*300*300*486/10^6/(min<htime?min:htime))}'
