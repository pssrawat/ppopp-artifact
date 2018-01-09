#!/bin/bash

awk '/#pragma begin/,/#pragma end/' $1 > stencils
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-a.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-b.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-c.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-d.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-e.cu

awk '/#pragma begin/{print $3}' stencils > stencilnames
awk '/unroll/{print $5}' stencils > unrollfactors 

while read -r name
do
uf=`awk 'NR==1' unrollfactors`
sed -i '1d' unrollfactors
awk '/#pragma begin '"$name"'/{flag=1;next} /#pragma end '"$name"'/{flag=0} flag' stencils > $name.idsl
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 0 --distribute-rhs true --split false  
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-a.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 0 --distribute-rhs false --split false  
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-b.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs true --split false 
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-c.cu
sed -i '/#pragma begin '"$name"'/r 'orig_"$name"'.cu' reordered-d.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs false --split false
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-e.cu
done < stencilnames

sed -i '/#pragma begin stencil/d' reordered-a.cu
sed -i '/#pragma end stencil/d' reordered-a.cu
#indent -kr -i8 reordered-a.cu
sed -i '/#pragma begin stencil/d' reordered-b.cu
sed -i '/#pragma end stencil/d' reordered-b.cu
#indent -kr -i8 reordered-b.cu
sed -i '/#pragma begin stencil/d' reordered-c.cu
sed -i '/#pragma end stencil/d' reordered-c.cu
#indent -kr -i8 reordered-c.cu
sed -i '/#pragma begin stencil/d' reordered-d.cu
sed -i '/#pragma end stencil/d' reordered-d.cu
#indent -kr -i8 reordered-d.cu
sed -i '/#pragma begin stencil/d' reordered-e.cu
sed -i '/#pragma end stencil/d' reordered-e.cu
#indent -kr -i8 reordered-e.cu

rm *~
