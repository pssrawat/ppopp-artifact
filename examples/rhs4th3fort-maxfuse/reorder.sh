#!/bin/bash

awk '/#pragma begin/,/#pragma end/' $1 > stencils
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-a.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-b.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-e.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-f.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-g.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-h.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-i.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-j.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-k.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > reordered-l.cu

awk '/#pragma begin/{print $3}' stencils > stencilnames
awk '/unroll/{print $5}' stencils > unrollfactors 

while read -r name
do
uf=`awk 'NR==1' unrollfactors`
sed -i '1d' unrollfactors
awk '/#pragma begin '"$name"'/{flag=1;next} /#pragma end '"$name"'/{flag=0} flag' stencils > $name.idsl
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 0 --distribute-rhs true --topo-sort false --split false  
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-a.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 0 --distribute-rhs true --topo-sort true --split false  
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-b.cu

../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs true --topo-sort false --split false 
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-e.cu
sed -i '/#pragma begin '"$name"'/r orig_'"$name"'.cu' reordered-f.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs true --topo-sort true --split false 
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-g.cu
sed -i '/#pragma begin '"$name"'/r orig_'"$name"'.cu' reordered-h.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs false --topo-sort false --split false
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-i.cu
sed -i '/#pragma begin '"$name"'/r orig_'"$name"'.cu' reordered-j.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs false --topo-sort true --split false
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' reordered-k.cu
sed -i '/#pragma begin '"$name"'/r orig_'"$name"'.cu' reordered-l.cu

done < stencilnames

sed -i '/#pragma begin stencil/d' reordered-a.cu
sed -i '/#pragma end stencil/d' reordered-a.cu
#indent -kr -i8 reordered-a.cu
sed -i '/#pragma begin stencil/d' reordered-b.cu
sed -i '/#pragma end stencil/d' reordered-b.cu
#indent -kr -i8 reordered-b.cu
sed -i '/#pragma begin stencil/d' reordered-e.cu
sed -i '/#pragma end stencil/d' reordered-e.cu
#indent -kr -i8 reordered-e.cu
sed -i '/#pragma begin stencil/d' reordered-f.cu
sed -i '/#pragma end stencil/d' reordered-f.cu
#indent -kr -i8 reordered-f.cu
sed -i '/#pragma begin stencil/d' reordered-g.cu
sed -i '/#pragma end stencil/d' reordered-g.cu
#indent -kr -i8 reordered-g.cu
sed -i '/#pragma begin stencil/d' reordered-h.cu
sed -i '/#pragma end stencil/d' reordered-h.cu
#indent -kr -i8 reordered-h.cu
sed -i '/#pragma begin stencil/d' reordered-i.cu
sed -i '/#pragma end stencil/d' reordered-i.cu
#indent -kr -i8 reordered-i.cu
sed -i '/#pragma begin stencil/d' reordered-j.cu
sed -i '/#pragma end stencil/d' reordered-j.cu
#indent -kr -i8 reordered-j.cu
sed -i '/#pragma begin stencil/d' reordered-k.cu
sed -i '/#pragma end stencil/d' reordered-k.cu
#indent -kr -i8 reordered-k.cu
sed -i '/#pragma begin stencil/d' reordered-l.cu
sed -i '/#pragma end stencil/d' reordered-l.cu
#indent -kr -i8 reordered-l.cu
#rm *~
