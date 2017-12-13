#!/bin/bash

awk '/#pragma begin/,/#pragma end/' $1 > stencils
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-a.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-b.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-c.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-d.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-e.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-f.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-g.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-h.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-i.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-j.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-k.cu
sed '/#pragma begin/,/#pragma end/{//!d}' $1 > max-reordered-l.cu

awk '/#pragma begin/{print $3}' stencils > stencilnames
awk '/unroll/{print $5}' stencils > unrollfactors 

while read -r name
do
uf=`awk 'NR==1' unrollfactors`
sed -i '1d' unrollfactors
awk '/#pragma begin '"$name"'/{flag=1;next} /#pragma end '"$name"'/{flag=0} flag' stencils > $name.idsl
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 0 --distribute-rhs true --topo-sort false --split false  
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' max-reordered-a.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 0 --distribute-rhs true --topo-sort true --split false  
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' max-reordered-b.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 0 --distribute-rhs false --topo-sort false --split false  
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' max-reordered-c.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 0 --distribute-rhs false --topo-sort true --split false  
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' max-reordered-d.cu

../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs true --topo-sort false --split false 
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' max-reordered-e.cu
sed -i '/#pragma begin '"$name"'/r orig_'"$name"'.cu' max-reordered-f.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs true --topo-sort true --split false 
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' max-reordered-g.cu
sed -i '/#pragma begin '"$name"'/r orig_'"$name"'.cu' max-reordered-h.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs false --topo-sort false --split false
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' max-reordered-i.cu
sed -i '/#pragma begin '"$name"'/r orig_'"$name"'.cu' max-reordered-j.cu
../../test $name.idsl --out-file $name.cu --unroll $uf --heuristic 1 --distribute-rhs false --topo-sort true --split false
sed -i '/#pragma begin '"$name"'/r '"$name"'.cu' max-reordered-k.cu
sed -i '/#pragma begin '"$name"'/r orig_'"$name"'.cu' max-reordered-l.cu

done < stencilnames

sed -i '/#pragma begin stencil/d' max-reordered-a.cu
sed -i '/#pragma end stencil/d' max-reordered-a.cu
#indent -kr -i8 max-reordered-a.cu
sed -i '/#pragma begin stencil/d' max-reordered-b.cu
sed -i '/#pragma end stencil/d' max-reordered-b.cu
#indent -kr -i8 max-reordered-b.cu
sed -i '/#pragma begin stencil/d' max-reordered-c.cu
sed -i '/#pragma end stencil/d' max-reordered-c.cu
#indent -kr -i8 max-reordered-c.cu
sed -i '/#pragma begin stencil/d' max-reordered-d.cu
sed -i '/#pragma end stencil/d' max-reordered-d.cu
#indent -kr -i8 max-reordered-d.cu
sed -i '/#pragma begin stencil/d' max-reordered-e.cu
sed -i '/#pragma end stencil/d' max-reordered-e.cu
#indent -kr -i8 max-reordered-e.cu
sed -i '/#pragma begin stencil/d' max-reordered-f.cu
sed -i '/#pragma end stencil/d' max-reordered-f.cu
#indent -kr -i8 max-reordered-f.cu
sed -i '/#pragma begin stencil/d' max-reordered-g.cu
sed -i '/#pragma end stencil/d' max-reordered-g.cu
#indent -kr -i8 max-reordered-g.cu
sed -i '/#pragma begin stencil/d' max-reordered-h.cu
sed -i '/#pragma end stencil/d' max-reordered-h.cu
#indent -kr -i8 max-reordered-h.cu
sed -i '/#pragma begin stencil/d' max-reordered-i.cu
sed -i '/#pragma end stencil/d' max-reordered-i.cu
#indent -kr -i8 max-reordered-i.cu
sed -i '/#pragma begin stencil/d' max-reordered-j.cu
sed -i '/#pragma end stencil/d' max-reordered-j.cu
#indent -kr -i8 max-reordered-j.cu
sed -i '/#pragma begin stencil/d' max-reordered-k.cu
sed -i '/#pragma end stencil/d' max-reordered-k.cu
#indent -kr -i8 max-reordered-k.cu
sed -i '/#pragma begin stencil/d' max-reordered-l.cu
sed -i '/#pragma end stencil/d' max-reordered-l.cu
#indent -kr -i8 max-reordered-l.cu
#rm *~
