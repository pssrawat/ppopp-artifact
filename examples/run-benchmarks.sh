if [[ -z "${CUDAHOME}" ]]; then 
	echo "CUDAHOME unset"
	exit 125
fi
if [[ -z "${CAPABILITY}" ]]; then 
	echo "CAPABILITY unset"
	exit 125
fi

cur=`pwd`
rm -f output.txt
touch output.txt

for dir in $(ls -d */)
do
    echo $'\n\n\n=========================================================='  >> output.txt
    echo $dir >> output.txt
    echo ==========================================================  >> output.txt
    cd ${dir}
    make
    ./common/time.awk >> ${cur}/output.txt
    cd ${cur}
done
