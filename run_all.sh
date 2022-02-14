#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <'Plot title'>"
    exit
fi

plot_title=$1

#Prepare path for results
OUT_BATCH=performance_results
PLOTS=plots

if [ -d $OUT_BATCH ]; then
   rm -rf $OUT_BATCH
fi

mkdir $OUT_BATCH

if [ -d $PLOTS ]; then
   rm -rf $PLOTS
fi

mkdir $PLOTS

#./do_plots.py $OUT_BATCH/googlelenet-imagenet-t1.csv "$plot_title" plots
#exit

#Test Batch:
Algorithms=("-DIM2COL" "-DCONVGEMM" "-DBLOCKED_SHALOM" "-DBLOCKED_BLIS" "-DBLOCKED_TZEMENG")
Blis_Kernels=("-DMK_8x12" "-DMK_4x20" "-DMK_BLIS")

#Construct compile instrucctions
all_cmds=()
for alg in ${Algorithms[@]}; do    
    _cmd="make"
    if [ $alg == "-DBLOCKED_SHALOM" ]; then
	cmd="$_cmd ALGORITHM=$alg MKERNEL=-DMK_7x12_NPA_U4"
	all_cmds+=("$cmd")
    elif [ $alg == "-DBLOCKED_TZEMENG" ]; then
	cmd="$_cmd ALGORITHM=$alg MKERNEL=-DMK_7x12_U4"
	all_cmds+=("$cmd")
    elif [ $alg == "-DBLOCKED_BLIS" ]; then
	_cmd="$_cmd ALGORITHM=$alg"
	for ker in ${Blis_Kernels[@]}; do
	    cmd="$_cmd MKERNEL=$ker"
	    all_cmds+=("$cmd")
	done	
    else
	cmd="$_cmd ALGORITHM=$alg"
	all_cmds+=("$cmd")
    fi
done

for te in $(ls test/cnn/)
do
    #Clean runs path to store new results
    if [ ! -z "$(ls -A ./runs/)" ]; then
	rm ./runs/*
    fi

    #Run tests batch
    for ((i = 0; i < ${#all_cmds[@]}; i++))
    do
	cmd="${all_cmds[$i]}"
	if [ ! -z "$(ls -A ./build/)" ]; then
	    make clean
	fi
	$cmd
	./directConvolution_Test.sh test/cnn/$te
    done
    
    #Preparing output format
    tail -n +2 runs/run0-Logs.csv | cut -f1  -d';' > $OUT_BATCH/.layer
    tail -n +2 runs/run0-Logs.csv | cut -f14 -d';' > $OUT_BATCH/.im2col
    tail -n +2 runs/run1-Logs.csv | cut -f14 -d';' > $OUT_BATCH/.convgemm
    tail -n +2 runs/run2-Logs.csv | cut -f14 -d';' > $OUT_BATCH/.shalom
    tail -n +2 runs/run3-Logs.csv | cut -f14 -d';' > $OUT_BATCH/.blis_8_12
    tail -n +2 runs/run4-Logs.csv | cut -f14 -d';' > $OUT_BATCH/.blis_4_20
    tail -n +2 runs/run5-Logs.csv | cut -f14 -d';' > $OUT_BATCH/.blis_blis
    tail -n +2 runs/run6-Logs.csv | cut -f14 -d';' > $OUT_BATCH/.tzemeng

    #Merge columns into final file
    echo "#Layer;Im2col;Convgemm;Shalom;Blis_mk-8x12;Blis_mk-4x20;Blis_mk-blis;Tze-meng" > $OUT_BATCH/$te.csv
    paste -d ";" $OUT_BATCH/.layer $OUT_BATCH/.im2col $OUT_BATCH/.convgemm \
	  $OUT_BATCH/.shalom $OUT_BATCH/.blis_8_12 $OUT_BATCH/.blis_4_20 \
	  $OUT_BATCH/.blis_blis $OUT_BATCH/.tzemeng >> $OUT_BATCH/$te.csv

    rm $OUT_BATCH/.layer $OUT_BATCH/.im2col $OUT_BATCH/.convgemm $OUT_BATCH/.shalom \
       $OUT_BATCH/.blis_8_12 $OUT_BATCH/.blis_4_20 $OUT_BATCH/.blis_blis $OUT_BATCH/.tzemeng
    

    if [ -d plots/$te ]; then
	rm -rf plots/$te
    fi    
    mkdir plots/$te
    
    ./do_plots.py $OUT_BATCH/$te.csv "$te $plot_title" plots/$te
    
done

echo
echo "NOTE: FINAL RESULTS ARE STORED IN THE FOLDER: '$OUT_BATCH'!"
echo
