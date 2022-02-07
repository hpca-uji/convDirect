#!/bin/bash

#********************************************************#
#   ********** TEST CONFIGURATION VARIABLES **********   #
#********************************************************#
# [*] TMIN: Minimum Execution Time for each convolution.
#           If TEST activated, TMIN value must be 0.0
TMIN=0.0
#--------------------------------------------------------#
# [*] TEST: Activate for convolution results evaluation.
#           [ T:Enable ], [ F:Disable ]
TEST=T
#--------------------------------------------------------#
# [*] DEBUG: Activate for debug mode. Prints matrix values.
#           [ T:Enable ], [ F:Disable ]
DEBUG=F
#--------------------------------------------------------#
# [*] FORMAT: Set matrix format for the test.
#           [ NHWC ], [ NCHW ], [ BOTH ]
FORMAT=NHWC
#********************************************************#
#********************************************************#
#********************************************************#


echo 
echo " Starting Driver for Direct Convolution..."

CONFIGFILE=$1
OUTPATH="runs"
RUNID=0

if [ ! -f $CONFIGFILE ]; then
    echo "ERROR: The Test configure doesn't exist. Please, enter a valid filename."
    exit -1
fi

if [ ! -d $OUTPATH ]; then
   mkdir $OUTPATH
else
   RUNID=$(ls $OUTPATH | wc -l)
fi

OUTCSV="$OUTPATH/run$RUNID-Logs.csv"

export OMP_NUM_THREADS=1
export OMP_BIND=true

if [[ $CONFIGFILE == *"cnn"* ]]; then
    #TEST FOR DIRECT CONVOLUTION CNN
    ./build/driver_convDirect.x "cnn" \
	$CONFIGFILE $TMIN $TEST $DEBUG $OUTCSV $FORMAT
else
    #TEST FOR DIRECT CONVOLUTION BATCH
    ./build/driver_convDirect.x "batch" \
	$CONFIGFILE $TMIN $TEST $DEBUG $OUTCSV $FORMAT
fi
