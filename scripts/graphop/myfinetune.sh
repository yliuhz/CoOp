#!/bin/bash

scratch=$1

if [ $1 == 0 ]
then
    echo "Removing old results"
    rm -r output/caltech101/GraphOp
fi

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" ./scripts/graphop/main.sh caltech101 rn50 end 16 16 False