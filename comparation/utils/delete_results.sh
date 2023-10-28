#!/bin/bash 

if [ $# -eq 3 ]
then
    EXP_PATH=$1
    NET=$2
    ALGORITHM=$3

    echo $EXP_PATH/${NET}/${ALGORITHM}/results
    rm -rf $EXP_PATH/${NET}/${ALGORITHM}/results

else
    echo $#
    echo "[Error] delete_results: You must put delete_results"
fi
