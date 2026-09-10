#!/bin/bash

datadir="/gpfs/wolf/cli120/world-shared/d3m088/cacti/staged_runs/20181204/gefs19/base/les/subset_d4/"
basename="corlasso_cld_"

# files=$(ls ${datadir}${basename}*.nc)
files=("${datadir}${basename}"*.nc)

# echo $files
for ifile in "${files[@]}"; do
    echo ${ifile}
    ncdump -h ${ifile} | grep -i LWP
done