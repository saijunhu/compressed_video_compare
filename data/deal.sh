#!/usr/bin/env bash

indir=$1
outdir=$2

mkdir $2
for c in $(ls ${indir})
do
    mkdir "${outdir}/${c}/"
	for inname in $(ls ${indir}/${c}/${c}/)
	do
	    cp "${indir}/${c}/${c}/${inname}" "${outdir}/${c}/"
	done
done
