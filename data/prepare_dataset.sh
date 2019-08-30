#!/usr/bin/env bash
#step 1:

if [[ "$#" -ne 2 ]]; then
    echo "Usage: ./reencode.sh [input dir] [output dir]"
fi

indir=$1
outdir=$2

mkdir outdir
if [[ ! -d "${outdir}" ]]; then
  echo "${outdir} doesn't exist. Creating it.";
  mkdir -p ${outdir}
fi

for c in $(ls ${indir})
do
	for inname in $(ls ${indir}/${c}/*mp4)
	do
		class_path="$(dirname "$inname")"
		class_name="${class_path##*/}"

		outname="${outdir}/${inname##*/}"
		outname="${outname%.*}.mp4"

		mkdir -p "$(dirname "$outname")"
		ffmpeg -i ${inname} -vf scale=340:256,setsar=1:1 -c:v mpeg4 ${outname}

	done

	for inname in $(ls ${indir}/${c}/*flv)
	do
		class_path="$(dirname "$inname")"
		class_name="${class_path##*/}"

		outname="${outdir}/${inname##*/}"
		outname="${outname%.*}.mp4"

		mkdir -p "$(dirname "$outname")"
		ffmpeg -i ${inname} -vf scale=340:256,setsar=1:1 -c:v mpeg4 ${outname}

	done

done
