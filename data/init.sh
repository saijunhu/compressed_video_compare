#!/usr/bin/env bash

inname=$1
outname=$2
start=$3
end=$4
ffmpeg -i ${inname} -vf scale=340:256,setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo ${outname}
#ffmpeg -ss ${start} -i ${inname} -to ${end} -c copy ${outname}