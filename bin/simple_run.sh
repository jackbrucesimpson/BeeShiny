#!/usr/bin/env bash

START=1
END=140

for i in $(seq $START $END)
do
./beeshiny -t 8 -f 8 -i /Volumes/drive12/$i.MP4 -o ~/Desktop/results/$i.csv
done
