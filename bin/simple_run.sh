#!/usr/bin/env bash

START=29
END=30

for i in $(seq $START $END)
do
./beeshiny -t 8 -f 8 -i /Volumes/JSIMPSON/e3/$i.MP4 -o ../results/$i.csv
done
