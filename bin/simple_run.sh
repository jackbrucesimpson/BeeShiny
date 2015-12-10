#!/usr/bin/env bash

START=61
END=118

for i in $(seq $START $END)
do
./beeshiny -t 8 -f 8 -i /Volumes/drive9/camera1/caffeine2/$i.MP4 -o ../results/$i.csv
done
