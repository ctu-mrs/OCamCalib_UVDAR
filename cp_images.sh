#!/bin/bash
if [[ $# == "0" ]]
then
  echo "Provide the camera number!"
  exit
fi

rm i_*.jpg
num=$1
for f in $(ls ~/bag_files/uvdar_calib/${num}/*.jpg)
do
  cp $f ./i_$( basename ${f} )
done
echo $num > current_cam.txt
