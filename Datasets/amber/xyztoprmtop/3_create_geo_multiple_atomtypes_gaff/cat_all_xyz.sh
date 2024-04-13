#!/bin/bash
n=18
for ((j=1;j<=${n};j++))
 do
  vr=$(awk "FNR == $j {print}" ../molecules.dat)
  cd ${vr}
  find -type f -name "${vr}.xyz" -exec cat {} >> "../all_molecs.xyz" \;
  cd ..
 done 
