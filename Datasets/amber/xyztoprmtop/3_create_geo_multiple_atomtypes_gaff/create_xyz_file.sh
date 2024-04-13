#!/bin/bash
n=18
for ((j=1;j<=${n};j++))
 do
  vr=$(awk "FNR == $j {print}" ../molecules.dat)
  cd ${vr}
  python ../xplo2xyz.py ${vr}.pdb input.xyz
  sed -i "s/xyz file converted from ${vr}.pdb/${vr}/g" input.xyz
  python ../align.py
  mv output.xyz ${vr}.xyz
  cp ../replace_atomtype.py ./ 
  sed -i "s/NNNN/${vr}/g" replace_atomtype.py
  python replace_atomtype.py
  mv structure.xyz ${vr}.xyz 
  cd ..
 done 
