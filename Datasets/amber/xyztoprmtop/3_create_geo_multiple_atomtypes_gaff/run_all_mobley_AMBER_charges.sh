#!/bin/bash
#n=$(wc -l < molecules.dat)
module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4
module load Amber/20
n=18
for ((j=1;j<=${n};j++))
 do
  vr=$(awk "FNR == $j {print}" ../molecules.dat)
  mkdir ${vr}
  cd ${vr}
  cp  ../tleap.in ./
  cp  ../parmed_print_charge.in ./
  cp ../../2_log_to_mol2/HF_resp_mol2_files/${vr}.mol2 ./
  mv ${vr}.mol2 CH.mol2 
  parmchk2 -i CH.mol2 -f mol2 -o CH.frcmod -s gaff
  tleap -f tleap.in
  ambpdb -p prmtop -c inpcrd > initial.pdb
  mv initial.pdb ${vr}.pdb
  parmed prmtop parmed_print_charge.in > atoms.dat
  sed -i '1,/ATOM    RES/d' atoms.dat
  sed -i "$(( $(wc -l <atoms.dat)-2+1 )),$ d" atoms.dat
  awk '{ print $10 }' atoms.dat > charges.dat
  awk '{ print $5 }' atoms.dat > atomtypes.dat
  mv charges.dat ${vr}_charges.dat
  mv atomtypes.dat ${vr}_atomtypes.dat
  cp prmtop ../${vr}.prmtop
  cd ..
 done 
