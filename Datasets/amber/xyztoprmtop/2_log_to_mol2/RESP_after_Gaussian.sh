#!/bin/bash
module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4
module load Amber/20
mkdir HF_resp_mol2_files
n=18
for ((j=1;j<=${n};j++))
 do
  vr=$(awk "FNR == $j {print}" ../molecules.dat)
  antechamber -fi gout -fo mol2 -i ../1_run_Gaussian_files/${vr}/${vr}.log -o ${vr}.mol2 -c resp
 mv ${vr}.mol2 ./HF_resp_mol2_files
 done 
rm ANTECHAMBER* ATOMTYPE.INF esout qout QOUT punch
