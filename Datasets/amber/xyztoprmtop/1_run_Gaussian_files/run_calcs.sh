n=18
for ((j=1;j<=${n};j++))
 do
  vr=$(awk "FNR == $j {print}" ../molecules.dat)
  mkdir ${vr}
  cd ${vr}
  cp ../sub.sh ./
  cp ../generate_gaussian_input.sh ./
  cp ../../2_pdb_to_xyz/xyz_files/${vr}.xyz ./
  sed -i '1,2d' ${vr}.xyz
  echo >> "${vr}.xyz"
  sed -i "s/JJJJ/${vr}/g" sub.sh
  sed -i "s/FFFF/${vr}/g" generate_gaussian_input.sh
  chmod +x generate_gaussian_input.sh
  ./generate_gaussian_input.sh
  sbatch sub.sh
  rm generate_gaussian_input.sh
  cd ..
 done
