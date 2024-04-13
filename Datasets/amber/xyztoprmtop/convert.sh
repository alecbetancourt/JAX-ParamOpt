#!/bin/bash

# Run Ab Inito Calculations
#cd 1_run_Gaussian_files
#./run_calcs.sh
#cd ..

# Generate RESP charges and .mol2 files
cd 2_log_to_mol2
./RESP_after_Gaussian.sh
cd ..

# Convert .mol2 into prmtop
cd 3_create_geo_multiple_atomtypes_gaff
./run_all_mobley_AMBER_charges.sh
./create_xyz_file.sh
./cat_all_xyz.sh

# Create .geo file for input
./xtob << fin
all_molecs.xyz
n
0
fin

rm all_molecs.xyz