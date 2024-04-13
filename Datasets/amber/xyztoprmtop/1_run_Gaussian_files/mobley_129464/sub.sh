#!/bin/bash --login
# Written by Madu Manathunga 10/18/2019

########## Resource Request ##########
#SBATCH --job-name  mobley_129464            # you can give your job a name for easier identification (same as -J)
#SBATCH --time=03:59:00                 # limit of wall clock time - how long the job will run (same as -t)
#SBATCH -A hmakmm
#SBATCH --nodes=1                       # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                      # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=4               # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=1G                # memory required per allocated CPU (or core) - amount of memory (in bytes)

######### Submission of g16 job #############
scontrol show job $SLURM_JOB_ID
module load  Gaussian/g16
mkdir -p $SCRATCH/g16.${SLURM_JOB_ID}
cp $SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}.com $SCRATCH/g16.${SLURM_JOB_ID}/
cd $SCRATCH/g16.$SLURM_JOB_ID/
srun g16 ${SLURM_JOB_NAME}.com >${SLURM_JOB_NAME}.run.log
cp *  $SLURM_SUBMIT_DIR/
cd $SLURM_SUBMIT_DIR
