#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=mqcl
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.log
#SBATCH --mail-type=NONE
#SBATCH --mem=0
 
cd $SLURM_SUBMIT_DIR

module load NiaEnv/2019b intel intelpython3

export OMP_NUM_THREADS=$(grep -c processor /proc/cpuinfo)

# begins here
make
make clean
sh bat.sh