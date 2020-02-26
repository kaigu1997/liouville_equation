#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=6:00:00
#SBATCH --job-name=mqcl
#SBATCH --output=out
#SBATCH --error=log
#SBATCH --mail-type=NONE
#SBATCH --mem=0
 
cd $SLURM_SUBMIT_DIR

module load intel/2018.3 valgrind

make
./mqcl
