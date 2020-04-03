#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --job-name=mqcl
#SBATCH --output=out
#SBATCH --error=log
#SBATCH --mail-type=NONE
#SBATCH --mem=0
 
cd $SLURM_SUBMIT_DIR

module load NiaEnv/2019b intel

export OMP_NUM_THREADS=$(grep -c processor /proc/cpuinfo)

# begins here
make
make clean
mass=2000.0
for (( i=-30;i<=10;i=i+1 ))
do
#    p=$(echo "scale=1;$i/10.0"|bc)
    p=$(echo "sqrt(2.0*${mass}*e(${i}/10.0))"|bc -l)
    sigmap=$(echo "scale=scale(${p});${p}/20.0"|bc)
    cat > input << END_FILE
mass:
${mass}
x0:
-10.0
p0:
${p}
sigma p:
${sigmap}
Left boundary:
-20.0
Right boundary:
20.0
Upper limit of dx:
1.0
Output period:
100.0
Upper limit of dt:
1.0
END_FILE
    ./mqcl >> output 2>>log
    python plot.py
    for f in phase.*
    do
        mv -- "$f" "${i}.${f#phase.}"
    done
    mv -- averages.txt "${i}.log"
    echo "Finished 10.0 * lnE = $i.0"
    echo $(date +"%Y-%m-%d %H:%M:%S.%N")
done
rm t.txt x.txt p.txt
