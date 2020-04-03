#!/bin/bash
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
0.1
Output period:
50.0
Upper limit of dt:
0.25
END_FILE
    ./mqcl >> output 2>>log
    python plot.py
    for f in phase.*
    do
        mv -- "$f" "${i}.${f#phase.}"
    done
    mv -- "averages.txt" "${i}.log"
    echo "Finished 10.0 * lnE = $i.0"
    echo $(date +"%Y-%m-%d %H:%M:%S.%N")
done
rm t.txt x.txt p.txt
