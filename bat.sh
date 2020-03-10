#!/bin/bash
make
make clean
mass=2000.0
for (( i=-40;i<=10;i=i+1 ))
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
Total time of evolution:
20000.0
Output period:
100.0
Upper limit of dt:
1.0
END_FILE
    ./mqcl >> output 2>>log
    python plot.py
    for f in psi.*
    do
        mv -- "$f" "${i}.${f#psi.}"
    done
    echo "Finished 10.0 * lnE = $i.0"
    echo $(date +"%Y-%m-%d %H:%M:%S.%N")
done
rm t.txt x.txt p.txt
