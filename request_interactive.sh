#!/bin/bash
if [ $NERSC_HOST = cori ]; then
    ACCOUNT=m3443
    CPU_PER_TASK=10
    module load cgpu
else
    ACCOUNT=m2616_g
    CPU_PER_TASK=10
fi

salloc -A $ACCOUNT -C gpu -q interactive --nodes 1 --ntasks-per-node 1 --gpus-per-task 1 --cpus-per-task $CPU_PER_TASK --time 02:00:00 --gpu-bind=none