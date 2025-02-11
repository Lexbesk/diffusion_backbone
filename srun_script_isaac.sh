#!/bin/bash

while true
do
    echo
    echo "#######################################################"
    echo

    srun -A nvr_srl_simpler \
       -p polar,polar3,polar4,grizzly \
       -N 1 \
       --ntasks=1 \
       --gpus 4 \
       --cpus-per-task 24 \
       --mem=80G \
       --unbuffered \
       -t 04:00:00 \
       /bin/bash -c "cd /lustre/fsw/portfolios/nvr/users/ngkanatsios/lbs/analogical_manipulation && bash scripts/train_isaac.sh"

    sleep 1m

done
# polar,polar2,polar3,polar4,grizzly