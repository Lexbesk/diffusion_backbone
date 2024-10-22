#!/bin/bash
#conda activate 3d_diffuser_actor
while true
do
    echo
    echo "#######################################################"
    echo

    srun -A nvr_srl_simpler \
       -p interactive \
       -N 1 \
       --gpus 6 \
       --cpus-per-task 32 \
       --unbuffered \
       -t 02:00:00 \
       /bin/bash -c "cd /lustre/fsw/portfolios/nvr/users/ngkanatsios/lbs/analogical_manipulation && bash scripts/train_keypose_gnfactor.sh"

    sleep 1m

done
# polar,polar2,polar3,polar4,grizzly