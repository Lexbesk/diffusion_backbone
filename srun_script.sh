#!/bin/bash
#conda activate 3d_diffuser_actor
while true
do
    echo
    echo "#######################################################"
    echo

    srun \
       -p debug \
       --nodes 1 \
       --ntasks-per-node 1 \
       --gres gpu:2 \
       --cpus-per-task 12 \
       --mem 48G \
       --unbuffered \
       --time 10:00:00 \
       --nodelist babel-14-9 \
       /bin/bash -c "cd /home/ngkanats/repos/lbs/analogical_manipulation && rsync -avP /data/user_data/ngkanats/GNFactor_zarr /scratch && bash debug.sh"

    sleep 1m

done
# srun --partition=debug --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=12 --mem=48G --time=12:00:00 --pty bash
