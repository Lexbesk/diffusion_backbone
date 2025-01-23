srun -A nvr_srl_simpler \
  --partition interactive_singlenode \
  --exclusive \
  --gpus 1 \
  --cpus-per-gpu 16 \
  -t 02:00:00 \
  --pty /bin/bash
