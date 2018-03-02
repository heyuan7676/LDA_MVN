#!/bin/bash -l
#SBATCH --time 100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --no-requeue

# =============================

module load python
python VI_LDA_MVN.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"
