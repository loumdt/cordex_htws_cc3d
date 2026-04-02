#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --time=24:00:00
#SBATCH --mem=80G

source /home/tmandonnet/.dev_cordex/bin/activate
python3 /home/tmandonnet/CORDEX/run_all.py $1 $2 $3 $4 $5 $6