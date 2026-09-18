#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --time=12:00:00
#SBATCH --mem=8G

source /home/tmandonnet/.dev_cordex3_12/bin/activate
python3 /home/tmandonnet/CORDEX/run_all_step1_BC.py $1 $2 $3 $4 $5 $6
