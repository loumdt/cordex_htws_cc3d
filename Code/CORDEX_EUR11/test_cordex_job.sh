#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --time=6:00:00
#SBATCH --mem=80G

source /home/tmandonnet/.dev_cordex/bin/activate
python3 /home/tmandonnet/CORDEX/test.py