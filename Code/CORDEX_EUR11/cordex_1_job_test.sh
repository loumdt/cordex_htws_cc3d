#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --time=6:00:00
#SBATCH --mem=64G

gcm="CNRM-CERFACS-CNRM-CM5"
rcm="GERICS-REMO2015"
scenario="rcp85"
version="v1"
historical_path="/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/historical/r1i1p1/GERICS-REMO2015/v1/day/tas/v20170208"
rcp_path="/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/rcp85/r1i1p1/GERICS-REMO2015/v1/day/tas/v20170208"

source /home/tmandonnet/.dev_cordex/bin/activate
python3 /home/tmandonnet/CORDEX/run_all_copy.py $gcm $rcm $scenario $version $historical_path $rcp_path
