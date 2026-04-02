#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --time=24:00:00
#SBATCH --mem=32G

source /home/tmandonnet/.gdal_env/bin/activate

for ssp in {1,2,3,4,5}
do
    echo "SSP ${ssp}"
    for year in {2020..2100..10}
    do
        echo "Year $year"
        FILE_IN_TIF="/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_${year}.tif"
        echo "gdal_translate tif to netCDF"
        gdal_translate -of NetCDF "$FILE_IN_TIF" "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_${year}.nc"
        echo "cdo sellonlatbox to reduce memory load"
        cdo sellonlatbox,-55,75,15,80 "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_${year}.nc" "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_${year}_smallbox.nc"
        rm "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_${year}.nc"
        echo "python3 add_time_dim.py"
        python3 /home/tmandonnet/CORDEX/add_time_dim.py "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_${year}_smallbox.nc" $year
        rm "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_${year}_smallbox.nc"

    done
    echo "cdo mergetime"
    cdo mergetime /scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_*_smallbox_time.nc /scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_merged.nc
    rm /scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_*_smallbox_time.nc
    
    echo "cdo inttime"
    cdo inttime,2020-01-01,00:00:00,1year /scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_merged.nc /scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_2020_2100.nc
    rm /scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_merged.nc

    echo "cdo -remapcon Lambert-Conformal"
    cdo -remapcon,/bdd/CORDEX/output/EUR-11/CNRM/CNRM-CERFACS-CNRM-CM5/historical/r1i1p1/CNRM-ALADIN63/v2/day/tasmax/latest/tasmax_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_day_19510101-19551231.nc "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_2020_2100.nc" "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_2020_2100_Lambert_Conformal.nc"
    echo "cdo -remapcon Rotated Latitude-Longitude"
    cdo -remapcon,/data/tmandonnet/CORDEX/sftlf/CORDEX_EUR-11_land_area_fraction_rotated_latitude_longitude.nc "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_2020_2100.nc" "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_2020_2100_rotated_latitude_longitude.nc"
    echo "cdo -remapcon Rotated Pole"
    cdo -remapcon,/bdd/CORDEX/output/EUR-11/IPSL/IPSL-IPSL-CM5A-MR/rcp85/r1i1p1/IPSL-WRF381P/v1/day/tasmax/v20190919/tasmax_EUR-11_IPSL-IPSL-CM5A-MR_rcp85_r1i1p1_IPSL-WRF381P_v1_day_20910101-20951231.nc "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_2020_2100.nc" "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_2020_2100_rotated_pole.nc"
    
    rm "/scratchu/tmandonnet/FPOP/FPOP_SSP${ssp}/FPOP_SSP${ssp}_2020_2100.nc"
    echo "Done"
    done
done