#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --time=6:00:00
#SBATCH --mem=32G

source /home/tmandonnet/.gdal_env/bin/activate

for year in {1975..2030..5}
do
    echo "Year $year"
    FILE_IN_TIF="/scratchu/tmandonnet/GHS-POP/GHS_POP_E${year}_GLOBE_R2023A_4326_30ss_V1_0.tif"
    echo "gdal_translate tif to netCDF"
    gdal_translate -of NetCDF "$FILE_IN_TIF" "/scratchu/tmandonnet/GHS-POP/GHS_POP_${year}.nc"
    echo "cdo sellonlatbox to reduce memory load"
    cdo sellonlatbox,-55,75,15,80 "/scratchu/tmandonnet/GHS-POP/GHS_POP_${year}.nc" "/scratchu/tmandonnet/GHS-POP/GHS_POP_${year}_smallbox.nc"
    rm "/scratchu/tmandonnet/GHS-POP/GHS_POP_${year}.nc"
    echo "python3 add_time_dim.py"
    python3 /home/tmandonnet/CORDEX/add_time_dim.py "/scratchu/tmandonnet/GHS-POP/GHS_POP_${year}_smallbox.nc" $year
    rm "/scratchu/tmandonnet/GHS-POP/GHS_POP_${year}_smallbox.nc"
done
echo "cdo mergetime"
cdo mergetime /scratchu/tmandonnet/GHS-POP/GHS_POP_*_smallbox_time.nc /scratchu/tmandonnet/GHS-POP/GHS_POP_merged.nc
rm /scratchu/tmandonnet/GHS-POP/GHS_POP_*_smallbox_time.nc

echo "cdo inttime"
cdo inttime,1975-01-01,00:00:00,1year /scratchu/tmandonnet/GHS-POP/GHS_POP_merged.nc /scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030.nc
rm /scratchu/tmandonnet/GHS-POP/GHS_POP_merged.nc

echo "cdo -remapsum Lambert-Conformal"
cdo -remapsum,/bdd/CORDEX/output/EUR-11/CNRM/CNRM-CERFACS-CNRM-CM5/historical/r1i1p1/CNRM-ALADIN63/v2/day/tasmax/latest/tasmax_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_day_19510101-19551231.nc "/scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030.nc" "/scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030_Lambert_Conformal.nc"
echo "cdo -remapcon Rotated Latitude-Longitude from Lambert-Conformal because remapsum does not work on curvilinear"
cdo -remapcon,/data/tmandonnet/CORDEX/sftlf/CORDEX_EUR-11_land_area_fraction_rotated_latitude_longitude.nc "/scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030_Lambert_Conformal.nc" "/scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030_rotated_latitude_longitude.nc"
echo "cdo -remapcon Rotated Pole from Lambert-Conformal because remapsum does not work on curvilinear"
cdo -remapcon,/bdd/CORDEX/output/EUR-11/IPSL/IPSL-IPSL-CM5A-MR/rcp85/r1i1p1/IPSL-WRF381P/v1/day/tasmax/v20190919/tasmax_EUR-11_IPSL-IPSL-CM5A-MR_rcp85_r1i1p1_IPSL-WRF381P_v1_day_20910101-20951231.nc "/scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030_Lambert_Conformal.nc" "/scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030_rotated_pole.nc"
echo "cdo -remapsum ERA5"
cdo -remapsum,/data/tmandonnet/CORDEX/cellarea/gridarea_CORDEX_EUR11_ERA5.nc "/scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030.nc" "/scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030_ERA5.nc"


rm "/scratchu/tmandonnet/GHS-POP/GHS_POP_1975_2030.nc"
echo "Done"

