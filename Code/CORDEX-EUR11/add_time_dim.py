import sys
import xarray as xr
import pandas as pd

input_file = sys.argv[1]
year = sys.argv[2]

ds = xr.open_dataset(input_file,engine='netcdf4')
ds = ds.assign_coords({'time':pd.to_datetime(year)}).expand_dims(dim={'time':1},axis=None)

output_file = input_file.replace(".nc","_time.nc")
ds.to_netcdf(output_file)