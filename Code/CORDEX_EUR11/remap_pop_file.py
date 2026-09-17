import sys
from os.path import join
import xarray as xr
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from tqdm import tqdm

if __name__=="__main__":
    print("WARNING: Longitude can either be [0,360] or [-180,180]. This script ensures the longitude from input and target are compatible only by checking if the maximum of longitude is higher than 180. Make sure this condition is sufficient in your case.")

    target_grid_file = sys.argv[1] 
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    da_input = xr.open_dataset(input_file,engine='netcdf4').Band1

    ds_output = xr.open_dataset(target_grid_file,engine='netcdf4')
    da_output = ds_output.Band1
    da_output.data = np.zeros(np.shape(da_output.data))

    if 'x' in da_output.dims and 'y' in da_output.dims:
        lon_name = 'x'
        lat_name = 'y'
    elif 'rlat' in da_output.dims and 'rlon' in da_output.dims:
        lon_name = 'rlon'
        lat_name = 'rlat'
    elif 'lat' in da_output.dims and 'lon' in da_output.dims:
        lon_name = 'lon'
        lat_name = 'lat'
    elif 'latitude' in da_output.dims and 'longitude' in da_output.dims:
        lon_name = 'longitude'
        lat_name = 'latitude'
    else:
        raise ValueError(f"Dimensions are not as expected. Should be either ('latitude','longitude'), ('lat','lon'), ('rlat','lon'), or ('x','y'). Dimensions are: {da_input.dims}")
    
    if 'x' in da_input.dims and 'y' in da_input.dims:
        lon_name_input = 'x'
        lat_name_input = 'y'
    elif 'rlat' in da_input.dims and 'rlon' in da_input.dims:
        lon_name_input = 'rlon'
        lat_name_input = 'rlat'
    elif 'lat' in da_input.dims and 'lon' in da_input.dims:
        lon_name_input = 'lon'
        lat_name_input = 'lat'
    elif 'latitude' in da_input.dims and 'longitude' in da_input.dims:
        lon_name_input = 'longitude'
        lat_name_input = 'latitude'
    else:
        raise ValueError(f"Spatial dimensions are not as expected. Should be either ('latitude','longitude'), ('lat','lon'), ('rlat','lon'), or ('x','y'). Dimensions are: {da_input.dims}")

    output_lat = getattr(da_output,lat_name)
    output_lon = getattr(da_output,lon_name)

    # Check if longitude coordinates are -180 -> 180 or 0 -> 360 to have proper comparison with polygon
    translate_input_lon = np.max(da_input.lon.data)>180
    translate_output_lon = np.max(da_output.lon.data)>180

    
    for i_lat in tqdm(range(len(output_lat))):
        for j_lon in tqdm(range(len(output_lon))):
            # Make sure the longitudes used for polygon are -180 -> 180 and not 0 -> 360
            lon_bounds = np.array([val-180*translate_output_lon for val in ds_output.lon_bnds.data[i_lat,j_lon,:]])
            lat_bounds = ds_output.lat_bnds.data[i_lat,j_lon,:]            

            # Create polygon from bounds
            lons_lats_vect = np.column_stack((lon_bounds, lat_bounds)) # Reshape coordinates
            poly = Polygon(lons_lats_vect) # Create polygon

            # Select input points that are within polygon
            # If input longitude is 0 -> 360, add 180 to poly.bounds[0] and poly.bounds[2] to have proper comparison
            da_input_bnds = da_input.where((da_input.lat>=poly.bounds[1])&(da_input.lat<=poly.bounds[3])&(da_input.lon>=poly.bounds[0]+180*translate_input_lon)&(da_input.lon<=poly.bounds[2]+180*translate_input_lon),drop=True)
            mask = np.zeros(np.shape(da_input_bnds.isel(time=0).data))
            
            input_lat = getattr(da_input_bnds,lat_name_input)
            input_lon = getattr(da_input_bnds,lon_name_input)

            #print(da_input_bnds)
            #print(da_input_bnds.lat.data)
            #print(da_input_bnds.lon.data)

            for i_input_lat in range(len(input_lat)):
                for j_input_lon in range(len(input_lon)):
                    #print(i_input_lat,j_input_lon)
                    try:
                        target_lat = da_input_bnds.lat.data[i_input_lat,j_input_lon] - 180*translate_input_lon
                        target_lon = da_input_bnds.lon.data[i_input_lat,j_input_lon] - 180*translate_input_lon
                    except:
                        target_lat = da_input_bnds.lat.data[i_input_lat] - 180*translate_input_lon
                        target_lon = da_input_bnds.lon.data[j_input_lon] - 180*translate_input_lon
                    mask[i_input_lat,j_input_lon] = poly.contains(Point((target_lon,target_lat)))
            
            # Sum these points for each time step
            #for time in range(len(da_input.time.data)):
                #da_output.data[time,:,:] = da_input_bnds[time,:,:].where(mask==1).sum().data
            da_output.data[0,:,:] = da_input_bnds[0,:,:].where(mask==1).sum().data

    da_output.to_dataset(name="Band1")
    da_output.to_netcdf(output_file)

    da_input_bnds.close()
    da_output.close()
    ds_output.close()
    da_input.close()
    ds_input.close()