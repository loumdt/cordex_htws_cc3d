#%%
import numpy as np
from cartopy.io import shapereader
import geopandas as gpd
from os.path import join
import regionmask
from tqdm import tqdm
import xarray as xr

if __name__ == "__main__":
    datadir = "/home/user/These/cordex_htws_cc3d/Data"

    # Countries that we want to keep
    list_countries = ['France','Ukraine','Belarus','Lithuania','Russia','Czechia','Germany','Estonia','Latvia','Norway',
    'Sweden','Finland','Luxembourg','Belgium','North Macedonia','Albania','Kosovo','Spain','Denmark','Romania','Hungary',
    'Slovakia','Poland','Ireland','United Kingdom','Greece','Austria','Italy','Switzerland','Netherlands','Liechtenstein',
    'Republic of Serbia','Croatia','Slovenia','Bulgaria','San Marino','Monaco','Andorra','Montenegro','Bosnia and Herzegovina',
    'Portugal','Moldova','Gibraltar','Vatican','Iceland','Malta','Jersey','Guernsey','Isle of Man','Aland','Faroe Islands',
    'Turkey','Armenia','Georgia','Azerbaijan']

    # Request data for use by geopandas
    resolution = '10m'
    category = 'cultural'
    name = 'admin_0_countries'

    shpfilename = shapereader.natural_earth(resolution, category, name)
    df = gpd.read_file(shpfilename)
    # Select countries that we want to keep
    df_to_mask = df[df['ADMIN'].isin(list_countries)]

    # Open target grid file
    file = "/home/user/These/cordex_htws_cc3d/Data/GHS-POP/GHS_POP_2020_4326_smallbox.nc"
    ds = xr.open_dataset(join(datadir,'cellarea',file),engine='netcdf4')

    # Rename variable and set data to zeros
    ds = ds.rename({'Band1':'mask'})
    da = ds.mask
    da.data = np.zeros(np.shape(da.data))

    # Compute mask
    mask = regionmask.mask_geopandas(df_to_mask, ds.lon, ds.lat)
    mask = mask.fillna(0)

    # Mask data where mask is zero
    da.data = (mask==0)

    # Export to netcdf
    da = da.to_dataset(name="mask")
    da.to_netcdf(join(datadir,'mask','GHS_POP_2020_4326_smallbox_mask_land_only.nc'))

    da.close()
    ds.close()