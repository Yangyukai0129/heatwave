import xarray as xr
import pandas as pd

# 1. ¸ü¤J¼Æ¾Ú
t = xr.open_dataset('./data/t_1964_1979.nc', chunks={'valid_time': 1000})
# temp = t['t']
temp = t.assign_coords(valid_time=pd.to_datetime(t.valid_time.values))

# 2. ²Ê¤Æ¨ì 2¢X¡]¥Î mean¡^
temp_coarse = temp.coarsen(latitude=20, longitude=20, boundary='trim').mean()


# 3. Àx¦s
encoding = {'t': {'dtype': 'int8', 'zlib': True, 'complevel': 5}}
temp_coarse.to_netcdf('./data/5deg(1964-1979).nc', encoding=encoding)