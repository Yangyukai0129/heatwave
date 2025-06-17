import xarray as xr
import pandas as pd

# 1. 載入數據
t = xr.open_dataset('./data/t_1980_2002.nc', chunks={'valid_time': 1000})
t = t.sel(pressure_level=1000)
temp = t['t']
temp = temp.assign_coords(valid_time=pd.to_datetime(temp.valid_time.values))

# 2. 粗化到 5°(用 mean)
temp_coarse = temp.coarsen(latitude=8, longitude=8, boundary='trim').mean()

# 3. 儲存
encoding = {'t': {'dtype': 'int8', 'zlib': True, 'complevel': 5}}
heatwave_events.to_netcdf('./data/heatwave_events_2deg(1980-2002).nc', encoding=encoding)