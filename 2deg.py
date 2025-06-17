import xarray as xr
import pandas as pd

# 1. ���J�ƾ�
t = xr.open_dataset('./data/t_1980_2002.nc', chunks={'valid_time': 1000})
t = t.sel(pressure_level=1000)
temp = t['t']
temp = temp.assign_coords(valid_time=pd.to_datetime(temp.valid_time.values))

# 2. �ʤƨ� 5�X(�� mean)
temp_coarse = temp.coarsen(latitude=8, longitude=8, boundary='trim').mean()

# 3. �x�s
encoding = {'t': {'dtype': 'int8', 'zlib': True, 'complevel': 5}}
heatwave_events.to_netcdf('./data/heatwave_events_2deg(1980-2002).nc', encoding=encoding)