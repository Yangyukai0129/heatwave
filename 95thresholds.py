import xarray as xr
import pandas as pd

# 1. 讀取資料
t = xr.open_dataset('./data/5deg/merged_1965_2024.nc', chunks={'valid_time': 1000})

# 將 valid_time 轉為 datetime 格式
temp = t.assign_coords(valid_time=pd.to_datetime(t.valid_time.values))

# 計算每日的最大氣溫
temp_daily = temp.resample(valid_time='1D').max()
temp_daily = temp_daily.chunk({'valid_time': -1})

# 2. 計算每個格點的 95 百分位數溫度（作為熱浪門檻）
thresholds = temp_daily.quantile(0.95, dim='valid_time')

# 3. 判斷是否為熱浪事件（溫度大於門檻則標記為 1）
heatwave_events = (temp_daily > thresholds).astype('int8')

# 計算總熱浪事件數量（所有時間 + 所有網格點的加總）
total_events = heatwave_events['t'].sum().compute().item()
print(f"總熱浪事件數量（所有時間 + 所有網格點）: {total_events}")

# 4. 儲存檔案（可選）
encoding = {'t': {'dtype': 'int8', 'zlib': True, 'complevel': 5}}
heatwave_events.to_netcdf('./data/5deg/heatwave_events_5deg(1965-2024).nc', encoding=encoding)