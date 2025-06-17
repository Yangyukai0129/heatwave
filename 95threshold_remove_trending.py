import xarray as xr
import pandas as pd
from scipy import stats
import numpy as np

# 1. 定義去除線性趨勢的函式（使用線性斜率法）
def detrend_dim(da, dim='time'):
    """
    去除 DataArray 中沿指定維度的線性趨勢，忽略 NaN。
    """
    def detrend_1d(y):
        x = np.arange(len(y))
        mask = ~np.isnan(y)
        if mask.sum() < 2:  # 少於兩個有效點無法擬合
            return y
        # 線性擬合（忽略 NaN）
        slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
        trend = slope * x + intercept
        return y - trend

    return xr.apply_ufunc(
        detrend_1d, da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[da.dtype],
    )

# 2. 載入 5 度解析度的 NetCDF 資料（氣溫變數為 't'）
ds = xr.open_dataset('./data/5deg/merged_1965_2024.nc', chunks={'valid_time': 1000})
ds = ds.assign_coords(valid_time=pd.to_datetime(ds.valid_time.values))  # 將時間座標轉為 datetime64
# 限制處理範圍只在夏季
ds = ds.sel(valid_time=ds['valid_time'].dt.month.isin([6, 7, 8]))

# 3. 將原始資料重採樣為每日資料（取每日最大氣溫）
ds_daily = ds.resample(valid_time='1D').max()
ds_daily = ds_daily.chunk({'valid_time': -1})  # 調整 chunk，讓 valid_time 為一個整體處理

# 5. 對補值後的資料進行趨勢移除，取得無趨勢的氣溫資料
t_detrended = detrend_dim(ds_daily['t'], dim='valid_time')

# 6. 將去趨勢後的資料指定回一個新的 DataArray
ds_detrended = ds_daily.copy()
ds_detrended['t'] = t_detrended

# 7. 計算去趨勢後的 95 百分位數閾值（每個格點自己的極端高溫門檻）
thresholds = ds_detrended['t'].quantile(0.95, dim='valid_time')

# 8. 判斷每一天是否為熱浪（高於該點 95% 閾值），結果為 0/1 的布林陣列
heatwave_events = (ds_detrended['t'] > thresholds).astype('int8')

# 9. 統計所有時間與所有格點上的熱浪事件總數
total_events = heatwave_events.sum().compute().item()

# 10. 輸出結果
print(f"總熱浪事件數量（所有時間 + 所有網格點）: {total_events}")

event_counts = heatwave_events.sum(dim='valid_time')  # 每個格點熱浪次數
max_event_loc = event_counts.where(event_counts == event_counts.max(), drop=True)
lat = float(max_event_loc.latitude.values[0])
lon = float(max_event_loc.longitude.values[0])
print(f"熱浪事件最多的格點：lat={lat}, lon={lon}")

# 假設 lat, lon 是你前面已經找好的熱浪最多的格點
sel_raw      = ds_daily['t'].sel(latitude=lat, longitude=lon)
sel_detrend  = t_detrended.sel(latitude=lat, longitude=lon)

# 將兩個時間序列轉為 DataFrame 比較
df_compare = pd.DataFrame({
    'date': sel_raw['valid_time'].values,
    'original_temp': sel_raw.values,
    'detrended_temp': sel_detrend.values
})

# 儲存成 CSV
df_compare.to_csv('./data/5deg/temp_timeseries_raw_vs_detrended.csv', index=False)

# 11. 儲存結果
#encoding = {'t': {'dtype': 'int8', 'zlib': True, 'complevel': 5}}
#heatwave_events.to_netcdf('./data/5deg/heatwave_events_5deg_remove_trending(1965-2024).nc', encoding=encoding)