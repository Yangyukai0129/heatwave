import xarray as xr
import pandas as pd
import numpy as np

# 讀入資料
t = xr.open_mfdataset(
    './data/5deg/merged_1965_2024.nc',
    combine='by_coords',
    chunks={'valid_time': 1000}
)

# 時間設定
temp = t.assign_coords(valid_time=pd.to_datetime(t.valid_time.values))
temp_daily = temp.resample(valid_time='1D').mean()
temp_daily_celsius = temp_daily - 273.15

# 建立 month_day 方便計算 climatology
temp_daily_celsius = temp_daily_celsius.assign_coords(
    month_day=temp_daily_celsius['valid_time'].dt.strftime('%m-%d')
)
clim_daily_mean = temp_daily_celsius.groupby('month_day').mean()

# 計算 anomaly
anomaly = temp_daily_celsius.groupby('month_day') - clim_daily_mean
anomaly = anomaly['t'].compute()

# 判斷是否 > 5°C
heat_bool = (anomaly > 5).astype('int8')

# 轉換成 DataFrame 並加入年月日
df = heat_bool.to_dataframe().reset_index()
df['year'] = pd.to_datetime(df['valid_time']).dt.year
df['date'] = pd.to_datetime(df['valid_time'])

# 對每個格點處理連續 5 天以上的事件
results = []

for (lat, lon), group in df.groupby(['latitude', 'longitude']):
    group = group.sort_values('date')
    group = group.set_index('date')
    
    is_heat = group['t'].values
    dates = group.index
    
    count = 0
    i = 0
    while i <= len(is_heat) - 5:
        if np.all(is_heat[i:i+5] == 1):
            # 找連續區段起始日
            start_date = dates[i]
            count += 1
            # 跳過這整段（直到遇到不為1）
            while i < len(is_heat) and is_heat[i] == 1:
                i += 1
        else:
            i += 1
    
    results.append({
        'latitude': lat,
        'longitude': lon,
        'heatwave_event_count': count
    })

# 轉成 DataFrame & 輸出
event_df = pd.DataFrame(results)
event_df = event_df[event_df['heatwave_event_count'] > 0]
event_df.to_csv('./data/heatwave_event_5days.csv', index=False)
print(event_df.head())
print(f"總共熱浪事件筆數（連續5天以上）：{len(event_df)}")
