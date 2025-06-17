import xarray as xr
import pandas as pd
import numpy as np

# ---------------- 1. 載入並前處理（保留你既有的寫法） ----------------
t = xr.open_dataset('./data/5deg/merged_1965_2024.nc',
                    chunks={'valid_time': 1000})
t = t.assign_coords(valid_time=pd.to_datetime(t.valid_time.values))

temp_daily  = (t.resample(valid_time='1D').mean()   # 先取每日平均
                 - 273.15)                         # 轉攝氏
temp_daily  = temp_daily.assign_coords(
                 month_day=temp_daily['valid_time'].dt.strftime('%m-%d'))

clim_daily_mean = temp_daily.groupby('month_day').mean()      # 日氣候平均
anomaly         = (temp_daily.groupby('month_day') -          # 日異常
                   clim_daily_mean)['t'].compute()

heat_bool = (anomaly > 5).astype('int8')                      # 異常 > 5 °C

# --------------- 2. 轉成長表並加 decade 欄位 ----------------
df = heat_bool.to_dataframe().reset_index()
df['date']   = pd.to_datetime(df['valid_time'])
df['year']   = df['date'].dt.year
df['decade'] = (df['year'] - 1965) // 10 * 10 + 1965          # 1965, 1975, …

# --------------- 3. 逐 (lat, lon, decade) 計算事件數 ------------
results = []

def count_events(is_heat, dates):
    """計算同一時段內「連續 5 天 = 1」的事件數"""
    cnt, i, n = 0, 0, len(is_heat)
    while i <= n-5:
        if np.all(is_heat[i:i+5] == 1):
            cnt += 1
            while i < n and is_heat[i] == 1:      # 跳過整段連續 1
                i += 1
        else:
            i += 1
    return cnt

for (lat, lon), g in df.groupby(['latitude', 'longitude']):
    g = g.sort_values('date')                     # 先排好時間
    for dec, sub in g.groupby('decade'):
        sub = sub.set_index('date')
        cnt = count_events(sub['t'].values, sub.index.values)
        if cnt:                                   # 只留有事件的格點
            results.append({
                'latitude' : lat,
                'longitude': lon,
                'decade'   : f'{dec}-{dec+9}',
                'heatwave_event_count': cnt
            })

# --------------- 4. 匯出結果 ----------------
event_df = pd.DataFrame(results)
event_df.to_csv('./data/5deg/heatwave_event_5days_by_decade.csv', index=False)

print(event_df.head())
print(f"共有 {len(event_df)} 筆 (lat,lon,decade) 熱浪事件紀錄")
