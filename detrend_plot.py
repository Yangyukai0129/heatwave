import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # Windows 推薦使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 避免負號顯示為方框


# 讀取資料並轉換日期格式
df = pd.read_csv('./data/temp_timeseries_raw_vs_detrended.csv', parse_dates=['date'])
df['year'] = df['date'].dt.year

# 建立年份區間（例如 1965-1969, 1970-1974, ..., 2020-2024）
start_year = df['year'].min()
end_year = df['year'].max()
interval = 10
year_ranges = [(y, min(y + interval - 1, end_year)) for y in range(start_year, end_year + 1, interval)]

# 建立子圖
n_rows = len(year_ranges)
fig, axes = plt.subplots(n_rows, 1, figsize=(13, 6 * n_rows), sharey=True)

# for ax, (y_start, y_end) in zip(axes, year_ranges):
#     df_sub = df[(df['year'] >= y_start) & (df['year'] <= y_end)]
#     ax.plot(df_sub['date'], df_sub['original_temp'], label='原始氣溫', alpha=0.6)
#     ax.plot(df_sub['date'], df_sub['detrended_temp'], label='去趨勢後氣溫', alpha=0.6)
#     ax.set_title(f'{y_start}–{y_end} 年氣溫比較')
#     ax.set_ylabel('氣溫 (°C)')
#     ax.legend()
#     ax.grid(True)

# axes[-1].set_xlabel('日期')
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.4)  # 增加子圖之間的垂直間距
# plt.show()
step = 3  # 每頁顯示 3 個區間
for i in range(0, len(year_ranges), step):
    fig, axes = plt.subplots(step, 1, figsize=(12, 3 * step), sharey=False)
    for ax, (y_start, y_end) in zip(axes, year_ranges[i:i+step]):
        df_sub = df[(df['year'] >= y_start) & (df['year'] <= y_end)]
        ax.plot(df_sub['date'], df_sub['original_temp'], label='Original Temperature', alpha=0.6)
        ax.plot(df_sub['date'], df_sub['detrended_temp'], label='Detrended Temperature', alpha=0.6)
        ax.set_title(f'Temperature Comparison: {y_start}–{y_end}')
        ax.set_ylabel('temp (°C)')
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()