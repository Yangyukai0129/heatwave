import xarray as xr
import glob

# 1. 讀取所有檔案路徑（請依實際情況修改）
file_paths = sorted(glob.glob("./data/5deg/*.nc"))  # 假設資料在 data 資料夾內

# 2. 合併所有檔案
ds_all = xr.open_mfdataset(file_paths, combine='by_coords')

# 3. 過濾時間範圍：只保留 1980~2024 年
ds_filtered = ds_all.sel(valid_time=slice("1965-06-01", "2024-08-31"))

# 4. 儲存為新檔案（可選）
ds_filtered.to_netcdf("merged_1965_2024.nc")