import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import xarray as xr
from tqdm import tqdm
import joblib

BASE_PATH = '/home/jaese/cpuserver_data/personal_data/project_KIHS/result/ASCAT'

# ==============================================================================
# Load LightGBM (LGBM)
# ==============================================================================
LGBM = joblib.load(BASE_PATH + "lightgbm_model.pkl")

# ==============================================================================
# Load TCA merging weights
# ==============================================================================
TCA_weights = xr.open_dataset(BASE_PATH +'/precipitation/TCA/TCA_weights_ERA5_GES.nc')





# ============================================================
# 경로 설정
# ============================================================
BASE_PATH = '/home/jaese/cpuserver_data/personal_data/project_KIHS/result/'
DATA_PATH = '/home/jaese/cpuserver_data/python_modules/kunhee/Results/SM2RAIN'
YEARS     = [2021, 2022, 2023, 2024, 2025]

TCA_MERGE_PATH = os.path.join(BASE_PATH, 'KIHS_data_all.nc')

OUT_FIG_PATH   = os.path.join('/home/jaese/bias_correction_result.png')

da_all = xr.open_dataset(TCA_MERGE_PATH)

idx = 20
SM2RAIN = np.flipud(da_all['SM2RAIN'][idx, :,:])
GPM = np.flipud(da_all['GPM'][idx, :,:])
ERA5 = np.flipud(da_all['ERA5'][idx, :,:])

# ============================================================
# 7. Anomaly 병합
# ============================================================
print("\nStep 7. Anomaly 병합...")

scale_era5 = np.full((49,49), np.nan)
scale_gpm  = np.full((49,49), np.nan)

for i in range(n_lat):
    for j in range(n_lon):
        s = A_sm[i, j, :];  a = A_era5[i, j, :];  g = A_gpm[i, j, :]
        valid = np.isfinite(s) & np.isfinite(a) & np.isfinite(g)
        if valid.sum() < MIN_TRIPLET:
            continue
        sv, av, gv = s[valid], a[valid], g[valid]
        d_y = float(av @ gv);  d_z = float(gv @ av)
        if abs(d_y) > 1e-12:
            scale_era5[i, j] = float(sv @ gv) / d_y
        if abs(d_z) > 1e-12:
            scale_gpm[i, j]  = float(sv @ av) / d_z

M_anomaly = np.full((n_lat, n_lon, T), np.nan)

for i in tqdm(range(n_lat), desc="  위도줄"):
    for j in range(n_lon):
        if not valid_w[i, j]:
            continue
        if not (np.isfinite(scale_era5[i,j]) and np.isfinite(scale_gpm[i,j])):
            continue
        xs = A_sm[i, j, :]
        ys = scale_era5[i, j] * A_era5[i, j, :]
        zs = scale_gpm[i, j]  * A_gpm[i, j, :]
        M_anomaly[i, j, :] = w_sm[i,j]*xs + w_era5[i,j]*ys + w_gpm[i,j]*zs

print("  Anomaly 병합 완료")


