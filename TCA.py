
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
X_sm, X_asos, X_gpm = np.load('/Users/kim/cpuserver_data/personal_data/project_KIHS/result/ASCAT/precipitation/TCA/X_sm.npy'), np.load('/Users/kim/cpuserver_data/personal_data/project_KIHS/result/ASCAT/precipitation/TCA/X_asos.npy'), np.load('/Users/kim/cpuserver_data/personal_data/project_KIHS/result/ASCAT/precipitation/TCA/X_gpm.npy')
common_dates = np.load('/Users/kim/cpuserver_data/personal_data/project_KIHS/result/ASCAT/precipitation/TCA/common_dates.npy')

# ============================================================
# 월별 Climatology 분리 및 Anomaly 계산
# ============================================================
def calc_anomaly(arr, dates):
    clim = np.full_like(arr, np.nan)
    for m in range(1, 13):
        idx = np.where(dates.month == m)[0]
        mu  = np.nanmean(arr[:, :, idx], axis=2, keepdims=True)
        clim[:, :, idx] = mu
    return arr - clim, clim

A_sm,   clim_sm   = calc_anomaly(X_sm,   common_dates)   # SM2RAIN anomaly
A_asos, _         = calc_anomaly(X_asos, common_dates)   # IDW_ASOS anomaly
A_gpm,  _         = calc_anomaly(X_gpm,  common_dates)   # GPM anomaly

n_lat, n_lon = X_sm.shape[0], X_sm.shape[1]

# ============================================================
# TCA error variance 추정  (ref = SM2RAIN)
# ============================================================
def tca(x, y, z):
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]

    ax_ay = (x @ z) / (y @ z)   # SM2RAIN 기준 ASOS 스케일
    ax_az = (x @ y) / (z @ y)   # SM2RAIN 기준 GPM  스케일

    xs, ys, zs = x, ax_ay * y, ax_az * z

    ex = np.mean((xs - ys) * (xs - zs))
    ey = np.mean((ys - xs) * (ys - zs))
    ez = np.mean((zs - xs) * (zs - ys))

    return ex, ey, ez

# 픽셀별 반복 수행
err2_sm   = np.full((n_lat, n_lon), np.nan)
err2_asos = np.full((n_lat, n_lon), np.nan)
err2_gpm  = np.full((n_lat, n_lon), np.nan)
scale_ay  = np.full((n_lat, n_lon), np.nan)   # ASOS 스케일 파라미터
scale_az  = np.full((n_lat, n_lon), np.nan)   # GPM  스케일 파라미터

for i in range(n_lat):
    for j in range(n_lon):
        s, a, g = A_sm[i,j,:], A_asos[i,j,:], A_gpm[i,j,:]
        if np.sum(np.isfinite(s) & np.isfinite(a) & np.isfinite(g)) < 100:
            continue

        err2_sm[i,j], err2_asos[i,j], err2_gpm[i,j] = tca(s, a, g)

        valid = np.isfinite(s) & np.isfinite(a) & np.isfinite(g)
        sv, av, gv = s[valid], a[valid], g[valid]
        scale_ay[i,j] = (sv @ gv) / (av @ gv)
        scale_az[i,j] = (sv @ av) / (gv @ av)

# 음수 error variance 제거 (TCA 신뢰 불가 픽셀)
err2_sm[err2_sm   <= 0] = np.nan
err2_asos[err2_asos <= 0] = np.nan
err2_gpm[err2_gpm  <= 0] = np.nan


# ============================================================
# Least-square 가중치 산출
# ============================================================
denom    = err2_sm * err2_asos + err2_asos * err2_gpm + err2_sm * err2_gpm

w_sm   = err2_asos * err2_gpm / denom   # SM2RAIN 가중치
w_asos = err2_sm   * err2_gpm / denom   # IDW_ASOS 가중치
w_gpm  = err2_sm   * err2_asos / denom  # GPM 가중치
# w_sm + w_asos + w_gpm = 1.0

T = X_sm.shape[2]

# ============================================================
# Anomaly 가중 병합 
# ============================================================
P_merged_anomaly = np.full((n_lat, n_lon, T), np.nan)

valid_w = np.isfinite(w_sm) & np.isfinite(w_asos) & np.isfinite(w_gpm)

for i in range(n_lat):
    for j in range(n_lon):
        if not valid_w[i, j]:
            continue
        # SM2RAIN 기준으로 스케일링된 anomaly 가중 평균
        xs = A_sm[i, j, :]
        ys = scale_ay[i, j] * A_asos[i, j, :]
        zs = scale_az[i, j] * A_gpm[i, j, :]

        P_merged_anomaly[i, j, :] = (w_sm[i,j]   * xs
                                   + w_asos[i,j]  * ys
                                   + w_gpm[i,j]   * zs)


# ============================================================
# Climatology 복원  →  최종 융합 강수량 산출
# ============================================================
P_merged = np.maximum(P_merged_anomaly + clim_sm, 0.0)



# ============================================================
# NetCDF 저장
# ============================================================
ds = xr.Dataset(
    data_vars={
        "precipitation": (["time", "lat", "lon"],
                          np.transpose(P_merged, (2, 0, 1)),
                          {"long_name": "TCA-merged precipitation",
                           "units": "mm/day"})
    },
    coords={
        "time": common_dates.values,
        "lat" : lat_1d,
        "lon" : lon_1d,
    }
)
ds.to_netcdf("SM2RAIN_TCA.nc")