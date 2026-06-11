"""
bias_correction_ASOS.py
=======================
Li et al. (2023, Remote Sens.) 방법론 기반 LightGBM Bias Correction

* TCA 결과 파일을 직접 읽어 독립 실행 가능
    TCA merged : TCA/SM2RAIN_TCA_interp.nc
    SM2RAIN    : SM2RAIN/SM2RAIN_interp.nc
    GPM        : DATA_PATH/GPM_{yr}.nc
    IDW_ASOS   : DATA_PATH/Precipitation_IDW_ASOS_{yr}.nc

흐름:
  Load  TCA merged + 개별 자료 로딩 → 공통 기간 슬라이싱
  BC-1  데이터 준비 + Train/Val 분할 (전반부/후반부)
  BC-2  LightGBM 하이퍼파라미터 최적화 (RandomizedSearchCV)
  BC-3  최종 모델 학습
  BC-4  Validation 평가 (RMSE, BIAS)
  BC-5  SHAP Feature Importance
  BC-6  공간 복원 (lat, lon, T_val) → 3D array
  BC-7  NetCDF 저장
  BC-8  시각화
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import xarray as xr
from tqdm import tqdm

from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
import shap

warnings.filterwarnings("ignore")

plt.rcParams['font.family']        = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 경로 설정
# ============================================================
BASE_PATH = '/home/jaese/cpuserver_data/personal_data/project_KIHS/result/ASCAT'
DATA_PATH = '/home/jaese/cpuserver_data/python_modules/kunhee/Results/SM2RAIN'
YEARS     = [2021, 2022, 2023, 2024, 2025]

TCA_MERGE_PATH = os.path.join(BASE_PATH, 'precipitation/TCA/TCA_ERA5_GES.nc')
SM2RAIN_PATH   = os.path.join(BASE_PATH, 'precipitation/SM2RAIN/SM2RAIN_interp.nc')
OUT_BC_PATH    = os.path.join(BASE_PATH, 'precipitation/biasC/TCA_BC.nc')
OUT_FIG_PATH   = os.path.join('/home/jaese/bias_correction_result.png')


# ============================================================
# 유틸 함수
# ============================================================
def load_precip(path):
    """nc 로딩 → arr(lat, lon, time), lat_1d, lon_1d, DatetimeIndex"""
    ds  = xr.open_dataset(path)
    var = next(v for v in ds.data_vars
               if v.lower() not in {'lat', 'lon', 'latitude', 'longitude', 'time'})
    da  = ds[var]

    def get_coord_1d(ds, keys, axis):
        for k in list(ds.coords) + list(ds.data_vars):
            if any(kw in k.lower() for kw in keys):
                v = ds[k].values.squeeze()
                if v.ndim == 2:
                    return v[:, 0] if axis == 0 else v[0, :]
                return v
        return None

    lat_1d  = get_coord_1d(ds, ['lat', 'latitude'],  axis=0)
    lon_1d  = get_coord_1d(ds, ['lon', 'longitude'], axis=1)
    time_1d = ds['time'].values

    dims    = [d.lower() for d in da.dims]
    lat_ax  = next((i for i, d in enumerate(dims) if 'lat' in d or d == 'y'), None)
    lon_ax  = next((i for i, d in enumerate(dims) if 'lon' in d or d == 'x'), None)
    time_ax = next((i for i, d in enumerate(dims) if 'time' in d), None)

    if None in (lat_ax, lon_ax, time_ax):
        sizes   = da.values.shape
        time_ax = int(np.argmax(sizes))
        rem     = [i for i in range(3) if i != time_ax]
        lat_ax, lon_ax = rem[0], rem[1]

    arr = np.transpose(da.values.astype(np.float64), (lat_ax, lon_ax, time_ax))

    if lat_1d is not None and lat_1d[0] > lat_1d[-1]:
        lat_1d = lat_1d[::-1];  arr = arr[::-1, :, :]
    if lon_1d is not None and lon_1d[0] > lon_1d[-1]:
        lon_1d = lon_1d[::-1];  arr = arr[:, ::-1, :]

    arr[arr < 0] = np.nan
    return arr, lat_1d, lon_1d, pd.DatetimeIndex(time_1d)


# ============================================================
# 데이터 로딩
# ============================================================
print("=" * 60)
print("데이터 로딩")
print("=" * 60)
# P_merged.shape
# plt.imshow(arr_sm[:, :, 0])
# plt.colorbar()
# plt.show()

# ── TCA merged ───────────────────────────────────────────────
P_merged, lat_1d, lon_1d, time_tca = load_precip(TCA_MERGE_PATH)
n_lat, n_lon = lat_1d.shape[0], lon_1d.shape[0]
print(f"  P_merged  : {P_merged.shape}  {time_tca[0].date()} ~ {time_tca[-1].date()}")

# ── SM2RAIN ──────────────────────────────────────────────────
arr_sm, _, _, time_sm = load_precip(SM2RAIN_PATH)
print(f"  SM2RAIN   : {arr_sm.shape}  {time_sm[0].date()} ~ {time_sm[-1].date()}")

# ── GPM (연도별 concat) ───────────────────────────────────────
gpm_arrs, gpm_times = [], []
for yr in YEARS:
    arr, _, _, t = load_precip(os.path.join(DATA_PATH, f'GPM_{yr}.nc'))
    gpm_arrs.append(arr);  gpm_times.append(t)
arr_gpm  = np.concatenate(gpm_arrs, axis=2)
time_gpm = gpm_times[0]
for t in gpm_times[1:]:  time_gpm = time_gpm.append(t)
print(f"  GPM       : {arr_gpm.shape}  {time_gpm[0].date()} ~ {time_gpm[-1].date()}")

# ── IDW_ASOS (연도별 concat) ──────────────────────────────────
asos_arrs, asos_times = [], []
for yr in YEARS:
    arr, _, _, t = load_precip(os.path.join(DATA_PATH, f'Precipitation_IDW_ASOS_{yr}.nc'))
    asos_arrs.append(arr);  asos_times.append(t)
arr_asos  = np.concatenate(asos_arrs, axis=2)
time_asos = asos_times[0]
for t in asos_times[1:]:  time_asos = time_asos.append(t)
print(f"  IDW_ASOS  : {arr_asos.shape}  {time_asos[0].date()} ~ {time_asos[-1].date()}")

# ── ERA5 (단일 파일) ──────────────────────────────────────────
ERA5_PATH = '/home/jaese/cpuserver_data/personal_data/project_KIHS/data/era5_'
arr_era5, _, _, time_era5 = load_precip(os.path.join(ERA5_PATH, 'era5_land_P_regrid.nc'))
print(f"  ERA5      : {arr_era5.shape}  {time_era5[0].date()} ~ {time_era5[-1].date()}")

# ── IDW_AWS (연도별 concat) ──────────────────────────────────
aws_arrs, aws_times = [], []

for yr in YEARS:
    fp = os.path.join(DATA_PATH, f'Precipitation_IDW_AWS_{yr}.nc')
    arr, _, _, t = load_precip(fp)

    aws_arrs.append(arr)
    aws_times.append(t)

arr_aws = np.concatenate(aws_arrs, axis=2)

time_aws = aws_times[0]
for t in aws_times[1:]:
    time_aws = time_aws.append(t)

print(f"  IDW_AWS   : {arr_aws.shape}  {time_aws[0].date()} ~ {time_aws[-1].date()}")


# ── 공통 기간 슬라이싱 ────────────────────────────────────────
common_dates = (time_tca
                .intersection(time_sm)
                .intersection(time_gpm)
                .intersection(time_asos)
                .intersection(time_aws)
                .intersection(time_era5))

common_dates = common_dates.sort_values()

print(f"\n  공통 기간 : {common_dates[0].date()} ~ {common_dates[-1].date()}  ({len(common_dates)}일)")

X_sm     = arr_sm    [:, :, time_sm.isin(common_dates)]
X_gpm    = arr_gpm   [:, :, time_gpm.isin(common_dates)]
X_asos   = arr_asos  [:, :, time_asos.isin(common_dates)]
X_aws    = arr_aws   [:, :, time_aws.isin(common_dates)]
P_merged = P_merged  [:, :, time_tca.isin(common_dates)]
X_era5   = arr_era5  [:, :, time_era5.isin(common_dates)]

T = len(common_dates)

print("After common-date slicing")
print("X_sm    :", X_sm.shape)
print("X_gpm   :", X_gpm.shape)
print("X_asos  :", X_asos.shape)
print("X_aws   :", X_aws.shape)
print("P_merged:", P_merged.shape)
print("X_era5  :", X_era5.shape)


df_all = pd.DataFrame({
        'SM2RAIN': ds_bc['SM2RAIN'].data.flatten(),
        'GPM':     ds_bc['GPM'].data.flatten(),
        'TCA':     ds_bc['TCA'].data.flatten(),
        'ERA5':    ds_bc['ERA5'].data.flatten(),
        'Y':       ds_bc['AWS'].data.flatten(),
    })




# df_all = pd.DataFrame({
#         'SM2RAIN': X_sm.flatten(),
#         'GPM':     X_gpm.flatten(),
#         'TCA':     P_merged.flatten(),
#         'ERA5':    X_era5.flatten(),
#         'Y':       X_asos.flatten(),
#     })
# df_all_ = df_all.dropna()

ds_bc = xr.Dataset(
    data_vars={
        # "BC": xr.DataArray(np.transpose(BC_map, (2, 0, 1)),dims=("time", "lat", "lon"),attrs={
        #         "long_name"   : "LightGBM bias-corrected TCA merged precipitation",
        #         "units"       : "mm/day",
        #         "method"      : "LightGBM regression",
        #         "features"    : ", ".join(FEATURES),
        #         "target"      : TARGET,
        #     }
        # ),
        "SM2RAIN": xr.DataArray(np.transpose(X_sm, (2, 0, 1)),dims=("time", "lat", "lon"),attrs={
                "long_name"   : "ASCAT SM2RAIN-based precipitation",
                "units"       : "mm/day",
                "method"      : "SM2RAIN",
                "note"      : "SM2RAIN parameters (A, B, Z) are obtained using interpolated AWS data",
            }
        ),
        "ERA5": xr.DataArray(np.transpose(X_era5, (2, 0, 1)),dims=("time", "lat", "lon"),attrs={
                "long_name"   : "ERA5 precipitation",
                "units"       : "mm/day",
            }
        ),
        "GPM": xr.DataArray(np.transpose(X_gpm, (2, 0, 1)),dims=("time", "lat", "lon"),attrs={
                "long_name"   : "GPM IMERG Final precipitation",
                "units"       : "mm/day",
            }
        ),
        "TCA": xr.DataArray(np.transpose(P_merged, (2, 0, 1)),dims=("time", "lat", "lon"),attrs={
                "long_name"   : "TCA merged precipitation",
                "units"       : "mm/day",
                "features"    : "SM2RAIN, ERA5, GPM",
            }
        ),
        "ASOS": xr.DataArray(np.transpose(X_asos, (2, 0, 1)),dims=("time", "lat", "lon"),attrs={
                "long_name"   : "Interpolated ASOS precipitation using IDW method",
                "units"       : "mm/day",
            }
        ),
        "AWS": xr.DataArray(np.transpose(X_aws, (2, 0, 1)), dims=("time", "lat", "lon"), attrs={
                "long_name"   : "Interpolated AWS precipitation using IDW method",
                "units"       : "mm/day",
            }
        ),
    },
    coords={
        "time": xr.DataArray(common_dates.values, dims="time"),
        "lat" : xr.DataArray(lat_1d, dims="lat",
                             attrs={"long_name": "latitude",  "units": "degrees_north"}),
        "lon" : xr.DataArray(lon_1d, dims="lon",
                             attrs={"long_name": "longitude", "units": "degrees_east"}),
    },
)


ds_bc.to_netcdf(BASE_PATH + 'KIHS_data_all.nc')

/Users/jslee/Desktop/CPU_home/cpuserver_data/personal_data/project_KIHS/result/ASCAT/lightgbm_model.pkl

cp /home/jaese/cpuserver_data/personal_data/project_KIHS/result/KIHS_data_all.nc /Users/hyunglokkim/data_2/projects/KIHS/
cp /home/jaese/cpuserver_data/personal_data/project_KIHS/result/KIHS_data_all.nc /Users/hyunglokkim/data_2/projects/KIHS/

# ============================================================
# BC-1. 데이터 준비 + Train/Val 분할
#   논문: "first half for training, second half for validation"
# ============================================================
print("\n" + "=" * 60)
print("BC-1. 데이터 준비 + Train/Val 분할")
print("=" * 60)


common_dates[T_half]



print(f"  Train: {common_dates[train_idx[ 0]].date()} ~ "
      f"{common_dates[train_idx[-1]].date()}  ({len(train_idx)}일)")
print(f"  Val  : {common_dates[val_idx[ 0]].date()} ~ "
      f"{common_dates[val_idx[-1]].date()}  ({len(val_idx)}일)")

FEATURES = ['SM2RAIN', 'GPM', 'ERA5', 'TCA']
TARGET   = 'AWS'

def make_flat_df(idx):
    df = pd.DataFrame({
        'SM2RAIN': ds_bc['SM2RAIN'][idx, :, :].data.flatten(),
        'GPM':     ds_bc['GPM'][idx, :, :].data.flatten(),
        'TCA':     ds_bc['TCA'][idx, :, :].data.flatten(),
        'ERA5':    ds_bc['ERA5'][idx, :, :].data.flatten(),
        'Y':       ds_bc['AWS'][idx, :, :].data.flatten(),
    })#.dropna()
    return df[(df[FEATURES] >= 0).all(axis=1) & (df['Y'] >= 0)]

T_half    = len(ds_bc.time) // 2
train_idx = np.arange(0, T_half)
val_idx   = np.arange(T_half, T)

df_train = make_flat_df(train_idx)
df_val   = make_flat_df(val_idx)

print(f"\n  Train 유효 샘플: {len(df_train):,}")
print(f"  Val   유효 샘플: {len(df_val):,}")
print(f"  Features : {FEATURES}")
print(f"  Target   : {TARGET}")



# ============================================================
# BC-2. LightGBM 하이퍼파라미터 최적화 (RandomizedSearchCV)
# ============================================================
print("\n" + "=" * 60)
print("BC-2. Hyperparameter 최적화 (RandomizedSearchCV)")
print("=" * 60)

PARAM_DIST = {
    'n_estimators':      [100, 200, 300, 500, 700],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth':         [-1, 5, 7, 10, 15],
    'num_leaves':        [15, 31, 63, 127, 255],
    'min_child_samples': [5, 10, 20, 30, 50],
    'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_split_gain':    [0.0, 0.05, 0.1, 0.2, 0.3],
    'reg_alpha':         [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda':        [0.5, 1.0, 1.5, 2.0, 3.0],
}

MAX_CV_SAMPLE = 50_000
df_cv = (df_train.sample(MAX_CV_SAMPLE, random_state=42)
         if len(df_train) > MAX_CV_SAMPLE else df_train)

rs = RandomizedSearchCV(
    estimator           = LGBMRegressor(random_state=42, n_jobs=1, verbose=-1),
    param_distributions = PARAM_DIST,
    n_iter              = 50,
    cv                  = 5,
    scoring             = 'neg_root_mean_squared_error',
    random_state        = 42,
    n_jobs              = 40,
    verbose             = 0,
)
rs.fit(df_cv[FEATURES], df_cv['Y'])

best_params  = rs.best_params_
best_cv_rmse = -rs.best_score_
print(f"  Best CV RMSE : {best_cv_rmse:.4f} mm/day")
print(f"  Best params  : {best_params}")


# ============================================================
# BC-3. 최종 모델 학습 (전체 train set)
# ============================================================
print("\n" + "=" * 60)
print("BC-3. 최종 모델 학습")
print("=" * 60)

lgbm_model = LGBMRegressor(
    random_state=42, n_jobs=40, verbose=-1, **best_params,
)
lgbm_model.fit(df_train[FEATURES], df_train['Y'])

df_all_ = df_all.dropna()

BC_all = lgbm_model.predict(df_all_[FEATURES])


# from lightgbm import LGBMRegressor
import joblib

# model = LGBMRegressor()
# model.fit(X_train, y_train)

# 저장
joblib.dump(lgbm_model, BASE_PATH + "/lightgbm_model.pkl")

import joblib

# model = joblib.load("lightgbm_model.pkl")
model2 = joblib.load(BASE_PATH + "/lightgbm_model.pkl")

X_sm.shape

plt.imshow(BC_map[30, :,:])

df_all_['BC'] = BC_all

df_all.loc[df_all_.index, 'BC'] = df_all_['BC']

BC_map = df_all['BC'].values.reshape(ds_bc['GPM'].shape)

ds_bc['BC'] = (('time','lat','lon'), BC_map)

plt.plot(ds_bc['BC'][:,30,30], alpha = .3)
plt.plot(ds_bc['ASOS'][:,30,30], alpha = .3)
plt.plot(ds_bc['AWS'][:,30,30], alpha = .3)

plt.scatter(ds_bc['AWS'].data, ds_bc['BC'].data)
plt.scatter(ds_bc['ASOS'].data, ds_bc['BC'].data)

fig,axs= plt.subplots(1,3, figsize = (12, 4))
im0 = axs[0].imshow(np.flipud(eval_maps_AWS[0]),vmin=0,vmax=1, cmap = 'jet')
im1 = axs[1].imshow(np.flipud(eval_maps_AWS[1]),vmin=0,vmax=15, cmap = 'jet')
im2 = axs[2].imshow(np.flipud(eval_maps_AWS[2]),vmin=-10,vmax=10, cmap = 'RdBu')
axs[0].set_title('R')
axs[1].set_title('ubRMSD [mm/day]')
axs[2].set_title('bias [mm/day]')
fig.colorbar(ax = axs[0], mappable = im0, orientation = 'horizontal')
fig.colorbar(ax = axs[1], mappable = im1, orientation = 'horizontal')
fig.colorbar(ax = axs[2], mappable = im2, orientation = 'horizontal')


fig,axs= plt.subplots(1,3, figsize = (12, 4))
im0 = axs[0].imshow(np.flipud(eval_maps_ASOS[0]),vmin=0,vmax=1, cmap = 'jet')
im1 = axs[1].imshow(np.flipud(eval_maps_ASOS[1]),vmin=0,vmax=15, cmap = 'jet')
im2 = axs[2].imshow(np.flipud(eval_maps_ASOS[2]),vmin=-10,vmax=10, cmap = 'RdBu')
axs[0].set_title('R')
axs[1].set_title('ubRMSD [mm/day]')
axs[2].set_title('bias [mm/day]')
fig.colorbar(ax = axs[0], mappable = im0, orientation = 'horizontal')
fig.colorbar(ax = axs[1], mappable = im1, orientation = 'horizontal')
fig.colorbar(ax = axs[2], mappable = im2, orientation = 'horizontal')

plt.plot(ds_bc['ASOS'][:,20,30], alpha = .3)
plt.plot(ds_bc['AWS'][:,20,30], alpha = .3)

eval_maps_AWS = get_eval_map(ds_bc['BC'],ds_bc['AWS'])
eval_maps_ASOS = get_eval_map(ds_bc['BC'],ds_bc['ASOS'])


def corr(predictions, targets):
    x = np.asarray(predictions)
    y = np.asarray(targets)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    return np.nanmean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std())

def ubrmsd(predictions, targets):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    mask = np.isfinite(predictions) & np.isfinite(targets)
    if mask.sum() < 2:
        return np.nan
    p = predictions[mask]
    t = targets[mask]
    return np.sqrt(np.nanmean(((p - np.nanmean(p)) - (t - np.nanmean(t))) ** 2))

def bias(predictions, targets):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    mask = np.isfinite(predictions) & np.isfinite(targets)
    if mask.sum() < 2:
        return np.nan
    return np.nanmean(predictions[mask] - targets[mask])

def corr_ubrmsd_bias(V1, V2):
    mask = np.isfinite(V1) & np.isfinite(V2)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan
    V1_masked = V1[mask]
    V2_masked = V2[mask]
    return corr(V1_masked, V2_masked), ubrmsd(V1_masked, V2_masked), bias(V1_masked, V2_masked)

def get_eval_map(da1, da2):
    corr_, ubrmsd_, bias_ = xr.apply_ufunc(
        lambda V1, V2: corr_ubrmsd_bias(V1, V2),
        da1,
        da2,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    return corr_, ubrmsd_, bias_

def add_metrics(x, y, ax=None, s=0, show_corr=True, show_ubRMSD=True, show_bias=True,
                show_line_eq=True, corner='top-right', color='black', alpha=0.2,
                fontsize=8, vminmax=None):

    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan

    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]

    corr_ = np.corrcoef(x, y)[0, 1]
    ubRMSD_ = ubrmsd(x, y)
    bias_ = np.nanmean(y - x)

    line = LinearRegression()
    line.fit(x.reshape(-1, 1), y.reshape(-1, 1))

    metrics = []
    if show_corr:
        metrics.append(r'$R=%.3f$' % corr_)
    if show_ubRMSD:
        metrics.append(r'$ubRMSD=%.3f$' % ubRMSD_)
    if show_bias:
        metrics.append(r'$bias=%.3f$' % bias_)
    if show_line_eq:
        metrics.append(r'$y = {:.4f}x {:+.3f}$'.format(line.coef_[0, 0], line.intercept_[0]))

    textstr = '\n'.join(metrics)

    corner_positions = {
        'top-right': (0.95, 0.95),
        'center-right': (0.95, 0.50),
        'top-left': (0.05, 0.95),
        'bottom-right': (0.95, 0.05),
        'bottom-left': (0.05, 0.05),
    }
    position = corner_positions.get(corner, (0.95, 0.95))

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(x, y, s=s, alpha=alpha, color=color)

    x_limits = ax.get_xlim()
    if vminmax is None:
        x_pred = np.linspace(x_limits[0], x_limits[1], 100)
    else:
        x_pred = np.linspace(vminmax[0], vminmax[1], 100)

    y_pred = line.predict(x_pred.reshape(-1, 1))
    ax.plot(x_pred, y_pred, color=color, alpha=0.8, linewidth=2)

    props = dict(boxstyle='round', facecolor='white', alpha=.9)

    ax.text(
        position[0], position[1], textstr,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment='top' if 'top' in corner else 'bottom',
        horizontalalignment='right' if 'right' in corner else 'left',
        bbox=props,
        color=color
    )

    return corr_, ubRMSD_, bias_
def get_eval_map(da1, da2):
    corr_, ubrmsd_, bias_ = xr.apply_ufunc(
        lambda V1, V2: corr_ubrmsd_bias(V1, V2),
        da1,
        da2,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    return corr_, ubrmsd_, bias_





















# y_pred = model.predict(X_test)

pred_train = lgbm_model.predict(df_train[FEATURES])
# pred_train = model2.predict(df_train[FEATURES])
rmse_train = np.sqrt(np.mean((pred_train - df_train['Y'].values) ** 2))
bias_train = (pred_train.sum() - df_train['Y'].values.sum()) / df_train['Y'].values.sum() * 100
print(f"  Train — RMSE : {rmse_train:.4f} mm/day  |  BIAS : {bias_train:+.2f}%")


# ============================================================
# BC-4. Validation 평가 (논문 eq.3, eq.4)
# ============================================================
print("\n" + "=" * 60)
print("BC-4. Validation 평가")
print("=" * 60)

pred_val = np.maximum(lgbm_model.predict(df_val[FEATURES]), 0.0)
y_val    = df_val['Y'].values
tca_val  = df_val['TCA'].values
sm_val   = df_val['SM2RAIN'].values
gpm_val  = df_val['GPM'].values
era5_val = df_val['ERA5'].values

def calc_rmse(p, o): return float(np.sqrt(np.mean((p - o) ** 2)))
def calc_bias(p, o): return float((p.sum() - o.sum()) / o.sum() * 100)

metrics = {
    'SM2RAIN':       (calc_rmse(sm_val,   y_val), calc_bias(sm_val,   y_val)),
    'GPM':           (calc_rmse(gpm_val,  y_val), calc_bias(gpm_val,  y_val)),
    'TCA Merged':    (calc_rmse(tca_val,  y_val), calc_bias(tca_val,  y_val)),
    'ERA5':          (calc_rmse(era5_val, y_val), calc_bias(era5_val, y_val)),
    'LGBM Corrected': (calc_rmse(pred_val, y_val), calc_bias(pred_val, y_val)),
}

print(f"\n  {'Method':<18} {'RMSE (mm/day)':>14}  {'BIAS (%)':>10}")
print(f"  {'-'*46}")
for name, (rmse, bias) in metrics.items():
    print(f"  {name:<18} {rmse:>14.4f}  {bias:>+10.2f}")

rmse_tca, bias_tca = metrics['TCA Merged']
rmse_lgbm, bias_lgbm = metrics['LGBM Corrected']


# ============================================================
# BC-5. SHAP Feature Importance
# ============================================================
print("\n" + "=" * 60)
print("BC-5. SHAP Feature Importance")
print("=" * 60)

MAX_SHAP  = 5_000
df_shap   = df_val.sample(min(MAX_SHAP, len(df_val)), random_state=42)
explainer = shap.TreeExplainer(lgbm_model)
shap_vals = explainer.shap_values(df_shap[FEATURES])

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
top_feature   = FEATURES[int(np.argmax(mean_abs_shap))]

print(f"  SHAP 계산 완료 ({len(df_shap):,} 샘플)")
for feat, ms in zip(FEATURES, mean_abs_shap):
    print(f"    {feat:<12} {ms:.4f}")
print(f"  → 최우선 변수 : {top_feature}")


# ============================================================
# BC-6. 공간 복원 (Validation 기간)
# ============================================================
print("\n" + "=" * 60)
print("BC-6. Validation 기간 공간 복원")
print("=" * 60)

T_val   = len(val_idx)
N_pix   = n_lat * n_lon
bc_flat = np.full((N_pix, T_val), np.nan)

sm_2d   = X_sm  [:, :, val_idx].reshape(-1, T_val)
gpm_2d  = X_gpm [:, :, val_idx].reshape(-1, T_val)
era5_2d  = X_era5[:, :, val_idx].reshape(-1, T_val)
asos_2d = X_asos[:, :, val_idx].reshape(-1, T_val)
tca_2d  = P_merged[:, :, val_idx].reshape(-1, T_val)

for px in tqdm(range(N_pix), desc="  공간 복원"):
    valid = (np.isfinite(sm_2d[px]) & np.isfinite(gpm_2d[px]) &
             np.isfinite(era5_2d[px]) & np.isfinite(tca_2d[px]))
    if valid.sum() == 0:
        continue
    X_pred = np.column_stack([sm_2d[px][valid], gpm_2d[px][valid],
                               asos_2d[px][valid], tca_2d[px][valid]])
    tmp = np.full(T_val, np.nan)
    tmp[valid] = np.maximum(lgbm_model.predict(X_pred), 0.0)
    bc_flat[px] = tmp

P_bc = bc_flat.reshape(n_lat, n_lon, T_val)
print(f"\n  복원 완료 : {P_bc.shape}")
print(f"  범위      : {np.nanmin(P_bc):.3f} ~ {np.nanmax(P_bc):.3f} mm/day")


# ============================================================
# BC-7. NetCDF 저장
# ============================================================
print("\n" + "=" * 60)
print("BC-7. NetCDF 저장")
print("=" * 60)

val_dates = common_dates[val_idx]

ds_bc

ds_bc = xr.Dataset(
    data_vars={
        "precipitation_bc": xr.DataArray(
            np.transpose(P_bc, (2, 0, 1)),
            dims=("time", "lat", "lon"),
            attrs={
                "long_name"   : "LightGBM bias-corrected TCA merged precipitation",
                "units"       : "mm/day",
                "method"      : "LightGBM regression (Li et al. 2023)",
                "features"    : ", ".join(FEATURES),
                "target"      : TARGET,
                "val_RMSE"    : float(rmse_lgbm),
                "val_BIAS_pct": float(bias_lgbm),
                "tca_RMSE"    : float(rmse_tca),
                "tca_BIAS_pct": float(bias_tca),
            }
        )
    },
    coords={
        "time": xr.DataArray(val_dates.values, dims="time"),
        "lat" : xr.DataArray(lat_1d, dims="lat",
                             attrs={"long_name": "latitude",  "units": "degrees_north"}),
        "lon" : xr.DataArray(lon_1d, dims="lon",
                             attrs={"long_name": "longitude", "units": "degrees_east"}),
    },
    attrs={
        "title" : "LightGBM Bias-Corrected Precipitation",
        "inputs": "SM2RAIN, GPM, ERA5, TCA_Merged",
        "period": f"{val_dates[0].date()} ~ {val_dates[-1].date()}",
    }
)
os.makedirs(os.path.dirname(OUT_BC_PATH), exist_ok=True)
ds_bc.to_netcdf(OUT_BC_PATH)
print(f"  저장 완료: {OUT_BC_PATH}")

# ============================================================
# BC-8. 시각화
# ============================================================
print("\n" + "=" * 60)
print("BC-8. 시각화")
print("=" * 60)
 
from scipy.stats import pearsonr
import matplotlib.lines as mlines
 
# ── 공통 색상/스타일 ──────────────────────────────────────────
COLORS = {
    'ASOS':    '#757575',
    'SM2RAIN': '#2196F3',
    'GPM':     '#FF9800',
    'TCA':     '#E53935',
    'ERA5':    '#27ae60',
    'BC':      '#9C27B0',
}
STYLE_TS = {
    'ASOS':    dict(color='#757575', lw=1.5, ls='-',              alpha=0.9,  zorder=5),
    'SM2RAIN': dict(color='#2196F3', lw=0.9, ls='--',             alpha=0.85, zorder=2),
    'GPM':     dict(color='#FF9800', lw=0.9, ls='-.',             alpha=0.85, zorder=2),
    'TCA':     dict(color='#E53935', lw=0.9, ls=':',              alpha=0.85, zorder=2),
    'ERA5':    dict(color='#27ae60', lw=0.9, ls=(0,(4,1,1,1)),   alpha=0.85, zorder=2),
    'BC':      dict(color='#9C27B0', lw=1.2, ls='-',              alpha=0.95, zorder=3),
}
STYLE_SC = {
    'SM2RAIN': dict(color='#2196F3', marker='o', s=35, alpha=0.55, zorder=2),
    'GPM':     dict(color='#FF9800', marker='^', s=35, alpha=0.55, zorder=2),
    'TCA':     dict(color='#E53935', marker='s', s=35, alpha=0.55, zorder=2),
    'ERA5':    dict(color='#27ae60', marker='D', s=35, alpha=0.55, zorder=2),
    'BC':      dict(color='#9C27B0', marker='<', s=40, alpha=0.70, zorder=3),
}
 
# ── Validation 기간 전체 픽셀 평균 시계열 ───────────────────
date_arr_val = common_dates[val_idx]
ts_asos = np.nanmean(X_asos  [:, :, val_idx].reshape(-1, T_val), axis=0)
ts_sm   = np.nanmean(X_sm    [:, :, val_idx].reshape(-1, T_val), axis=0)
ts_gpm  = np.nanmean(X_gpm   [:, :, val_idx].reshape(-1, T_val), axis=0)
ts_tca  = np.nanmean(P_merged[:, :, val_idx].reshape(-1, T_val), axis=0)
ts_era5 = np.nanmean(X_era5  [:, :, val_idx].reshape(-1, T_val), axis=0)
ts_bc   = np.nanmean(P_bc.reshape(-1, T_val), axis=0)
 
lim = float(max(np.nanpercentile(y_val,    99),
                np.nanpercentile(pred_val, 99),
                np.nanpercentile(tca_val,  99))) * 1.15
 
# ── (a) TCA vs ASOS scatter ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
ax.scatter(y_val, tca_val, s=6, alpha=0.25, color=COLORS['TCA'], rasterized=True)
ax.plot([0, lim], [0, lim], 'k--', lw=1.5, alpha=0.7)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('ASOS (mm/day)', fontsize=13)
ax.set_ylabel('TCA (mm/day)', fontsize=13)
ax.set_title(f'TCA vs ASOS\nBIAS = {bias_tca:+.2f}%   RMSE = {rmse_tca:.3f} mm/day',
             fontsize=14, fontweight='bold', pad=12)
ax.tick_params(labelsize=12)
ax.grid(alpha=0.25, ls='--', lw=0.6)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()
 
 
# ── (b) BC vs ASOS scatter ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
ax.scatter(y_val, pred_val, s=6, alpha=0.25, color=COLORS['BC'], rasterized=True)
ax.plot([0, lim], [0, lim], 'k--', lw=1.5, alpha=0.7)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('ASOS (mm/day)', fontsize=13)
ax.set_ylabel('BC (mm/day)', fontsize=13)
ax.set_title(f'BC vs ASOS\nBIAS = {bias_lgbm:+.2f}%   RMSE = {rmse_lgbm:.3f} mm/day',
             fontsize=14, fontweight='bold', pad=12)
ax.tick_params(labelsize=12)
ax.grid(alpha=0.25, ls='--', lw=0.6)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()
 
 
# ── (c) 빈도 분포 비교 ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
bins = np.linspace(0, lim, 60)
ax.hist(y_val,    bins=bins, alpha=0.60, color=COLORS['ASOS'],    label='ASOS (ref)', density=True)
ax.hist(sm_val,   bins=bins, alpha=0.45, color=COLORS['SM2RAIN'], label='SM2RAIN',    density=True)
ax.hist(gpm_val,  bins=bins, alpha=0.45, color=COLORS['GPM'],     label='GPM',        density=True)
ax.hist(tca_val,  bins=bins, alpha=0.45, color=COLORS['TCA'],     label='TCA',        density=True)
ax.hist(era5_val, bins=bins, alpha=0.45, color=COLORS['ERA5'],    label='ERA5',       density=True)
ax.hist(pred_val, bins=bins, alpha=0.55, color=COLORS['BC'],      label='BC',         density=True)
ax.set_xlabel('강수량 (mm/day)', fontsize=13)
ax.set_ylabel('Density', fontsize=13)
ax.set_title('빈도 분포 비교  (Validation 기간)', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=11, framealpha=0.9, ncol=3)
ax.tick_params(labelsize=12)
ax.grid(alpha=0.25, ls='--', lw=0.6)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()
 
 
# ── (d) SHAP Beeswarm ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
plt.sca(ax)
shap.summary_plot(shap_vals, df_shap[FEATURES],
                  plot_type='dot', show=False,
                  max_display=len(FEATURES), color_bar=True)
ax.set_title('SHAP Feature Importance  (Beeswarm)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('SHAP value  (impact on model output)', fontsize=12)
ax.tick_params(labelsize=12)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()
 
 
# ── (e) 성능 지표 Bar chart (5개 자료) ───────────────────────
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
method_names = ['SM2RAIN', 'GPM', 'ERA5', 'TCA Merged', 'LGBM Corrected']
bar_colors   = [COLORS['SM2RAIN'], COLORS['GPM'], COLORS['ERA5'], COLORS['TCA'], COLORS['BC']]
rmse_list = [metrics[k][0] for k in method_names]
bias_list = [abs(metrics[k][1]) for k in method_names]
 
x = np.arange(len(method_names)); w = 0.35
bars_r = ax.bar(x - w/2, rmse_list, width=w, label='RMSE (mm/day)',
                color=bar_colors, alpha=0.85, edgecolor='white')
bars_b = ax.bar(x + w/2, bias_list, width=w, label='|BIAS| (%)',
                color=bar_colors, alpha=0.50, edgecolor='white', hatch='//')
for bar in list(bars_r) + list(bars_b):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.08,
            f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(method_names, fontsize=11)
ax.set_title('성능 비교  (Validation  vs  ASOS)', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=11, framealpha=0.9)
ax.tick_params(labelsize=11)
ax.grid(axis='y', alpha=0.25, ls='--', lw=0.6)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()
 
 
# ── (f) 전체 영역 시계열 (5개 + ASOS) ──────────────────────
fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
ax.fill_between(date_arr_val, ts_asos, alpha=0.10, color=COLORS['ASOS'])
ax.plot(date_arr_val, ts_asos, label='ASOS (ref)', **STYLE_TS['ASOS'])
ax.plot(date_arr_val, ts_sm,   label='SM2RAIN',    **STYLE_TS['SM2RAIN'])
ax.plot(date_arr_val, ts_gpm,  label='GPM',        **STYLE_TS['GPM'])
ax.plot(date_arr_val, ts_tca,  label='TCA',        **STYLE_TS['TCA'])
ax.plot(date_arr_val, ts_era5, label='ERA5',       **STYLE_TS['ERA5'])
ax.plot(date_arr_val, ts_bc,   label='BC',         **STYLE_TS['BC'])
ax.set_ylabel('강수량 (mm/day)', fontsize=13)
ax.set_title('전체 영역 시계열  (Validation 기간 전체 픽셀 평균)',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=11, ncol=6, loc='upper right', framealpha=0.9, edgecolor='#ccc')
ax.tick_params(axis='x', rotation=20, labelsize=11)
ax.tick_params(axis='y', labelsize=12)
ax.grid(axis='y', alpha=0.2, ls='--', lw=0.6)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
plt.tight_layout(); plt.show()
 
 
# ── (g) 전체 영역 — 시계열 & 산점도 ─────────────────────────
def _gstats(p, r):
    ok = np.isfinite(p) & np.isfinite(r)
    if ok.sum() < 5: return np.nan, np.nan, np.nan
    diff = p[ok] - r[ok]
    bias = float(np.mean(diff))
    ubr  = float(np.sqrt(np.mean((diff - bias) ** 2)))
    R, _ = pearsonr(p[ok], r[ok])
    return float(R), ubr, bias
 
products_g = {'SM2RAIN': ts_sm, 'GPM': ts_gpm, 'TCA': ts_tca, 'ERA5': ts_era5, 'BC': ts_bc}
 
fig = plt.figure(figsize=(22, 5.5), dpi=150)
gs  = gridspec.GridSpec(1, 3, figure=fig,
                        left=0.05, right=0.97, top=0.82, bottom=0.16,
                        wspace=0.36, width_ratios=[3.2, 1.3, 1.3])
 
ax_ts = fig.add_subplot(gs[0, 0])
ax_ts.fill_between(date_arr_val, ts_asos, alpha=0.10, color=COLORS['ASOS'])
ax_ts.plot(date_arr_val, ts_asos, label='ASOS', **STYLE_TS['ASOS'])
for name, ts_p in products_g.items():
    ax_ts.plot(date_arr_val, ts_p, label=name, **STYLE_TS[name])
ax_ts.set_ylabel('강수량 (mm/day)', fontsize=11)
ax_ts.set_title('(a) 전체 영역 시계열  (Validation 기간 전체 픽셀 평균)',
                fontsize=11, fontweight='bold', pad=6)
ax_ts.tick_params(axis='x', rotation=25, labelsize=9)
ax_ts.tick_params(axis='y', labelsize=10)
ax_ts.xaxis.set_major_locator(mticker.MaxNLocator(7))
ax_ts.grid(axis='y', alpha=0.2, ls='--', lw=0.5)
ax_ts.spines[['top', 'right']].set_visible(False)
legend_ts_g = [
    mlines.Line2D([], [], color=STYLE_TS[k]['color'], lw=STYLE_TS[k]['lw'],
                  ls=STYLE_TS[k]['ls'], alpha=STYLE_TS[k]['alpha'], label=k)
    for k in ['ASOS'] + list(products_g.keys())
]
ax_ts.legend(handles=legend_ts_g, fontsize=8.5, ncol=6,
             loc='upper center', bbox_to_anchor=(0.5, 1.22),
             framealpha=0.95, edgecolor='#ccc', columnspacing=0.7)
 
ax_b  = fig.add_subplot(gs[0, 1])
lim_b = 0.0
for name, ts_p in products_g.items():
    ok = np.isfinite(ts_p) & np.isfinite(ts_asos)
    if ok.sum() < 5: continue
    ax_b.scatter(ts_asos[ok], ts_p[ok], **STYLE_SC[name])
    lim_b = max(lim_b, float(np.nanmax(ts_asos[ok])), float(np.nanmax(ts_p[ok])))
lim_b *= 1.10
ax_b.plot([0, lim_b], [0, lim_b], '--', color='gray', lw=1.0, alpha=0.7)
ax_b.set_xlim(0, lim_b); ax_b.set_ylim(0, lim_b)
ax_b.set_xlabel('ASOS (mm/day)', fontsize=10)
ax_b.set_ylabel('Estimated (mm/day)', fontsize=10)
ax_b.set_title('(b) 전체 영역 산점도\nvs ASOS', fontsize=11, fontweight='bold')
ax_b.tick_params(labelsize=10)
ax_b.grid(alpha=0.2, ls='--', lw=0.5)
ax_b.spines[['top', 'right']].set_visible(False)
y_txt = 0.97
for name, ts_p in products_g.items():
    R_, ubr_, bias_ = _gstats(ts_p, ts_asos)
    if not np.isfinite(R_): continue
    fc = STYLE_SC[name]['color']
    ax_b.text(0.97, y_txt, f"R={R_:.3f}\nubRMSD={ubr_:.3f}\nbias={bias_:+.3f}",
              transform=ax_b.transAxes, fontsize=7, va='top', ha='right',
              color=fc, fontweight='bold',
              bbox=dict(fc='white', alpha=0.70, ec=fc, lw=0.8, boxstyle='round,pad=0.25'))
    y_txt -= 0.20
legend_sc_g = [
    mlines.Line2D([], [], linestyle='None', marker=STYLE_SC[k]['marker'],
                  color=STYLE_SC[k]['color'], markersize=7, alpha=0.8, label=k)
    for k in STYLE_SC
] + [mlines.Line2D([], [], color='gray', lw=1.0, ls='--', label='1:1 line')]
ax_b.legend(handles=legend_sc_g, fontsize=7.5, loc='upper left',
            framealpha=0.9, edgecolor='#ccc', handletextpad=0.4)
 
ax_c  = fig.add_subplot(gs[0, 2])
ok_c  = np.isfinite(ts_bc) & np.isfinite(ts_asos)
if ok_c.sum() >= 5:
    ax_c.scatter(ts_asos[ok_c], ts_bc[ok_c], **STYLE_SC['BC'])
lim_c = (max(float(np.nanmax(ts_asos[ok_c])),
             float(np.nanmax(ts_bc[ok_c]))) * 1.10) if ok_c.sum() else 1.0
ax_c.plot([0, lim_c], [0, lim_c], '--', color='gray', lw=1.0, alpha=0.7)
ax_c.set_xlim(0, lim_c); ax_c.set_ylim(0, lim_c)
ax_c.set_xlabel('ASOS (mm/day)', fontsize=10)
ax_c.set_ylabel('BC (mm/day)', fontsize=10)
ax_c.set_title('(c) BC 단독 산점도\nvs ASOS', fontsize=11, fontweight='bold')
R_c, ubr_c, bias_c = _gstats(ts_bc, ts_asos)
ax_c.text(0.05, 0.95, f"R = {R_c:.3f}\nubRMSD = {ubr_c:.3f}\nbias = {bias_c:+.3f}",
          transform=ax_c.transAxes, fontsize=10, va='top',
          color=COLORS['BC'], fontweight='bold',
          bbox=dict(fc='white', alpha=0.85, ec=COLORS['BC'], lw=1.0, boxstyle='round,pad=0.35'))
ax_c.tick_params(labelsize=10)
ax_c.grid(alpha=0.2, ls='--', lw=0.5)
ax_c.spines[['top', 'right']].set_visible(False)
fig.suptitle('전체 영역  —  시계열 & 산점도  (기준: ASOS)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
 
 
# ============================================================
# 유역 정의 및 헬퍼 함수
# ============================================================
BASINS = {
    '한강':  (36.8, 38.5, 126.0, 129.0),
    '낙동강': (35.0, 37.0, 127.5, 129.5),
    '금강':  (35.8, 37.2, 126.3, 128.0),
    '영산강': (34.5, 35.8, 126.3, 127.5),
}
 
def _bmask(lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max):
    return ((lat_1d >= lat_min) & (lat_1d <= lat_max))[:,None] & \
           ((lon_1d >= lon_min) & (lon_1d <= lon_max))[None,:]
 
def _bmean(arr_3d, mask_2d):
    ii, jj = np.where(mask_2d)
    if len(ii) == 0: return np.full(arr_3d.shape[2], np.nan)
    return np.nanmean(arr_3d[ii, jj, :], axis=0)
 
def _bstats(p, r):
    ok = np.isfinite(p) & np.isfinite(r)
    if ok.sum() < 5: return np.nan, np.nan, np.nan
    diff = p[ok] - r[ok]
    bias = float(np.mean(diff))
    ubr  = float(np.sqrt(np.mean((diff - bias) ** 2)))
    R, _ = pearsonr(p[ok], r[ok])
    return float(R), ubr, bias
 
 # ============================================================
# F-4A. 유역별 시계열 + 산점도 (5개 자료 전체)
# ============================================================
print("\n[F-4A] 유역별 시계열 + 산점도 (5개 자료)...")

legend_ts_b = [
    mlines.Line2D([], [], color=STYLE_TS[k]['color'], lw=STYLE_TS[k]['lw'],
                  ls=STYLE_TS[k]['ls'], alpha=STYLE_TS[k]['alpha'], label=k)
    for k in ['ASOS', 'SM2RAIN', 'GPM', 'TCA', 'ERA5', 'BC']
]
legend_sc_b = [
    mlines.Line2D([], [], linestyle='None', marker=STYLE_SC[k]['marker'],
                  color=STYLE_SC[k]['color'], markersize=10, alpha=0.8, label=k)
    for k in STYLE_SC
] + [mlines.Line2D([], [], color='gray', lw=1.0, ls='--', label='1:1 line')]

for basin_name, (lat_min, lat_max, lon_min, lon_max) in BASINS.items():
    bmask = _bmask(lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max)
    n_pix = int(bmask.sum())

    ts_asos_b = _bmean(X_asos  [:, :, val_idx], bmask)
    ts_sm_b   = _bmean(X_sm    [:, :, val_idx], bmask)
    ts_gpm_b  = _bmean(X_gpm   [:, :, val_idx], bmask)
    ts_tca_b  = _bmean(P_merged[:, :, val_idx], bmask)
    ts_era5_b = _bmean(X_era5  [:, :, val_idx], bmask)
    ts_bc_b   = _bmean(P_bc,                    bmask)

    products_b = {'SM2RAIN': ts_sm_b, 'GPM': ts_gpm_b, 'TCA': ts_tca_b,
                  'ERA5': ts_era5_b, 'BC': ts_bc_b}

    fig = plt.figure(figsize=(44, 11), dpi=150)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.05, right=0.97, top=0.82, bottom=0.16,
                            wspace=0.36, width_ratios=[3.2, 1.3, 1.3])

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.fill_between(date_arr_val, ts_asos_b, alpha=0.10, color=COLORS['ASOS'])
    ax_ts.plot(date_arr_val, ts_asos_b, label='ASOS', **STYLE_TS['ASOS'])
    for name, ts_p in products_b.items():
        ax_ts.plot(date_arr_val, ts_p, label=name, **STYLE_TS[name])
    ax_ts.set_ylabel('강수량 (mm/day)', fontsize=14)
    ax_ts.set_title(f' {basin_name}  —  시계열  (유역 내 {n_pix}픽셀 평균)',
                    fontsize=14, fontweight='bold', pad=6)
    ax_ts.tick_params(axis='x', rotation=25, labelsize=12)
    ax_ts.tick_params(axis='y', labelsize=12)
    ax_ts.xaxis.set_major_locator(mticker.MaxNLocator(7))
    ax_ts.grid(axis='y', alpha=0.2, ls='--', lw=0.5)
    ax_ts.spines[['top', 'right']].set_visible(False)
    ax_ts.legend(handles=legend_ts_b, fontsize=11, ncol=6,
                 loc='upper center', bbox_to_anchor=(0.5, 1.22),
                 framealpha=0.95, edgecolor='#ccc', columnspacing=0.7)

    ax_b = fig.add_subplot(gs[0, 1])
    lim_b = 0.0
    for name, ts_p in products_b.items():
        ok = np.isfinite(ts_p) & np.isfinite(ts_asos_b)
        if ok.sum() < 5: continue
        ax_b.scatter(ts_asos_b[ok], ts_p[ok], **STYLE_SC[name])
        lim_b = max(lim_b, float(np.nanmax(ts_asos_b[ok])), float(np.nanmax(ts_p[ok])))
    lim_b *= 1.10
    ax_b.plot([0, lim_b], [0, lim_b], '--', color='gray', lw=1.0, alpha=0.7)
    ax_b.set_xlim(0, lim_b); ax_b.set_ylim(0, lim_b)
    ax_b.set_xlabel('ASOS (mm/day)', fontsize=13)
    ax_b.set_ylabel('Estimated (mm/day)', fontsize=13)
    ax_b.set_title(f'{basin_name}\nvs ASOS', fontsize=13, fontweight='bold')
    ax_b.tick_params(labelsize=12)
    ax_b.grid(alpha=0.2, ls='--', lw=0.5)
    ax_b.spines[['top', 'right']].set_visible(False)
    y_txt = 0.97
    for name, ts_p in products_b.items():
        R_, ubr_, bias_ = _bstats(ts_p, ts_asos_b)
        if not np.isfinite(R_): continue
        fc = STYLE_SC[name]['color']
        ax_b.text(0.97, y_txt, f"R={R_:.3f}\nubRMSD={ubr_:.3f}\nbias={bias_:+.3f}",
                  transform=ax_b.transAxes, fontsize=9.5, va='top', ha='right',
                  color=fc, fontweight='bold',
                  bbox=dict(fc='white', alpha=0.70, ec=fc, lw=0.8, boxstyle='round,pad=0.25'))
        y_txt -= 0.20
    ax_b.legend(handles=legend_sc_b, fontsize=10, loc='upper left',
                framealpha=0.9, edgecolor='#ccc', handletextpad=0.4)

    ax_c = fig.add_subplot(gs[0, 2])
    ok_c = np.isfinite(ts_bc_b) & np.isfinite(ts_asos_b)
    if ok_c.sum() >= 5:
        ax_c.scatter(ts_asos_b[ok_c], ts_bc_b[ok_c], **STYLE_SC['BC'])
    lim_c = (max(float(np.nanmax(ts_asos_b[ok_c])),
                 float(np.nanmax(ts_bc_b[ok_c]))) * 1.10) if ok_c.sum() else 1.0
    ax_c.plot([0, lim_c], [0, lim_c], '--', color='gray', lw=1.0, alpha=0.7)
    ax_c.set_xlim(0, lim_c); ax_c.set_ylim(0, lim_c)
    ax_c.set_xlabel('ASOS (mm/day)', fontsize=13)
    ax_c.set_ylabel('BC (mm/day)', fontsize=13)
    ax_c.set_title(' BC 단독 산점도\nvs ASOS', fontsize=13, fontweight='bold')
    R_c, ubr_c, bias_c = _bstats(ts_bc_b, ts_asos_b)
    ax_c.text(0.05, 0.95, f"R = {R_c:.3f}\nubRMSD = {ubr_c:.3f}\nbias = {bias_c:+.3f}",
              transform=ax_c.transAxes, fontsize=12, va='top',
              color=COLORS['BC'], fontweight='bold',
              bbox=dict(fc='white', alpha=0.85, ec=COLORS['BC'], lw=1.0, boxstyle='round,pad=0.35'))
    ax_c.tick_params(labelsize=12)
    ax_c.grid(alpha=0.2, ls='--', lw=0.5)
    ax_c.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{basin_name}  —  시계열 & 산점도  (기준: ASOS)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(); plt.show()
# ============================================================
# F-4B. 유역별 TCA vs BC 비교 (2개만)
# ============================================================
print("\n[F-4B] 유역별 TCA vs BC 비교...")
 
STYLE_TS_2 = {
    'ASOS': dict(color='#757575', lw=1.5, ls='-',  alpha=0.9,  zorder=5),
    'TCA':  dict(color='#E53935', lw=0.9, ls='--', alpha=0.85, zorder=2),
    'BC':   dict(color='#4CAF50', lw=1.2, ls='-',  alpha=0.95, zorder=3),
}
STYLE_SC_2 = {
    'TCA': dict(color='#E53935', marker='s', s=18, alpha=0.55, zorder=2),
    'BC':  dict(color='#4CAF50', marker='<', s=22, alpha=0.70, zorder=3),
}
legend_ts_2 = [
    mlines.Line2D([], [], color=STYLE_TS_2[k]['color'], lw=STYLE_TS_2[k]['lw'],
                  ls=STYLE_TS_2[k]['ls'], alpha=STYLE_TS_2[k]['alpha'], label=k)
    for k in STYLE_TS_2
]
legend_sc_2 = [
    mlines.Line2D([], [], linestyle='None', marker=STYLE_SC_2[k]['marker'],
                  color=STYLE_SC_2[k]['color'], markersize=7, alpha=0.8, label=k)
    for k in STYLE_SC_2
] + [mlines.Line2D([], [], color='gray', lw=1.0, ls='--', label='1:1 line')]
 
for basin_name, (lat_min, lat_max, lon_min, lon_max) in BASINS.items():
    bmask = _bmask(lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max)
    n_pix = int(bmask.sum())
 
    ts_asos_b = _bmean(X_asos  [:, :, val_idx], bmask)
    ts_tca_b  = _bmean(P_merged[:, :, val_idx], bmask)
    ts_bc_b   = _bmean(P_bc,                    bmask)
    products_b2 = {'TCA': ts_tca_b, 'BC': ts_bc_b}
 
    fig = plt.figure(figsize=(21, 5.5), dpi=150)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.05, right=0.97, top=0.82, bottom=0.16,
                            wspace=0.36, width_ratios=[3.2, 1.3, 1.3])
 
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.fill_between(date_arr_val, ts_asos_b, alpha=0.15, color='#757575')
    ax_ts.plot(date_arr_val, ts_asos_b, **STYLE_TS_2['ASOS'])
    ax_ts.plot(date_arr_val, ts_tca_b,  **STYLE_TS_2['TCA'])
    ax_ts.plot(date_arr_val, ts_bc_b,   **STYLE_TS_2['BC'])
    ax_ts.set_ylabel('강수량 (mm/day)', fontsize=11)
    ax_ts.set_title(f'(a) {basin_name}  —  시계열  (유역 내 {n_pix}픽셀 평균)',
                    fontsize=11, fontweight='bold', pad=6)
    ax_ts.tick_params(axis='x', rotation=25, labelsize=9)
    ax_ts.tick_params(axis='y', labelsize=10)
    ax_ts.xaxis.set_major_locator(mticker.MaxNLocator(7))
    ax_ts.grid(axis='y', alpha=0.2, ls='--', lw=0.5)
    ax_ts.spines[['top', 'right']].set_visible(False)
    ax_ts.legend(handles=legend_ts_2, fontsize=9, ncol=3,
                 loc='upper center', bbox_to_anchor=(0.5, 1.22),
                 framealpha=0.95, edgecolor='#ccc', columnspacing=0.8)
 
    ax_b = fig.add_subplot(gs[0, 1])
    lim_b = 0.0
    for name, ts_p in products_b2.items():
        ok = np.isfinite(ts_p) & np.isfinite(ts_asos_b)
        if ok.sum() < 5: continue
        ax_b.scatter(ts_asos_b[ok], ts_p[ok], **STYLE_SC_2[name])
        lim_b = max(lim_b, float(np.nanmax(ts_asos_b[ok])), float(np.nanmax(ts_p[ok])))
    lim_b *= 1.10
    ax_b.plot([0, lim_b], [0, lim_b], '--', color='gray', lw=1.0, alpha=0.7)
    ax_b.set_xlim(0, lim_b); ax_b.set_ylim(0, lim_b)
    ax_b.set_xlabel('ASOS (mm/day)', fontsize=10)
    ax_b.set_ylabel('Estimated (mm/day)', fontsize=10)
    ax_b.set_title(f'(b) {basin_name}\nvs ASOS', fontsize=11, fontweight='bold')
    ax_b.tick_params(labelsize=10)
    ax_b.grid(alpha=0.2, ls='--', lw=0.5)
    ax_b.spines[['top', 'right']].set_visible(False)
    y_txt = 0.97
    for name, ts_p in products_b2.items():
        R_, ubr_, bias_ = _bstats(ts_p, ts_asos_b)
        if not np.isfinite(R_): continue
        fc = STYLE_SC_2[name]['color']
        ax_b.text(0.97, y_txt, f"R={R_:.3f}\nubRMSD={ubr_:.3f}\nbias={bias_:+.3f}",
                  transform=ax_b.transAxes, fontsize=8, va='top', ha='right',
                  color=fc, fontweight='bold',
                  bbox=dict(fc='white', alpha=0.70, ec=fc, lw=0.8, boxstyle='round,pad=0.25'))
        y_txt -= 0.32
    ax_b.legend(handles=legend_sc_2, fontsize=8, loc='upper left',
                framealpha=0.9, edgecolor='#ccc', handletextpad=0.4)
 
    ax_c = fig.add_subplot(gs[0, 2])
    ok_c = np.isfinite(ts_bc_b) & np.isfinite(ts_asos_b)
    if ok_c.sum() >= 5:
        ax_c.scatter(ts_asos_b[ok_c], ts_bc_b[ok_c], **STYLE_SC_2['BC'])
    lim_c = (max(float(np.nanmax(ts_asos_b[ok_c])),
                 float(np.nanmax(ts_bc_b[ok_c]))) * 1.10) if ok_c.sum() else 1.0
    ax_c.plot([0, lim_c], [0, lim_c], '--', color='gray', lw=1.0, alpha=0.7)
    ax_c.set_xlim(0, lim_c); ax_c.set_ylim(0, lim_c)
    ax_c.set_xlabel('ASOS (mm/day)', fontsize=10)
    ax_c.set_ylabel('BC (mm/day)', fontsize=10)
    ax_c.set_title('(c) BC 단독 산점도\nvs ASOS', fontsize=11, fontweight='bold')
    R_c, ubr_c, bias_c = _bstats(ts_bc_b, ts_asos_b)
    ax_c.text(0.05, 0.95, f"R = {R_c:.3f}\nubRMSD = {ubr_c:.3f}\nbias = {bias_c:+.3f}",
              transform=ax_c.transAxes, fontsize=10, va='top',
              color=COLORS['BC'], fontweight='bold',
              bbox=dict(fc='white', alpha=0.85, ec=COLORS['BC'], lw=1.0, boxstyle='round,pad=0.35'))
    ax_c.tick_params(labelsize=10)
    ax_c.grid(alpha=0.2, ls='--', lw=0.5)
    ax_c.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{basin_name}  —  TCA vs BC  (기준: ASOS)', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()
 
 
# ============================================================
# 최종 요약
# ============================================================
print("\n" + "=" * 60)
print("✅  Bias Correction 완료")
print("=" * 60)
print(f"  Train : {common_dates[train_idx[0]].date()} ~ {common_dates[train_idx[-1]].date()}")
print(f"  Val   : {common_dates[val_idx[0]].date()} ~ {common_dates[val_idx[-1]].date()}")
print()
print(f"  {'Method':<18} {'RMSE (mm/day)':>14}  {'BIAS (%)':>10}")
print(f"  {'-'*46}")
for name, (rmse, bias) in metrics.items():
    print(f"  {name:<18} {rmse:>14.4f}  {bias:>+10.2f}")
print()
print(f"  Best params      : {best_params}")
print(f"  SHAP top feature : {top_feature}")
print(f"\n  저장 파일: {OUT_BC_PATH}")
print("=" * 60)
 
 
########################################################
# FA ~ FE  추가 분석 그림
########################################################
import cartopy.crs as ccrs
import cartopy.feature as cfeature
 
RAIN_THRESH = 1.0  # mm/day
 
# ── 공통 자료 딕셔너리 (validation 기간) ──────────────────
REF = X_asos[:, :, val_idx]
PRODUCTS = {
    'SM2RAIN': X_sm  [:, :, val_idx],
    'GPM':     X_gpm [:, :, val_idx],
    'TCA':     P_merged[:, :, val_idx],
    'ERA5':    X_era5[:, :, val_idx],
    'BC':      P_bc,
}
 
# ── Cartopy 지도 헬퍼 ──────────────────────────────────────
def _edges(c):
    h = np.diff(c)/2.0; e = np.empty(len(c)+1)
    e[1:-1] = c[:-1]+h; e[0] = c[0]-h[0]; e[-1] = c[-1]+h[-1]
    return e
 
def _pcm(ax, lon_1d, lat_1d, data, **kw):
    lon2d, lat2d = np.meshgrid(_edges(lon_1d), _edges(lat_1d))
    return ax.pcolormesh(lon2d, lat2d, data, shading='flat',
                         transform=ccrs.PlateCarree(), zorder=2, **kw)
 
def _map_ax(ax, margin=(0.35, 0.22)):
    ax.set_extent([lon_1d.min()-margin[0], lon_1d.max()+margin[0],
                   lat_1d.min()-margin[1], lat_1d.max()+margin[1]],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),     facecolor='#ddeeff', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'),      facecolor='#f7f7f2', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='#333', lw=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'),   edgecolor='#888',
                   lw=0.5, linestyle=':', zorder=3)
    gl = ax.gridlines(draw_labels=True, linestyle='--', lw=0.35, alpha=0.5,
                      x_inline=False, y_inline=False)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}
    return ax
 
 
# ============================================================
# F-A. R² Heatmap (유역 × 자료)
# ============================================================
print("\n[F-A] R² Heatmap...")
 
prod_names  = ['SM2RAIN', 'GPM', 'TCA', 'ERA5', 'BC']
basin_names = list(BASINS.keys())
r2_matrix   = np.full((len(basin_names), len(prod_names)), np.nan)
 
def _fa_stats(p, r):
    ok = np.isfinite(p) & np.isfinite(r)
    if ok.sum() < 5: return np.nan
    R, _ = pearsonr(p[ok], r[ok])
    return float(R)**2
 
for bi, (bname, (lat_min, lat_max, lon_min, lon_max)) in enumerate(BASINS.items()):
    bmask  = _bmask(lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max)
    ts_ref = _bmean(REF, bmask)
    for pi, pname in enumerate(prod_names):
        ts_p = _bmean(PRODUCTS[pname], bmask)
        r2_matrix[bi, pi] = _fa_stats(ts_p, ts_ref)
 
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
im = ax.imshow(r2_matrix, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
for bi in range(len(basin_names)):
    for pi in range(len(prod_names)):
        val = r2_matrix[bi, pi]
        txt = f"{val:.2f}" if np.isfinite(val) else "N/A"
        ax.text(pi, bi, txt, ha='center', va='center',
                fontsize=13, fontweight='bold',
                color='white' if val > 0.6 else '#333')
ax.set_xticks(range(len(prod_names)));   ax.set_xticklabels(prod_names,  fontsize=12)
ax.set_yticks(range(len(basin_names)));  ax.set_yticklabels(basin_names, fontsize=12)
cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
cb.set_label('R²', fontsize=12)
ax.set_title('R² by Basin and Rainfall Input', fontsize=14, fontweight='bold', pad=10)
plt.tight_layout(); plt.show()
 
 
# ============================================================
# F-B. R² vs Bias Scatter (유역 × 자료)
# ============================================================
print("\n[F-B] R² vs Bias Scatter...")
 
MARKER_FB = {'SM2RAIN': 'o', 'GPM': 'v', 'TCA': 's', 'ERA5': 'D', 'BC': '^'}
COLOR_B = {'한강': '#2196F3', '낙동강': '#E53935', '금강': '#FF9800', '영산강': '#9C27B0'}
 
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
for bi, (bname, (lat_min, lat_max, lon_min, lon_max)) in enumerate(BASINS.items()):
    bmask  = _bmask(lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max)
    ts_ref = _bmean(REF, bmask)
    for pname in prod_names:
        ts_p = _bmean(PRODUCTS[pname], bmask)
        r2 = _fa_stats(ts_p, ts_ref)
        ok = np.isfinite(ts_p) & np.isfinite(ts_ref)
        if not np.isfinite(r2) or ok.sum() < 5: continue
        diff = ts_p[ok] - ts_ref[ok]
        bias_v = abs(float(np.mean(diff)))
        ax.scatter(r2, bias_v, marker=MARKER_FB[pname], s=120,
                   color=COLOR_B[bname], edgecolors='white', lw=0.8, zorder=3, alpha=0.85)
        ax.text(r2 + 0.01, bias_v + 0.1, pname,
                fontsize=7, color=COLOR_B[bname], alpha=0.85)
ax.axvline(0.5, color='gray', lw=1.0, ls='--', alpha=0.6)
ax.axhline(2.0, color='gray', lw=1.0, ls='--', alpha=0.6)
basin_handles = [mlines.Line2D([], [], linestyle='None', marker='o',
                               color=COLOR_B[b], markersize=9, label=b)
                 for b in basin_names]
prod_handles  = [mlines.Line2D([], [], linestyle='None', marker=MARKER_FB[p],
                               color='gray', markersize=9, label=p)
                 for p in prod_names]
leg1 = ax.legend(handles=basin_handles, title='유역', fontsize=9,
                 loc='upper left', framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=prod_handles, title='자료', fontsize=9,
          loc='upper right', framealpha=0.9)
ax.set_xlabel('R²', fontsize=12)
ax.set_ylabel('|Bias| (mm/day)', fontsize=12)
ax.set_xlim(0, 1.05); ax.set_ylim(bottom=0)
ax.set_title('R² vs |Bias|  (유역별·자료별)', fontsize=14, fontweight='bold', pad=10)
ax.grid(alpha=0.2, ls='--', lw=0.5)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()
 
 
# ============================================================
# F-C. BC 연평균 강수량 지도
# ============================================================
print("\n[F-C] BC 연평균 강수량 지도...")
 
val_dates = common_dates[val_idx]
mask_2024 = val_dates.year == 2024
 
if mask_2024.sum() == 0:
    print("  [경고] validation 기간에 2024년 데이터 없음")
else:
    P_bc_2024      = P_bc[:, :, mask_2024]
    annual_mean_bc = np.nanmean(P_bc_2024, axis=2) * 365
 
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150,
                           subplot_kw={'projection': ccrs.PlateCarree()})
    _map_ax(ax)
    vmax = np.nanpercentile(annual_mean_bc, 98)
    im   = _pcm(ax, lon_1d, lat_1d, annual_mean_bc, cmap='YlGnBu', vmin=0, vmax=vmax)
    cb   = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.06, shrink=0.85)
    cb.set_label('연강수량 (mm/year)', fontsize=11)
    ax.set_title('BC  연평균 강수량  (2024년)', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout(); plt.show()
 
 
# ============================================================
# F-D. POD / FAR / CSI 지도
# ============================================================
print("\n[F-D] POD / FAR / CSI 지도...")
 
def detection_skill(pred, ref, thresh=RAIN_THRESH):
    nl, nx = pred.shape[:2]
    t   = min(pred.shape[2], ref.shape[2])
    p_  = pred[:, :, :t]; r_ = ref[:, :, :t]
    POD = np.full((nl, nx), np.nan)
    FAR = np.full((nl, nx), np.nan)
    CSI = np.full((nl, nx), np.nan)
    for i in range(nl):
        for j in range(nx):
            p = p_[i, j, :]; r = r_[i, j, :]
            ok = np.isfinite(p) & np.isfinite(r)
            if ok.sum() < 10: continue
            pv = p[ok]; rv = r[ok]
            A = np.sum((pv >= thresh) & (rv >= thresh))
            B = np.sum((pv >= thresh) & (rv <  thresh))
            C = np.sum((pv <  thresh) & (rv >= thresh))
            if (A + C) > 0: POD[i, j] = A / (A + C)
            if (A + B) > 0: FAR[i, j] = B / (A + B)
            if (A + B + C) > 0: CSI[i, j] = A / (A + B + C)
    return POD, FAR, CSI
 
det_products = {'SM2RAIN': PRODUCTS['SM2RAIN'], 'TCA': PRODUCTS['TCA'], 'BC': PRODUCTS['BC']}
skill_results = {}
for pname, arr in det_products.items():
    POD, FAR, CSI = detection_skill(arr, REF)
    skill_results[pname] = {'POD': POD, 'FAR': FAR, 'CSI': CSI}
    print(f"  {pname:10s}: POD={np.nanmean(POD):.3f}  FAR={np.nanmean(FAR):.3f}  CSI={np.nanmean(CSI):.3f}")
 
skill_names  = ['POD', 'FAR', 'CSI']
skill_labels = ['POD (탐지율)', 'FAR (오탐율)', 'CSI (임계성공지수)']
 
fig, axes = plt.subplots(3, 3, figsize=(18, 16), dpi=150,
                         subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(left=0.04, right=0.92, top=0.93,
                    bottom=0.04, wspace=0.08, hspace=0.25)
for row, pname in enumerate(det_products.keys()):
    for col, (skey, slabel) in enumerate(zip(skill_names, skill_labels)):
        ax   = axes[row, col]
        data = skill_results[pname][skey]
        _map_ax(ax)
        im   = _pcm(ax, lon_1d, lat_1d, data, cmap='YlGn', vmin=0, vmax=1)
        mean_v = np.nanmean(data)
        ax.set_title(f'{pname} — {slabel}\nmean = {mean_v:.3f}',
                     fontsize=10, fontweight='bold', pad=4)
        if col == 2:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08, shrink=0.85)
            cb.ax.tick_params(labelsize=8)
fig.suptitle(f'강수 탐지 스킬  (임계값 {RAIN_THRESH} mm/day)  —  SM2RAIN / TCA / BC',
             fontsize=14, fontweight='bold')
plt.show()
 
 
# ============================================================
# F-E. Daily Precip Frequency (PDF) — Log scale
# ============================================================
print("\n[F-E] Daily Precip Frequency PDF...")
 
from scipy.stats import gaussian_kde
 
ts_asos_all = X_asos  [:, :, val_idx].flatten()
ts_gpm_all  = X_gpm   [:, :, val_idx].flatten()
ts_tca_all  = P_merged[:, :, val_idx].flatten()
ts_bc_all   = P_bc.flatten()
ts_era5_all = X_era5  [:, :, val_idx].flatten()
ts_sm_all   = X_sm    [:, :, val_idx].flatten()
 
THRESH = 1.0
 
def make_pdf(data, bins):
    d = data[np.isfinite(data) & (data >= THRESH)]
    counts, _ = np.histogram(d, bins=bins, density=True)
    return counts * 100
 
bins        = np.linspace(1, 105, 60)
bin_centers = (bins[:-1] + bins[1:]) / 2
 
pdf_asos = make_pdf(ts_asos_all, bins)
pdf_gpm  = make_pdf(ts_gpm_all,  bins)
pdf_tca  = make_pdf(ts_tca_all,  bins)
pdf_era5 = make_pdf(ts_era5_all, bins)
pdf_sm   = make_pdf(ts_sm_all,   bins)
pdf_bc   = make_pdf(ts_bc_all,   bins)
 
# 강수일 평균
def _wet_mean(d):
    v = d[np.isfinite(d) & (d >= THRESH)]
    return float(np.mean(v)) if len(v) > 0 else 0.0
 
mean_asos = _wet_mean(ts_asos_all)
mean_gpm  = _wet_mean(ts_gpm_all)
mean_tca  = _wet_mean(ts_tca_all)
mean_era5 = _wet_mean(ts_era5_all)
mean_sm   = _wet_mean(ts_sm_all)
mean_bc   = _wet_mean(ts_bc_all)
 
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.plot(bin_centers, pdf_asos, color='black',         lw=2.0, ls='-',
        label=f'ASOS ({mean_asos:.2f} mm/day)',     zorder=5)
ax.plot(bin_centers, pdf_gpm,  color=COLORS['GPM'],   lw=1.5, ls='--',
        label=f'GPM ({mean_gpm:.2f} mm/day)',       zorder=3)
ax.plot(bin_centers, pdf_tca,  color=COLORS['TCA'],   lw=1.5, ls='-.',
        label=f'TCA ({mean_tca:.2f} mm/day)',       zorder=3)
ax.plot(bin_centers, pdf_era5, color=COLORS['ERA5'],  lw=1.5, ls=':',
        label=f'ERA5 ({mean_era5:.2f} mm/day)',     zorder=3)
ax.plot(bin_centers, pdf_sm,   color=COLORS['SM2RAIN'], lw=1.5, ls='--',
        label=f'SM2RAIN ({mean_sm:.2f} mm/day)',    zorder=3)
ax.plot(bin_centers, pdf_bc,   color=COLORS['BC'],    lw=1.8, ls='-',
        label=f'BC ({mean_bc:.2f} mm/day)',         zorder=4)
ax.set_yscale('log')
ax.set_xlim(1, 105)
ax.set_ylim(1e-5, 10)
ax.set_xlabel('Rain Rate (mm/day)', fontsize=12)
ax.set_ylabel('PDF (%)', fontsize=12)
ax.set_title('Daily Precip Frequency\n(Validation 기간, 강수일 ≥ 1 mm/day)',
             fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=10, framealpha=0.9, edgecolor='#ccc')
ax.grid(alpha=0.2, ls='--', lw=0.5, which='both')
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(labelsize=10)
plt.tight_layout(); plt.show()
 
 
# ============================================================
# F-B3. 유역별 산점도 (유역당 1장, 1×5 패널)
# ============================================================
print("\n[F-B3] 유역별 산점도 (1×5)...")
 
PRODUCTS_SC = {
    'SM2RAIN': (X_sm    [:, :, val_idx], COLORS['SM2RAIN'], 'o'),
    'GPM':     (X_gpm   [:, :, val_idx], COLORS['GPM'],     '^'),
    'TCA':     (P_merged[:, :, val_idx], COLORS['TCA'],     's'),
    'ERA5':    (X_era5  [:, :, val_idx], COLORS['ERA5'],    'D'),
    'BC':      (P_bc,                    COLORS['BC'],       '<'),
}
N_SAMPLE = 8000
 
for basin_name, (lat_min, lat_max, lon_min, lon_max) in BASINS.items():
    bmask = _bmask(lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max)
    n_pix = int(bmask.sum())
    ii, jj = np.where(bmask)
 
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=150)
 
    ref_basin = REF[ii, jj, :].flatten()
 
    for idx, (pname, (arr, color, marker)) in enumerate(PRODUCTS_SC.items()):
        ax      = axes[idx]
        p_basin = arr[ii, jj, :].flatten()
        ok      = np.isfinite(p_basin) & np.isfinite(ref_basin)
        p_ok    = p_basin[ok]; r_ok = ref_basin[ok]
 
        n    = min(N_SAMPLE, len(p_ok))
        sidx = np.random.choice(len(p_ok), n, replace=False)
        ax.scatter(r_ok[sidx], p_ok[sidx], s=8, alpha=0.35, color=color,
                   marker=marker, rasterized=True, zorder=2)
 
        lim_p = max(float(np.nanpercentile(r_ok, 99)),
                    float(np.nanpercentile(p_ok, 99))) * 1.10
        ax.plot([0, lim_p], [0, lim_p], '--', color='gray', lw=1.2, alpha=0.7)
        ax.set_xlim(0, lim_p); ax.set_ylim(0, lim_p)
 
        diff   = p_ok - r_ok
        bias_v = float(np.mean(diff))
        ubrmsd = float(np.sqrt(np.mean((diff - bias_v)**2)))
        R, _   = pearsonr(p_ok, r_ok)
        ax.text(0.97, 0.97, f"R = {R:.3f}\nubRMSD = {ubrmsd:.3f}\nbias = {bias_v:+.3f}",
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                color=color, fontweight='bold',
                bbox=dict(fc='white', alpha=0.85, ec=color, lw=1.0, boxstyle='round,pad=0.35'))
        ax.set_xlabel('ASOS (mm/day)', fontsize=10)
        ax.set_ylabel(f'{pname} (mm/day)', fontsize=10)
        ax.set_title(f'{pname}  vs  ASOS', fontsize=11, fontweight='bold', pad=8)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.2, ls='--', lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
 
    fig.suptitle(f'{basin_name}  —  자료별 산점도  vs  ASOS  (유역 내 {n_pix}픽셀)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.show()
 
 
# ============================================================
# F-B4. 픽셀별 성능 박스플롯 (R / RMSE / Bias)
# ============================================================
print("\n[F-B4] 픽셀별 성능 박스플롯...")
 
# 픽셀별 통계 계산
pix_stats = {p: {'R': [], 'RMSE': [], 'Bias': []} for p in prod_names}
 
n_lat_g, n_lon_g = REF.shape[:2]
for i in range(n_lat_g):
    for j in range(n_lon_g):
        r_pix = REF[i, j, :]
        ok_ref = np.isfinite(r_pix)
        if ok_ref.sum() < 10: continue
        for pname in prod_names:
            p_pix = PRODUCTS[pname][i, j, :]
            ok = ok_ref & np.isfinite(p_pix)
            if ok.sum() < 10: continue
            p_v = p_pix[ok]; r_v = r_pix[ok]
            diff = p_v - r_v
            R_v, _ = pearsonr(p_v, r_v)
            pix_stats[pname]['R'].append(float(R_v))
            pix_stats[pname]['RMSE'].append(float(np.sqrt(np.mean(diff**2))))
            pix_stats[pname]['Bias'].append(float(np.mean(diff)))
 
COLORS_PIX = {
    'SM2RAIN': COLORS['SM2RAIN'], 'GPM': COLORS['GPM'],
    'TCA': COLORS['TCA'], 'ERA5': COLORS['ERA5'], 'BC': COLORS['BC'],
}
 
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
plt.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.12, wspace=0.35)
 
stat_cfgs = [
    ('R',    'Pearson R',     None, None),
    ('RMSE', 'RMSE (mm/day)', 0,    None),
    ('Bias', 'Bias (mm/day)', None, None),
]
for ax, (skey, ylabel, ymin, ymax) in zip(axes, stat_cfgs):
    data_list  = [pix_stats[p][skey] for p in prod_names]
    colors_box = [COLORS_PIX[p] for p in prod_names]
 
    bp = ax.boxplot(data_list, patch_artist=True, notch=False, widths=0.45,
                    medianprops=dict(color='white', lw=2.5),
                    whiskerprops=dict(lw=1.5, color='#555'),
                    capprops=dict(lw=1.5, color='#555'),
                    flierprops=dict(marker='o', ms=3, alpha=0.3, markeredgewidth=0),
                    boxprops=dict(linewidth=1.2))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.85)
 
    if skey == 'R':
        ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    elif skey == 'Bias':
        ax.axhline(0, color='gray', lw=1.2, ls='--', alpha=0.7)
 
    if ymin is not None: ax.set_ylim(bottom=ymin)
    if ymax is not None: ax.set_ylim(top=ymax)
    ax.set_xticks(range(1, len(prod_names) + 1))
    ax.set_xticklabels(prod_names, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight='bold', pad=8)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.25, ls='--', lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    for i, color in enumerate(colors_box):
        ax.axvspan(i + 0.6, i + 1.4, alpha=0.04, color=color, zorder=0)
 
fig.suptitle('픽셀별 성능 비교  (Validation vs ASOS)',
             fontsize=14, fontweight='bold')
plt.show()


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shap
import numpy as np

# 한글 피처명 매핑
KOR_FEATURES = {
    'SM2RAIN': 'SM2RAIN',
    'GPM':     'GPM',
    'ERA5':    'ERA5',
    'TCA':     'TCA 병합'
}

# SHAP 계산 시 피처명을 한글로 교체
df_shap_kor = df_shap[FEATURES].copy()
df_shap_kor.columns = [KOR_FEATURES[f] for f in FEATURES]
shap_vals_kor = shap_vals  # 값은 그대로, 컬럼명만 바꿈

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
plt.sca(ax)
shap.summary_plot(shap_vals_kor, df_shap_kor,
                  plot_type='dot', show=False,
                  max_display=len(FEATURES), color_bar=True)

ax.set_title('SHAP 피처 중요도  (Beeswarm)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('SHAP ', fontsize=12)

# 컬러바 라벨 한글 변환
cb = plt.gcf().axes[-1]  # 컬러바 축
cb.set_ylabel('피처 값', fontsize=11)
cb.set_yticks([0, 1])
cb.set_yticklabels(['낮음', '높음'], fontsize=10)

ax.tick_params(labelsize=12)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()
#%%