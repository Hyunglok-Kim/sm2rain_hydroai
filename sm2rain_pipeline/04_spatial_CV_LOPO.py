"""
04_spatial_CV_LOPO.py  (Leave-One-Pixel-Out 공간 교차검증)
==========================================================
BC_1 / BC_2(03_BC_LightGBM.py와 동일한 모델·특징)의 공간 일반화 성능을
Leave-One-Pixel-Out 방식으로 검증한다.
(출처: TCA_merging_clim.py의 LOPO 부분, 동일 알고리즘.)

방법
  - 각 픽셀마다 그 픽셀의 표본을 제외한 2021년 자료로 LightGBM을 학습한 뒤,
    해당 픽셀의 2021/2022/2023-2025 값을 예측 (미관측 지점을 모사).
  - LOPO_BC_1: 특징 = [SM2RAIN, ERA5, GPM, TCA, lon, lat]
  - LOPO_BC_2: 특징 = LOPO_BC_1 + [같은 날 AWS]
  - 픽셀별 n_estimators=100 (계산량 때문에 03보다 축소), 최소 학습표본 500
  - 평가: 2023-2025 기간 AWS 기준 픽셀별 R / ubRMSD / bias 지도

입력 : data/ds_merged_LR.nc
출력 : output/LOPO_metrics_BC1_BC2_2023_2025.nc (지표 지도)

* 계산량이 매우 큼. 빠른 테스트는 LOPO_MAX_PIXELS=20으로 먼저 확인할 것.
* 유틸(지표/데이터 로딩/모델)을 파일에 내장하여 단독 실행 가능.
  03_BC_LightGBM.py와 동일한 유틸을 쓰므로 수정 시 둘을 같이 맞출 것.
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.ensemble import HistGradientBoostingRegressor
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 최소 환경 대비 편의 처리
    def tqdm(iterable, *args, **kwargs):
        return iterable

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 경로 / 상수
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")       # 입력 .nc 파일을 여기에 둘 것
OUTPUT_DIR = os.path.join(HERE, "output")   # 결과가 저장되는 폴더
INPUT_PATH = os.path.join(DATA_DIR, "ds_merged_LR.nc")

Y = "AWS"                                   # BC 목표변수 (IDW_AWS)
BASE_FEATURES = ["SM2RAIN", "ERA5", "GPM", "TCA", "lon", "lat"]
X1 = BASE_FEATURES                          # BC_1: AWS 미포함
X2 = BASE_FEATURES + ["AWS"]                # BC_2: 같은 날 AWS 포함
POSITIVE_FEATURES = ["SM2RAIN", "ERA5", "GPM", "TCA"]

OUT_DIR = OUTPUT_DIR


# ==============================================================================
# 평가지표
# ==============================================================================
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
    """픽셀별 (R, ubRMSD, bias) 지도."""
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


# ==============================================================================
# 데이터 로딩 / DataFrame 구성 (2021 학습, 2022 검증, 2023-2025 시험)
# ==============================================================================
def aws_2021_clim_da(ds, time_sel):
    """대상 기간의 각 날짜를 같은 월-일의 2021년 AWS 값에 대응시킨다
    (2/29 -> 2/28)."""
    template = ds["AWS"].sel(time=time_sel)
    dates = pd.DatetimeIndex(template["time"].values)
    clim_dates = []
    for date in dates:
        month, day = int(date.month), int(date.day)
        if month == 2 and day == 29:
            day = 28
        clim_dates.append(pd.Timestamp(year=2021, month=month, day=day))

    clim = ds["AWS"].sel(time=pd.DatetimeIndex(clim_dates))
    clim = clim.assign_coords(time=template["time"].values)
    return clim.transpose(*template.dims)


def flatten_sin_doy(ds, time_sel):
    template = ds["SM2RAIN"].sel(time=time_sel)
    dates = pd.DatetimeIndex(template["time"].values)
    sin_doy = np.sin(2 * np.pi * dates.dayofyear.values / 365.25)
    sin_doy = xr.DataArray(
        sin_doy, dims=("time",), coords={"time": template["time"].values},
    )
    return sin_doy.broadcast_like(template).transpose(*template.dims).data.flatten()


def load_merged_dataset(path=INPUT_PATH):
    """ds_merged_LR.nc를 열고 AWS_clim2021 변수를 추가한다."""
    ds = xr.open_dataset(path)
    ds["AWS_clim2021"] = xr.concat(
        [
            aws_2021_clim_da(ds, "2021"),
            aws_2021_clim_da(ds, "2022"),
            aws_2021_clim_da(ds, slice("2023", "2025")),
        ],
        dim="time",
    ).sortby("time").reindex(time=ds["time"].values)
    return ds


def _split_df(ds, time_sel):
    return pd.DataFrame({
        "SM2RAIN": ds["SM2RAIN"].sel(time=time_sel).data.flatten(),
        "ERA5": ds["ERA5"].sel(time=time_sel).data.flatten(),
        "GPM": ds["GPM"].sel(time=time_sel).data.flatten(),
        "TCA": ds["TCA"].sel(time=time_sel).data.flatten(),
        "AWS": ds["AWS"].sel(time=time_sel).data.flatten(),
        "ASOS": ds["ASOS"].sel(time=time_sel).data.flatten(),
        "lon": ds["LON"].sel(time=time_sel).data.flatten(),
        "lat": ds["LAT"].sel(time=time_sel).data.flatten(),
        "AWS_clim2021": aws_2021_clim_da(ds, time_sel).data.flatten(),
        "sin_doy": flatten_sin_doy(ds, time_sel),
    })


def build_split_dataframes(ds):
    """(df_cal 2021, df_val 2022, df_tst 2023-2025). cal은 강수<=0인 행 제거."""
    df_cal = _split_df(ds, "2021")
    df_val = _split_df(ds, "2022")
    df_tst = _split_df(ds, slice("2023", "2025"))

    # 학습(2021)은 강우 사례만 사용 (모든 산출물 > 0)
    df_cal[df_cal["SM2RAIN"] <= 0] = np.nan
    df_cal[df_cal["ERA5"] <= 0] = np.nan
    df_cal[df_cal["GPM"] <= 0] = np.nan
    df_cal[df_cal["TCA"] <= 0] = np.nan
    df_cal[df_cal["AWS"] <= 0] = np.nan

    return df_cal, df_val, df_tst


# ==============================================================================
# ML 헬퍼
# ==============================================================================
def valid_ml_rows(df, features, target=None, target_positive_col=None):
    columns = list(dict.fromkeys(features))
    if target is not None and target not in columns:
        columns.append(target)

    valid = np.isfinite(df[columns]).all(axis=1)
    valid &= np.isfinite(df[POSITIVE_FEATURES]).all(axis=1)
    valid &= (df[POSITIVE_FEATURES] > 0).all(axis=1)
    if target_positive_col is not None:
        valid &= np.isfinite(df[target_positive_col])
        valid &= df[target_positive_col] > 0
    return valid


def make_bc_model(random_state, n_estimators=500):
    """LightGBM 우선 사용; 없으면 sklearn HistGradientBoosting으로 대체."""
    if LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=n_estimators,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=random_state,
    )


def flat_to_da(flat_pred, template, name):
    return xr.DataArray(
        flat_pred.reshape(template.shape),
        dims=template.dims,
        coords={dim: template[dim].values for dim in template.dims},
        name=name,
        attrs={
            "long_name": f"{name} bias-corrected precipitation",
            "units": "mm/day",
            "method": "direct LightGBM/HistGradientBoosting prediction of AWS",
        },
    )


def put_bc_to_ds(ds, name, cal_flat, val_flat, tst_flat):
    da_cal = flat_to_da(cal_flat, ds["AWS"].sel(time="2021"), name)
    da_val = flat_to_da(val_flat, ds["AWS"].sel(time="2022"), name)
    da_tst = flat_to_da(tst_flat, ds["AWS"].sel(time=slice("2023", "2025")), name)
    ds[name] = xr.concat(
        [da_cal, da_val, da_tst],
        dim="time",
    ).sortby("time").reindex(time=ds["time"].values)


def pixel_ids_for_template(template):
    if template.dims[0] != "time":
        raise ValueError(
            "LOPO pixel id assumes the df flatten order is (time, lat, lon). "
            f"current dims={template.dims}"
        )
    spatial_dims = [dim for dim in template.dims if dim != "time"]
    n_pixel = int(np.prod([template.sizes[dim] for dim in spatial_dims]))
    return np.tile(np.arange(n_pixel), template.sizes["time"])


def robust_percentile(arrays, q):
    vals = np.concatenate([np.asarray(arr).ravel() for arr in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return np.nanpercentile(vals, q)

LOPO_MAX_PIXELS = None        # 빠른 테스트: 예를 들어 20으로 제한
LOPO_N_ESTIMATORS = 100
LOPO_MIN_TRAIN_SAMPLES = 500
LOPO_EVAL_TIME = slice("2023", "2025")
LOPO_EVAL_LABEL = "2023_2025"
LOPO_METRICS_PATH = os.path.join(
    OUT_DIR,
    f"LOPO_metrics_BC1_BC2_{LOPO_EVAL_LABEL}.nc",
)

# ==============================================================================
# 데이터 로드 (03과 동일한 분할)
# ==============================================================================
ds_done = load_merged_dataset(INPUT_PATH)
df_cal, df_val, df_tst = build_split_dataframes(ds_done)


def lopo_fit_predict(name, features, max_pixels=None, n_estimators=200):
    cal_template = ds_done["AWS"].sel(time="2021")
    val_template = ds_done["AWS"].sel(time="2022")
    tst_template = ds_done["AWS"].sel(time=slice("2023", "2025"))

    cal_pixel = pixel_ids_for_template(cal_template)
    val_pixel = pixel_ids_for_template(val_template)
    tst_pixel = pixel_ids_for_template(tst_template)

    train_base = valid_ml_rows(df_cal, features, Y, target_positive_col=Y).values
    pixel_ids = np.unique(cal_pixel[train_base])
    if max_pixels is not None:
        pixel_ids = pixel_ids[:max_pixels]

    cal_pred = np.full(len(df_cal), np.nan, dtype=np.float64)
    val_pred = np.full(len(df_val), np.nan, dtype=np.float64)
    tst_pred = np.full(len(df_tst), np.nan, dtype=np.float64)

    for pixel_id in tqdm(pixel_ids, desc=f"LOPO {name}"):
        train_valid = train_base & (cal_pixel != pixel_id)
        if np.sum(train_valid) < LOPO_MIN_TRAIN_SAMPLES:
            continue

        model = make_bc_model(
            random_state=1000 + int(pixel_id),
            n_estimators=n_estimators,
        )
        model.fit(df_cal.loc[train_valid, features], df_cal.loc[train_valid, Y])

        for df, split_pixel, split_pred in (
            (df_cal, cal_pixel, cal_pred),
            (df_val, val_pixel, val_pred),
            (df_tst, tst_pixel, tst_pred),
        ):
            pred_valid = valid_ml_rows(df, features).values & (split_pixel == pixel_id)
            if np.any(pred_valid):
                split_pred[pred_valid] = np.maximum(
                    model.predict(df.loc[pred_valid, features]),
                    0.0,
                )

    put_bc_to_ds(ds_done, name, cal_pred, val_pred, tst_pred)


def plot_lopo_metric_maps(metric_rows):
    n_rows = len(metric_rows)
    fig = plt.figure(figsize=(8, 4 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(
        n_rows, 3,
        width_ratios=[1, 1, 0.045],
        wspace=0.12, hspace=0.22,
    )

    for row, (label, maps, vmin, vmax, cmap) in enumerate(metric_rows):
        for col, (da, name) in enumerate(maps):
            ax = fig.add_subplot(gs[row, col])
            im = ax.pcolormesh(
                ds_done.lon, ds_done.lat, da,
                vmin=vmin, vmax=vmax, cmap=cmap,
            )
            ax.set_title(f"{name} {label}: {np.nanmedian(da).round(3)}")
            ax.grid(alpha=0.3)
            if row < n_rows - 1:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])

        cax = fig.add_subplot(gs[row, 2])
        fig.colorbar(im, cax=cax, label=label)
    plt.tight_layout()
    plt.show()


def save_lopo_metrics(
    r_map_lopo_bc1, ubrmsd_map_lopo_bc1, bias_map_lopo_bc1,
    r_map_lopo_bc2, ubrmsd_map_lopo_bc2, bias_map_lopo_bc2,
):
    ds_metrics = xr.Dataset(
        {
            "R_LOPO_BC_1": r_map_lopo_bc1,
            "ubRMSD_LOPO_BC_1": ubrmsd_map_lopo_bc1,
            "bias_LOPO_BC_1": bias_map_lopo_bc1,
            "R_LOPO_BC_2": r_map_lopo_bc2,
            "ubRMSD_LOPO_BC_2": ubrmsd_map_lopo_bc2,
            "bias_LOPO_BC_2": bias_map_lopo_bc2,
        },
        attrs={
            "description": "Leave-one-pixel-out metric maps against AWS",
            "eval_time": LOPO_EVAL_LABEL,
            "BC_1_features": ", ".join(X1),
            "BC_2_features": ", ".join(X2),
            "n_estimators": int(LOPO_N_ESTIMATORS),
            "min_train_samples": int(LOPO_MIN_TRAIN_SAMPLES),
        },
    )
    ds_metrics.to_netcdf(LOPO_METRICS_PATH)
    print(f"Saved LOPO metrics: {LOPO_METRICS_PATH}")
    return ds_metrics


# ==============================================================================
# 실행
# ==============================================================================
lopo_fit_predict(
    "LOPO_BC_1", X1,
    max_pixels=LOPO_MAX_PIXELS,
    n_estimators=LOPO_N_ESTIMATORS,
)
lopo_fit_predict(
    "LOPO_BC_2", X2,
    max_pixels=LOPO_MAX_PIXELS,
    n_estimators=LOPO_N_ESTIMATORS,
)

ds_lopo = ds_done.sel(time=LOPO_EVAL_TIME)
r_map_LOPO_BC1, ubrmsd_map_LOPO_BC1, bias_map_LOPO_BC1 = get_eval_map(
    ds_lopo["LOPO_BC_1"], ds_lopo["AWS"],
)
r_map_LOPO_BC2, ubrmsd_map_LOPO_BC2, bias_map_LOPO_BC2 = get_eval_map(
    ds_lopo["LOPO_BC_2"], ds_lopo["AWS"],
)

lopo_common_nan = np.isnan(r_map_LOPO_BC1.data) | np.isnan(r_map_LOPO_BC2.data)
for da in [
    r_map_LOPO_BC1, ubrmsd_map_LOPO_BC1, bias_map_LOPO_BC1,
    r_map_LOPO_BC2, ubrmsd_map_LOPO_BC2, bias_map_LOPO_BC2,
]:
    da.data[lopo_common_nan] = np.nan

ubrmsd_vmax = robust_percentile(
    [ubrmsd_map_LOPO_BC1, ubrmsd_map_LOPO_BC2], 95,
)
bias_abs = max(
    abs(robust_percentile([bias_map_LOPO_BC1, bias_map_LOPO_BC2], 5)),
    abs(robust_percentile([bias_map_LOPO_BC1, bias_map_LOPO_BC2], 95)),
)

ds_lopo_metrics = save_lopo_metrics(
    r_map_LOPO_BC1, ubrmsd_map_LOPO_BC1, bias_map_LOPO_BC1,
    r_map_LOPO_BC2, ubrmsd_map_LOPO_BC2, bias_map_LOPO_BC2,
)

plot_lopo_metric_maps(
    [
        (
            "R [-]",
            [(r_map_LOPO_BC1, "LOPO_BC_1"), (r_map_LOPO_BC2, "LOPO_BC_2")],
            0, 1, "jet",
        ),
        (
            "ubRMSD [mm/day]",
            [(ubrmsd_map_LOPO_BC1, "LOPO_BC_1"), (ubrmsd_map_LOPO_BC2, "LOPO_BC_2")],
            0, ubrmsd_vmax, "jet",
        ),
        (
            "bias [mm/day]",
            [(bias_map_LOPO_BC1, "LOPO_BC_1"), (bias_map_LOPO_BC2, "LOPO_BC_2")],
            -bias_abs, bias_abs, "RdBu_r",
        ),
    ]
)
