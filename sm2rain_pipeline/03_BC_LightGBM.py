"""
03_BC_LightGBM.py  (LightGBM 편향보정)
=============================================
SM2RAIN / ERA5 / GPM / TCA를 입력특징으로 IDW_AWS를 직접 예측하는
LightGBM을 학습하여 강수를 편향보정한다.
(출처: TCA_merging_clim.py의 편향보정 부분, 동일 알고리즘.)

방법
  - 목표변수 Y = AWS (IDW_AWS 일강수)
  - BC_1: 특징 = [SM2RAIN, ERA5, GPM, TCA, lon, lat]        (AWS 미포함)
  - BC_2: 특징 = BC_1 + [같은 날 AWS]                       (상한 확인용)
  - 2021 학습 / 2022 검증 / 2023-2025 시험 (시간 분할)
  - 학습 표본: 모든 강수 산출물이 > 0인 강우 사례만 사용
  - 음수 예측값은 0으로 절단
  - LightGBM: n_estimators=500, lr=0.03, num_leaves=31, subsample=0.9,
    colsample=0.9, reg_lambda=1.0 (미설치 시 HistGradientBoosting으로 대체)

평가
  - 기간별 산점도 (BC_1, BC_2, ERA5, ASOS vs AWS)
  - 픽셀별 R / ubRMSD / bias 지도 (2023-2025)

입력 : data/ds_merged_LR.nc (SM2RAIN/ERA5/GPM/TCA/AWS/ASOS 변수 포함)
출력 : output/ds_merged_LR_BC.nc (BC_1, BC_2 변수 추가)

* 유틸(지표/데이터 로딩/모델)을 파일에 내장하여 단독 실행 가능.
  04_spatial_CV_LOPO.py와 동일한 유틸을 쓰므로 수정 시 둘을 같이 맞출 것.
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

OUT_PATH = os.path.join(OUTPUT_DIR, "ds_merged_LR_BC.nc")
SAVE_OUTPUT = True


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


def add_metrics(x, y, ax=None, s=0, show_corr=True, show_ubRMSD=True,
                show_bias=True, show_line_eq=True, corner="top-right",
                color="black", alpha=0.2, fontsize=8, vminmax=None):
    """축에 산점도와 R/ubRMSD/bias/회귀선 텍스트를 추가한다."""
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan

    x = np.array(x[mask])
    y = np.array(y[mask])

    corr_ = np.corrcoef(x, y)[0, 1]
    ubRMSD_ = ubrmsd(x, y)
    bias_ = np.nanmean(y - x)

    if np.nanstd(x) < 1e-12:
        slope = 0.0
        intercept = float(np.nanmean(y))
    else:
        slope, intercept = np.polyfit(x, y, 1)

    metrics = []
    if show_corr:
        metrics.append(r"$R=%.3f$" % corr_)
    if show_ubRMSD:
        metrics.append(r"$ubRMSD=%.3f$" % ubRMSD_)
    if show_bias:
        metrics.append(r"$bias=%.3f$" % bias_)
    if show_line_eq:
        metrics.append(r"$y = {:.4f}x {:+.3f}$".format(slope, intercept))
    textstr = "\n".join(metrics)

    corner_positions = {
        "top-right": (0.95, 0.95),
        "top-left": (0.05, 0.95),
        "bottom-right": (0.95, 0.05),
        "bottom-left": (0.05, 0.05),
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
    ax.plot(x_pred, slope * x_pred + intercept, color=color, alpha=0.8, linewidth=2)

    props = dict(boxstyle="round", facecolor="white", alpha=0.9)
    ax.text(
        position[0], position[1], textstr,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top" if "top" in corner else "bottom",
        horizontalalignment="right" if "right" in corner else "left",
        bbox=props,
        color=color,
    )
    ax.set_facecolor("w")


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
def valid_pair(target_da, pred_da, time_sel):
    target = target_da.sel(time=time_sel).values.flatten()
    pred = pred_da.sel(time=time_sel).values.flatten()
    valid = (
        np.isfinite(target)
        & np.isfinite(pred)
        & (target > 0)
        & (pred > 0)
    )
    return target[valid], pred[valid]


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


def predict_flat(model, df, features):
    pred_valid = valid_ml_rows(df, features)
    pred = np.full(len(df), np.nan, dtype=np.float64)
    pred[pred_valid.values] = np.maximum(
        model.predict(df.loc[pred_valid, features]),
        0.0,
    )
    return pred


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

# ==============================================================================
# 데이터 로드 및 학습/검증/시험 분할
# ==============================================================================
ds_done = load_merged_dataset(INPUT_PATH)
df_cal, df_val, df_tst = build_split_dataframes(ds_done)

# ==============================================================================
# 모델 1: AWS를 입력으로 쓰지 않고 AWS를 직접 예측.
# ==============================================================================
cal_valid1 = valid_ml_rows(df_cal, X1, Y, target_positive_col=Y)
bc_model1 = make_bc_model(random_state=42)
bc_model1.fit(df_cal.loc[cal_valid1, X1], df_cal.loc[cal_valid1, Y])

BC1_cal = predict_flat(bc_model1, df_cal, X1)
BC1_val = predict_flat(bc_model1, df_val, X1)
BC1_tst = predict_flat(bc_model1, df_tst, X1)
put_bc_to_ds(ds_done, "BC_1", BC1_cal, BC1_val, BC1_tst)

# ==============================================================================
# 모델 2: 같은 날 AWS를 입력에 추가하여 AWS를 직접 예측.
# ==============================================================================
cal_valid2 = valid_ml_rows(df_cal, X2, Y, target_positive_col=Y)
bc_model2 = make_bc_model(random_state=43)
bc_model2.fit(df_cal.loc[cal_valid2, X2], df_cal.loc[cal_valid2, Y])

BC2_cal = predict_flat(bc_model2, df_cal, X2)
BC2_val = predict_flat(bc_model2, df_val, X2)
BC2_tst = predict_flat(bc_model2, df_tst, X2)
put_bc_to_ds(ds_done, "BC_2", BC2_cal, BC2_val, BC2_tst)

# ==============================================================================
# 산점도 평가 (학습/검증/시험 x BC_1/BC_2/ERA5/ASOS)
# ==============================================================================
cal_y_bc1, BC1_cal_valid = valid_pair(ds_done["AWS"], ds_done["BC_1"], "2021")
val_y_bc1, BC1_val_valid = valid_pair(ds_done["AWS"], ds_done["BC_1"], "2022")
tst_y_bc1, BC1_tst_valid = valid_pair(
    ds_done["AWS"], ds_done["BC_1"], slice("2023", "2025"),
)

cal_y_bc2, BC2_cal_valid = valid_pair(ds_done["AWS"], ds_done["BC_2"], "2021")
val_y_bc2, BC2_val_valid = valid_pair(ds_done["AWS"], ds_done["BC_2"], "2022")
tst_y_bc2, BC2_tst_valid = valid_pair(
    ds_done["AWS"], ds_done["BC_2"], slice("2023", "2025"),
)

cal_y_era5, ERA5_cal = valid_pair(ds_done["AWS"], ds_done["ERA5"], "2021")
val_y_era5, ERA5_val = valid_pair(ds_done["AWS"], ds_done["ERA5"], "2022")
tst_y_era5, ERA5_tst = valid_pair(
    ds_done["AWS"], ds_done["ERA5"], slice("2023", "2025"),
)

cal_y_asos, ASOS_cal = valid_pair(ds_done["AWS"], ds_done["ASOS"], "2021")
val_y_asos, ASOS_val = valid_pair(ds_done["AWS"], ds_done["ASOS"], "2022")
tst_y_asos, ASOS_tst = valid_pair(
    ds_done["AWS"], ds_done["ASOS"], slice("2023", "2025"),
)

fig, axs = plt.subplots(4, 3, figsize=(10, 12), sharex=True, sharey=True)
axs = axs.flatten()
add_metrics(cal_y_bc1, BC1_cal_valid, s=2, ax=axs[0])
add_metrics(val_y_bc1, BC1_val_valid, s=2, ax=axs[1])
add_metrics(tst_y_bc1, BC1_tst_valid, s=2, ax=axs[2])

add_metrics(cal_y_bc2, BC2_cal_valid, s=2, ax=axs[3])
add_metrics(val_y_bc2, BC2_val_valid, s=2, ax=axs[4])
add_metrics(tst_y_bc2, BC2_tst_valid, s=2, ax=axs[5])

add_metrics(cal_y_era5, ERA5_cal, s=2, ax=axs[6])
add_metrics(val_y_era5, ERA5_val, s=2, ax=axs[7])
add_metrics(tst_y_era5, ERA5_tst, s=2, ax=axs[8])

add_metrics(cal_y_asos, ASOS_cal, s=2, ax=axs[9])
add_metrics(val_y_asos, ASOS_val, s=2, ax=axs[10])
add_metrics(tst_y_asos, ASOS_tst, s=2, ax=axs[11])

axs[0].set_title("2021 train")
axs[1].set_title("2022 val")
axs[2].set_title("2023-2025 test")
axs[0].set_ylabel("BC_1 no AWS input [mm/day]")
axs[3].set_ylabel("BC_2 with AWS input [mm/day]")
axs[6].set_ylabel("ERA5 [mm/day]")
axs[9].set_ylabel("ASOS [mm/day]")
axs[9].set_xlabel("AWS [mm/day]")
axs[10].set_xlabel("AWS [mm/day]")
axs[11].set_xlabel("AWS [mm/day]")
for ax in axs:
    ax.set_ylim(0, 350)
    ax.set_xlim(0, 350)
    ax.plot([0, 350], [0, 350], c="k")
plt.tight_layout()

# ==============================================================================
# 픽셀별 평가 지도 (2023-2025 시험기간)
# ==============================================================================
ds_done_ = ds_done.sel(time=slice("2023", "2025"))

r_map_BC1, ubrmsd_map_BC1, bias_map_BC1 = get_eval_map(ds_done_["BC_1"], ds_done_["AWS"])
r_map_BC2, ubrmsd_map_BC2, bias_map_BC2 = get_eval_map(ds_done_["BC_2"], ds_done_["AWS"])
r_map_ASOS, ubrmsd_map_ASOS, bias_map_ASOS = get_eval_map(ds_done_["ASOS"], ds_done_["AWS"])

# BC_1과 BC_2 모두 유효한 픽셀만 비교
common_nan = np.isnan(r_map_BC1.data) | np.isnan(r_map_BC2.data)
for da in [
    r_map_BC1, ubrmsd_map_BC1, bias_map_BC1,
    r_map_BC2, ubrmsd_map_BC2, bias_map_BC2,
    r_map_ASOS, ubrmsd_map_ASOS, bias_map_ASOS,
]:
    da.data[common_nan] = np.nan

plot_items = [
    [
        (r_map_BC1, "R (AWS, BC_1)", 0, 1, "jet", "R [-]"),
        (r_map_BC2, "R (AWS, BC_2)", 0, 1, "jet", "R [-]"),
        (r_map_ASOS, "R (AWS, ASOS)", 0, 1, "jet", "R [-]"),
    ],
    [
        (ubrmsd_map_BC1, "ubRMSD (AWS, BC_1)", 0, 20, "jet", "ubRMSD [mm/day]"),
        (ubrmsd_map_BC2, "ubRMSD (AWS, BC_2)", 0, 20, "jet", "ubRMSD [mm/day]"),
        (ubrmsd_map_ASOS, "ubRMSD (AWS, ASOS)", 0, 20, "jet", "ubRMSD [mm/day]"),
    ],
    [
        (bias_map_BC1, "Bias (AWS, BC_1)", -10, 10, "RdBu_r", "Bias [mm/day]"),
        (bias_map_BC2, "Bias (AWS, BC_2)", -10, 10, "RdBu_r", "Bias [mm/day]"),
        (bias_map_ASOS, "Bias (AWS, ASOS)", -10, 10, "RdBu_r", "Bias [mm/day]"),
    ],
]

fig, axs = plt.subplots(3, 3, figsize=(8, 8), constrained_layout=True)
for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        da, title, vmin, vmax, cmap, cbar_label = plot_items[i][j]
        im = ax.pcolormesh(ds_done.lon, ds_done.lat, da, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f"{title}: {np.nanmedian(da):.3f}", fontsize=10)
        ax.grid(alpha=0.3)
        if i < 2:
            ax.set_xticklabels([])
        if j > 0:
            ax.set_yticklabels([])
    fig.colorbar(im, ax=axs[i, :], fraction=0.025, pad=0.02, label=cbar_label)

plt.show()

# ==============================================================================
# 저장
# ==============================================================================
if SAVE_OUTPUT:
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
    ds_done.to_netcdf(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
