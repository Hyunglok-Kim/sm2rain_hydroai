"""
02_TCA.py  (Triple Collocation Analysis 병합)
================================================
세 강수 산출물 - SM2RAIN(ASCAT), ERA5, IDW_AWS - 을 TCA로 병합한다.
(출처: TCA_merging_clim.py의 TCA 부분, 동일 알고리즘.)

방법 (Dong 방식 TCA, anomaly 병합)
  1) 각 산출물을 월별 기후값으로 분해: X = clim + anomaly
  2) 2021년 anomaly만 사용하여 픽셀별로 추정
     - 재척도 계수 (ERA5/AWS를 SM2RAIN 기준 척도로 변환)
     - TCA 오차분산 err2_i = <(x_i - x_j)(x_i - x_k)>  (교차곱)
  3) 가중치 w_i는 1/err2_i에 비례
     (w_sm = err2_era5*err2_aws / denom, 합 = 1)
  4) 전체 기간(2021-2025)의 anomaly를 가중 병합하고, SM2RAIN 기후값을
     되더한 뒤 음수는 0으로 절단.

입력 : data/ds_merged_LR.nc  (SM2RAIN / ERA5 / AWS 변수, time-lat-lon)
출력 : output/ds_merged_LR_TCA_SM2RAIN_ERA5_AWS_2021.nc  (병합 강수 추가)
       output/TCA_weights_LR_SM2RAIN_ERA5_AWS_2021.nc    (err2, w, scale 지도)
"""
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 최소 환경 대비 편의 처리
    def tqdm(iterable, *args, **kwargs):
        return iterable

warnings.filterwarnings("ignore", category=RuntimeWarning)


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")       # 입력 .nc 파일을 여기에 둘 것
OUT_DIR = os.path.join(HERE, "output")      # 결과가 저장되는 폴더
INPUT_PATH = os.path.join(DATA_DIR, "ds_merged_LR.nc")
OUT_MERGE_PATH = os.path.join(
    OUT_DIR,
    "ds_merged_LR_TCA_SM2RAIN_ERA5_AWS_2021.nc",
)
OUT_WEIGHTS_PATH = os.path.join(
    OUT_DIR,
    "TCA_weights_LR_SM2RAIN_ERA5_AWS_2021.nc",
)

MEMBERS = ("SM2RAIN", "ERA5", "AWS")
MERGED_VAR = "TCA_SM2RAIN_ERA5_AWS_2021"
FIT_YEAR = 2021
MIN_TRIPLET = 100

# True면 TCA 오차분산이 음수/0으로 나온 육지 픽셀이 사라지지 않게 채운다.
# 순수 TCA 유효 픽셀만 엄격하게 쓰려면 False로 설정.
FILL_NONPOSITIVE_ERR2 = True


def _require_members(ds, members):
    missing = [name for name in members if name not in ds.data_vars]
    if missing:
        raise KeyError(f"required variables missing from da_merged_LR: {missing}")


def _as_lat_lon_time(ds, var_name):
    da = ds[var_name]
    needed = {"time", "lat", "lon"}
    if not needed.issubset(set(da.dims)):
        raise ValueError(
            f"{var_name} dims must include time/lat/lon. current dims={da.dims}"
        )

    arr = da.transpose("lat", "lon", "time").values.astype(np.float64)
    arr[arr < 0] = np.nan
    return arr


def calc_monthly_climatology(arr_3d, dates):
    """월별 anomaly와 (전체 시간축으로 broadcast된) 월별 기후값을 반환."""
    months = dates.month
    clim = np.full_like(arr_3d, np.nan, dtype=np.float64)

    for month in range(1, 13):
        idx = np.where(months == month)[0]
        if len(idx) == 0:
            continue
        clim[:, :, idx] = np.nanmean(arr_3d[:, :, idx], axis=2, keepdims=True)

    return arr_3d - clim, clim


def dong_tca_pixel(x, y, z, min_n):
    """한 픽셀에 대한 Dong 방식 TCA. x가 기준 산출물."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n_valid = int(valid.sum())
    if n_valid < min_n:
        return np.nan, np.nan, np.nan, n_valid

    xv = x[valid]
    yv = y[valid]
    zv = z[valid]

    denom_y = float(yv @ zv)
    denom_z = float(zv @ yv)
    if abs(denom_y) < 1e-12 or abs(denom_z) < 1e-12:
        return np.nan, np.nan, np.nan, n_valid

    scale_y = float(xv @ zv) / denom_y
    scale_z = float(xv @ yv) / denom_z

    xs = xv
    ys = scale_y * yv
    zs = scale_z * zv

    err_x = float(np.mean((xs - ys) * (xs - zs)))
    err_y = float(np.mean((ys - xs) * (ys - zs)))
    err_z = float(np.mean((zs - xs) * (zs - ys)))
    return err_x, err_y, err_z, n_valid


def _sanitize_error_variance(err2, label):
    bad = np.isfinite(err2) & (err2 <= 0)
    if not FILL_NONPOSITIVE_ERR2:
        err2[bad] = np.nan
        print(f"  {label}: negative/zero error variance -> NaN {int(bad.sum())} pixels")
        return

    positive = err2[np.isfinite(err2) & (err2 > 0)]
    fill_value = float(np.nanmedian(positive)) if positive.size else 1.0
    err2[bad] = fill_value
    print(
        f"  {label}: negative/zero error variance fill "
        f"{int(bad.sum())} pixels -> {fill_value:.4f}"
    )


def estimate_tca_weights(A_sm_fit, A_era5_fit, A_aws_fit, min_triplet):
    n_lat, n_lon, _ = A_sm_fit.shape
    shape = (n_lat, n_lon)

    err2_sm = np.full(shape, np.nan, dtype=np.float64)
    err2_era5 = np.full(shape, np.nan, dtype=np.float64)
    err2_aws = np.full(shape, np.nan, dtype=np.float64)
    n_valid_map = np.zeros(shape, dtype=np.int16)

    for i in tqdm(range(n_lat), desc="TCA weights"):
        for j in range(n_lon):
            ex, ey, ez, n_valid = dong_tca_pixel(
                A_sm_fit[i, j, :],
                A_era5_fit[i, j, :],
                A_aws_fit[i, j, :],
                min_triplet,
            )
            err2_sm[i, j] = ex
            err2_era5[i, j] = ey
            err2_aws[i, j] = ez
            n_valid_map[i, j] = n_valid

    _sanitize_error_variance(err2_sm, "SM2RAIN")
    _sanitize_error_variance(err2_era5, "ERA5")
    _sanitize_error_variance(err2_aws, "AWS")

    denom = (
        err2_sm * err2_era5
        + err2_era5 * err2_aws
        + err2_sm * err2_aws
    )

    w_sm = err2_era5 * err2_aws / denom
    w_era5 = err2_sm * err2_aws / denom
    w_aws = err2_sm * err2_era5 / denom
    valid_w = (
        np.isfinite(w_sm)
        & np.isfinite(w_era5)
        & np.isfinite(w_aws)
        & np.isfinite(denom)
        & (denom != 0)
    )

    return {
        "err2_sm": err2_sm,
        "err2_era5": err2_era5,
        "err2_aws": err2_aws,
        "w_sm": w_sm,
        "w_era5": w_era5,
        "w_aws": w_aws,
        "valid_w": valid_w,
        "n_valid_map": n_valid_map,
    }


def estimate_scales(A_sm_fit, A_era5_fit, A_aws_fit, min_triplet):
    n_lat, n_lon, _ = A_sm_fit.shape
    scale_era5 = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    scale_aws = np.full((n_lat, n_lon), np.nan, dtype=np.float64)

    for i in tqdm(range(n_lat), desc="TCA scales"):
        for j in range(n_lon):
            s = A_sm_fit[i, j, :]
            e = A_era5_fit[i, j, :]
            a = A_aws_fit[i, j, :]
            valid = np.isfinite(s) & np.isfinite(e) & np.isfinite(a)
            if valid.sum() < min_triplet:
                continue

            sv = s[valid]
            ev = e[valid]
            av = a[valid]

            denom_era5 = float(ev @ av)
            denom_aws = float(av @ ev)
            if abs(denom_era5) > 1e-12:
                scale_era5[i, j] = float(sv @ av) / denom_era5
            if abs(denom_aws) > 1e-12:
                scale_aws[i, j] = float(sv @ ev) / denom_aws

    return scale_era5, scale_aws


def merge_anomalies(A_sm, A_era5, A_aws, weights, scale_era5, scale_aws):
    n_lat, n_lon, n_time = A_sm.shape
    merged = np.full((n_lat, n_lon, n_time), np.nan, dtype=np.float64)

    valid_w = weights["valid_w"]
    w_sm = weights["w_sm"]
    w_era5 = weights["w_era5"]
    w_aws = weights["w_aws"]

    for i in tqdm(range(n_lat), desc="TCA merge"):
        for j in range(n_lon):
            if not valid_w[i, j]:
                continue
            if not (np.isfinite(scale_era5[i, j]) and np.isfinite(scale_aws[i, j])):
                continue

            xs = A_sm[i, j, :]
            ys = scale_era5[i, j] * A_era5[i, j, :]
            zs = scale_aws[i, j] * A_aws[i, j, :]

            merged[i, j, :] = (
                w_sm[i, j] * xs
                + w_era5[i, j] * ys
                + w_aws[i, j] * zs
            )

    return merged


def build_weight_dataset(ds, weights, scale_era5, scale_aws):
    lat = ds["lat"].values
    lon = ds["lon"].values

    return xr.Dataset(
        data_vars={
            "err2_SM2RAIN": xr.DataArray(
                weights["err2_sm"], dims=("lat", "lon"),
                attrs={"long_name": "TCA error variance (SM2RAIN)", "units": "mm2/day2"},
            ),
            "err2_ERA5": xr.DataArray(
                weights["err2_era5"], dims=("lat", "lon"),
                attrs={"long_name": "TCA error variance (ERA5)", "units": "mm2/day2"},
            ),
            "err2_AWS": xr.DataArray(
                weights["err2_aws"], dims=("lat", "lon"),
                attrs={"long_name": "TCA error variance (AWS)", "units": "mm2/day2"},
            ),
            "w_SM2RAIN": xr.DataArray(
                weights["w_sm"], dims=("lat", "lon"),
                attrs={"long_name": "TCA weight SM2RAIN"},
            ),
            "w_ERA5": xr.DataArray(
                weights["w_era5"], dims=("lat", "lon"),
                attrs={"long_name": "TCA weight ERA5"},
            ),
            "w_AWS": xr.DataArray(
                weights["w_aws"], dims=("lat", "lon"),
                attrs={"long_name": "TCA weight AWS"},
            ),
            "scale_ERA5": xr.DataArray(
                scale_era5, dims=("lat", "lon"),
                attrs={"long_name": "Scaling parameter ax/ay (ERA5 to SM2RAIN ref)"},
            ),
            "scale_AWS": xr.DataArray(
                scale_aws, dims=("lat", "lon"),
                attrs={"long_name": "Scaling parameter ax/az (AWS to SM2RAIN ref)"},
            ),
            "n_valid_2021": xr.DataArray(
                weights["n_valid_map"], dims=("lat", "lon"),
                attrs={"long_name": "number of valid 2021 triplet observations"},
            ),
        },
        coords={
            "lat": xr.DataArray(lat, dims="lat"),
            "lon": xr.DataArray(lon, dims="lon"),
        },
        attrs={
            "title": "TCA weights for da_merged_LR",
            "inputs": ", ".join(MEMBERS),
            "reference_member": "SM2RAIN",
            "fit_year": str(FIT_YEAR),
            "min_triplet": MIN_TRIPLET,
            "nonpositive_error_variance_policy": (
                "filled with positive median"
                if FILL_NONPOSITIVE_ERR2
                else "set to NaN"
            ),
        },
    )


def run_tca_merge(
    input_path=INPUT_PATH,
    out_merge_path=OUT_MERGE_PATH,
    out_weights_path=OUT_WEIGHTS_PATH,
):
    print("=" * 70)
    print("TCA merging from da_merged_LR")
    print(f"  input   : {input_path}")
    print(f"  members : {', '.join(MEMBERS)}")
    print(f"  fit year: {FIT_YEAR}")
    print("=" * 70)

    ds = xr.open_dataset(input_path).sortby(["time", "lat", "lon"])
    _require_members(ds, MEMBERS)

    dates = pd.DatetimeIndex(ds["time"].values)
    fit_mask = dates.year == FIT_YEAR
    if int(fit_mask.sum()) < MIN_TRIPLET:
        raise ValueError(
            f"the valid {FIT_YEAR} period ({int(fit_mask.sum())} days) is "
            f"shorter than MIN_TRIPLET ({MIN_TRIPLET})."
        )

    print(f"  period  : {dates[0].date()} ~ {dates[-1].date()} ({len(dates)} days)")
    print(f"  {FIT_YEAR} fit: {int(fit_mask.sum())} days")

    X_sm = _as_lat_lon_time(ds, "SM2RAIN")
    X_era5 = _as_lat_lon_time(ds, "ERA5")
    X_aws = _as_lat_lon_time(ds, "AWS")

    A_sm, clim_sm = calc_monthly_climatology(X_sm, dates)
    A_era5, _ = calc_monthly_climatology(X_era5, dates)
    A_aws, _ = calc_monthly_climatology(X_aws, dates)

    A_sm_fit = A_sm[:, :, fit_mask]
    A_era5_fit = A_era5[:, :, fit_mask]
    A_aws_fit = A_aws[:, :, fit_mask]

    weights = estimate_tca_weights(A_sm_fit, A_era5_fit, A_aws_fit, MIN_TRIPLET)
    scale_era5, scale_aws = estimate_scales(
        A_sm_fit, A_era5_fit, A_aws_fit, MIN_TRIPLET,
    )
    M_anomaly = merge_anomalies(A_sm, A_era5, A_aws, weights, scale_era5, scale_aws)

    P_merged = np.maximum(M_anomaly + clim_sm, 0.0)
    valid_pixels = int(np.sum(np.any(np.isfinite(P_merged), axis=2)))

    print("Summary")
    print(f"  valid weight pixels: {int(weights['valid_w'].sum())}")
    print(f"  merged valid pixels: {valid_pixels}")
    print(f"  w_SM2RAIN mean     : {np.nanmean(weights['w_sm']):.3f}")
    print(f"  w_ERA5 mean        : {np.nanmean(weights['w_era5']):.3f}")
    print(f"  w_AWS mean         : {np.nanmean(weights['w_aws']):.3f}")
    print(
        "  w_sum mean         : "
        f"{np.nanmean(weights['w_sm'] + weights['w_era5'] + weights['w_aws']):.6f}"
    )

    ds_out = ds.copy()
    ds_out[MERGED_VAR] = xr.DataArray(
        np.transpose(P_merged, (2, 0, 1)),
        dims=("time", "lat", "lon"),
        coords={
            "time": ds["time"].values,
            "lat": ds["lat"].values,
            "lon": ds["lon"].values,
        },
        attrs={
            "long_name": "TCA-merged precipitation (SM2RAIN + ERA5 + AWS)",
            "units": "mm/day",
            "method": "Dong-style TCA anomaly merge",
            "fit_period": f"{FIT_YEAR} only",
            "reference_climatology": "SM2RAIN",
            "members": ", ".join(MEMBERS),
        },
    )
    ds_out.attrs.update(
        {
            "TCA_SM2RAIN_ERA5_AWS_fit_year": str(FIT_YEAR),
            "TCA_SM2RAIN_ERA5_AWS_members": ", ".join(MEMBERS),
            "TCA_SM2RAIN_ERA5_AWS_weight_file": os.path.basename(out_weights_path),
        }
    )

    ds_weights = build_weight_dataset(ds, weights, scale_era5, scale_aws)

    os.makedirs(os.path.dirname(out_merge_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_weights_path), exist_ok=True)
    ds_out.to_netcdf(out_merge_path)
    ds_weights.to_netcdf(out_weights_path)

    print("Saved")
    print(f"  merged : {out_merge_path}")
    print(f"  weights: {out_weights_path}")

    return ds_out, ds_weights


if __name__ == "__main__":
    run_tca_merge()
