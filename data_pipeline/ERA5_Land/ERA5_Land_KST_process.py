"""
ERA5_Land_KST_process.py  (시간별 누계 해제 + KST 일강수, 한 파일)
================================================================
ERA5_Land_download.py 가 받은 시간별 누계 강수(원본)를
  (1) 누계 해제(de-accumulate)  →  (2) KST(+9h) 변환  →  (3) 일강수 재집계
한 흐름으로 처리하여 연도별 KST 일강수 nc 를 만든다.

ERA5-Land tp 는 stepType=accum(00:00 UTC 예보 시작부터 누적, 단위 m)이라
그대로 하루 마지막 값을 쓰면 'UTC 일경계'가 된다. 이 코드는 시간별로 풀어
+9h 후 KST 하루(00~24시 KST)로 다시 합친다. (원본 파일은 건드리지 않음)

누계 해제의 두 가지 함정
  - 00:00 스텝은 '전날 누계의 꼬리'이므로 당일 baseline 0 으로 처리.
  - 23~24시(UTC) 강수 = (다음날 파일의 00:00 = 당일 총량) − (당일 23:00 누계)
    로 정밀 계산한다 (이래서 직전/다음 날 파일을 함께 본다).

입력 : raw/{YYYY}/{YYYY.MM.DD}/ERA5_Land_P_{YYYYMMDD}.nc   (다운로드 산출)
       - tp: (valid_time=24, latitude, longitude), stepType=accum, 단위 m
출력 : output/era5_land{YYYY}_KST.nc   (lat, lon, time / mm/day, KST 일경계)
       output/era5_land_P_KST.nc       (전체 병합)

* 지상관측(ASOS/AWS)과 시간대(KST)를 맞춘 ERA5 강수장으로, TCA 입력에 쓴다.
* 격자는 ERA5-Land native(약 0.1°) 그대로 두며, SM2RAIN 격자 정합은
  이후 TCA 단계에서 in-memory regrid 로 처리한다.
* 의존성: numpy, pandas, xarray, netCDF4, tqdm
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 최소 환경 대비 편의 처리
    def tqdm(iterable, *args, **kwargs):
        return iterable

# ==============================================================================
# 설정
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
ERA5_RAW_DIR = os.path.join(HERE, "raw")       # 시간별 누계 원본
SAVE_DIR = os.path.join(HERE, "output")
YEARS = [2021, 2022, 2023, 2024, 2025]
KST_OFFSET_H = 9

# 한국 영역 subset (원본 lat 은 내림차순, lon 은 오름차순)
LAT_MAX, LAT_MIN = 39.5, 32.5
LON_MIN, LON_MAX = 123.5, 130.5


# ==============================================================================
# 한 UTC 일 파일 → 시간별 강수(mm) 배열 (누계 해제)
# ==============================================================================
def deaccumulate_day(nc_path):
    """반환: hourly(0~22시), lat, lon, ocean, cum23, art00.
      hourly[H] = UTC (date + H시)에 시작하는 1시간 강수 (H=0..22).
      00:00 스텝은 전날 누계 꼬리라 baseline 0 으로 처리한다.
      H=23(23~24시 UTC)은 build 단계에서 다음날 00:00 값으로 정밀 계산한다.
      cum23 = 00:00→23:00 누계,  art00 = 이 날 00:00 값(= 전날 총량).
    """
    with xr.open_dataset(nc_path) as ds:
        sub = ds.sel(latitude=slice(LAT_MAX, LAT_MIN),
                     longitude=slice(LON_MIN, LON_MAX))
        raw = sub["tp"].values.astype("float64") * 1000.0   # (24, ny, nx) m→mm, 누계
        lat = sub["latitude"].values
        lon = sub["longitude"].values

    if raw.shape[0] != 24:
        raise ValueError(f"시간축 길이가 24가 아님: {nc_path} ({raw.shape[0]})")

    # ERA5-Land 는 바다에서 전 시간 NaN → 육지/바다 마스크 (계산은 0으로 채워 진행)
    ocean = np.isnan(raw).all(axis=0)          # (ny, nx) True=바다(무효)
    raw = np.nan_to_num(raw, nan=0.0)

    art00 = raw[0].copy()                      # 이 날 00:00 값 = 전날 총량
    tpc = raw.copy()
    tpc[0] = 0.0                               # 00:00 함정 → 당일 baseline 0

    hourly = np.zeros_like(tpc)                # (24, ny, nx)
    hourly[:23] = np.diff(tpc, axis=0)         # H=0..22: tpc[H+1]-tpc[H]
    np.clip(hourly, 0.0, None, out=hourly)
    cum23 = tpc[23].copy()                     # 00:00→23:00 누계
    return hourly, lat, lon, ocean, cum23, art00


# ==============================================================================
# 일 파일 목록 (연속 KST 경계를 위해 직전 연말 하루 포함)
# ==============================================================================
def list_day_files(years, raw_dir):
    files = []
    need_years = sorted(set(years) | {min(years) - 1})   # 경계용 직전 해
    for yr in need_years:
        yr_dir = os.path.join(raw_dir, str(yr))
        if not os.path.isdir(yr_dir):
            continue
        for day_dir in sorted(os.listdir(yr_dir)):
            dpath = os.path.join(yr_dir, day_dir)
            if not os.path.isdir(dpath) or day_dir.startswith("."):
                continue
            date_str = day_dir.replace(".", "")            # YYYYMMDD
            nc = os.path.join(dpath, f"ERA5_Land_P_{date_str}.nc")
            if not os.path.exists(nc):
                continue
            try:
                d = pd.Timestamp(date_str)
            except Exception:  # noqa: BLE001
                continue
            # 직전 해는 12/31 하루만 (KST 1/1 경계 채우기용)
            if d.year == min(years) - 1 and not (d.month == 12 and d.day == 31):
                continue
            files.append((d, nc))
    return sorted(files, key=lambda x: x[0])


# ==============================================================================
# STEP 1. 누계 해제 + KST 일강수 생성
# ==============================================================================
def build_kst_daily(years, raw_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    day_files = list_day_files(years, raw_dir)
    if not day_files:
        raise FileNotFoundError(f"입력 일 파일 없음: {raw_dir} (먼저 ERA5_Land_download.py 실행)")
    print(f"대상 일 파일: {len(day_files)}개 "
          f"({day_files[0][0].date()} ~ {day_files[-1][0].date()})")

    acc = {}                 # kst_date -> (ny,nx) 합산 강수
    hour_counts = {}         # kst_date -> 합산된 1시간 값 개수 (24 완전성 검증)
    lat_ref = lon_ref = None
    ocean_mask = None
    prev = None              # (직전 날짜, 직전 cum23) — 23~24시 정밀 계산용
    skipped = []

    def _add(kd, val):
        if kd not in acc:
            acc[kd] = np.zeros(val.shape, dtype="float64")
            hour_counts[kd] = 0
        acc[kd] += val
        hour_counts[kd] += 1

    for date, nc in tqdm(day_files, desc="누계해제 + KST"):
        try:
            hourly, lat, lon, ocean, cum23, art00 = deaccumulate_day(nc)
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  [스킵] 읽기 실패: {nc} — {type(e).__name__}: {e}")
            skipped.append((date, nc))
            prev = None
            continue

        if lat_ref is None:
            lat_ref, lon_ref, ocean_mask = lat, lon, ocean

        # 정밀화: 직전 날 23~24시 UTC = (이 날 art00=전날총량) − (직전 cum23)
        if prev is not None:
            pdate, pcum23 = prev
            if (date - pdate) == pd.Timedelta(days=1):
                h23 = np.clip(art00 - pcum23, 0.0, None)
                kd = (pdate + pd.Timedelta(hours=23 + KST_OFFSET_H)).normalize()
                _add(kd, h23)

        # 이 날 0~22시 UTC → +9h 하여 KST 날짜에 배정
        for H in range(23):
            kd = (date + pd.Timedelta(hours=H + KST_OFFSET_H)).normalize()
            _add(kd, hourly[H])

        prev = (date, cum23)

    if skipped:
        print(f"  [경고] 읽기 실패 {len(skipped)}개 건너뜀")

    # 요청 연도이면서 24시간이 모두 모인 날짜만 남긴다 (경계 partial 제외)
    candidates = sorted(d for d in acc if d.year in years)
    incomplete = [d for d in candidates if hour_counts[d] != 24]
    if incomplete:
        print(f"  [경고] 24시간 미만 partial 일자 {len(incomplete)}개 제외: "
              f"{incomplete[0].date()} ~ {incomplete[-1].date()}")
    dates = [d for d in candidates if hour_counts[d] == 24]
    if not dates:
        raise RuntimeError("24시간이 완전한 KST 일강수 날짜가 없습니다.")

    arr = np.stack([acc[d] for d in dates], axis=2)        # (ny, nx, time)
    time = pd.DatetimeIndex(dates)

    # 바다(무효) 픽셀은 NaN 으로 복원
    if ocean_mask is not None:
        arr[ocean_mask, :] = np.nan

    # lat 오름차순으로 통일 (원본은 내림차순)
    if lat_ref[0] > lat_ref[-1]:
        lat_ref = lat_ref[::-1]
        arr = arr[::-1, :, :]

    for yr in years:
        m = time.year == yr
        if not m.any():
            continue
        ds_out = xr.Dataset(
            {"tp": (["lat", "lon", "time"], arr[:, :, m])},
            coords={"lat": lat_ref, "lon": lon_ref, "time": time[m]},
        )
        ds_out["tp"].attrs["units"] = "mm/day"
        ds_out["tp"].attrs["long_name"] = "Total Precipitation (KST daily)"
        ds_out.attrs["time_zone"] = "KST (UTC+9); daily boundary 00-24 KST"
        ds_out.attrs["method"] = "hourly de-accumulation -> +9h -> KST daily sum"
        path = os.path.join(save_dir, f"era5_land{yr}_KST.nc")
        ds_out.to_netcdf(path)
        print(f"  저장: {path}  {arr[:, :, m].shape}  "
              f"({time[m][0].date()} ~ {time[m][-1].date()})")
    return arr, lat_ref, lon_ref, time


# ==============================================================================
# STEP 2. 전체 병합
# ==============================================================================
def merge_yearly(years, save_dir):
    arrays, times = [], []
    lat_ref = lon_ref = None
    for yr in years:
        p = os.path.join(save_dir, f"era5_land{yr}_KST.nc")
        if not os.path.exists(p):
            continue
        with xr.open_dataset(p) as ds:
            lat = ds["lat"].values
            lon = ds["lon"].values
            if lat_ref is None:
                lat_ref, lon_ref = lat, lon
            elif not (np.array_equal(lat_ref, lat) and np.array_equal(lon_ref, lon)):
                raise ValueError(f"연도별 파일의 격자가 다름: {p}")
            arrays.append(ds["tp"].values)
            times.append(pd.DatetimeIndex(ds["time"].values))

    if not arrays:
        raise FileNotFoundError(f"병합할 연도별 KST 파일 없음: {save_dir}")

    full = np.concatenate(arrays, axis=2)
    t = times[0]
    for x in times[1:]:
        t = t.append(x)
    out = os.path.join(save_dir, "era5_land_P_KST.nc")
    ds_out = xr.Dataset({"tp": (["lat", "lon", "time"], full)},
                        coords={"lat": lat_ref, "lon": lon_ref, "time": t})
    ds_out["tp"].attrs["units"] = "mm/day"
    ds_out.attrs["time_zone"] = "KST (UTC+9)"
    ds_out.to_netcdf(out)
    print(f"병합 저장: {out}  {full.shape}")
    return out


# ==============================================================================
# 메인
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1. ERA5 누계해제 → KST 일강수")
    print("=" * 60)
    build_kst_daily(YEARS, ERA5_RAW_DIR, SAVE_DIR)

    print("=" * 60)
    print("STEP 2. 전체 병합")
    print("=" * 60)
    merge_yearly(YEARS, SAVE_DIR)
    print("\n완료. output/era5_land_P_KST.nc (KST 일경계 ERA5 강수)")
