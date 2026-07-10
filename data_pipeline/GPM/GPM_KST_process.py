"""
GPM_KST_process.py  (GPM 30분 원본 → KST 일강수, 0.125° 한국격자)
================================================================
GPM_download.py 가 받은 IMERG 30분 원본(mm/hr)을 KST(+9h) 일경계로
재집계하여 0.125° 한국격자 일강수 nc 를 연도별로 만든다.

UTC 일경계(00~24 UTC)로 이미 합산된 일자료는 라벨만 바꿀 수 있을 뿐이라,
반드시 30분 원본 시각을 +9h 하여 KST 하루(00~24시 KST)로 다시 묶는다.
(GPM 30분 값은 순간 강수율이므로 0.5h 를 곱해 합산하면 됨)

핵심 절차
  1. 30분 파일명에서 UTC 시작시각 파싱 → +9h → KST 날짜에 배정
     (KST 하루 = UTC 15:00(전날) ~ 15:00(당일), 48개 30분 파일)
  2. 각 파일 precipitation(mm/hr) × 0.5h 를 KST 날짜별 누적 합산 → mm/day
     (native 0.1° 한국 subset 에서 누적하여 메모리 절약)
  3. KST 일강수를 0.125° 한국격자(lat 39→33, lon 124→130, 49×49)로 regrid
  4. 연도별 저장

입력 : raw/{YYYY}/{MM}/{DD}/3B-HHR.MS.MRG.3IMERG.*.HDF5  (GPM_download.py 산출)
       data/GPM_ref_mask.nc  (선택; 한국 육지마스크 참조용 기존 GPM nc.
                              없으면 마스크 미적용, 전 격자 출력)
출력 : output/GPM_{YYYY}_KST.nc   (lat, lon, time / mm/day, KST 일경계)

* 의존성: numpy, pandas, xarray, netCDF4, h5py, tqdm
"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import h5py
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
GPM_RAW_DIR = os.path.join(HERE, "raw")           # 30분 원본 루트
SAVE_DIR = os.path.join(HERE, "output")
REF_MASK_NC = os.path.join(HERE, "data", "GPM_ref_mask.nc")   # 선택 입력

YEARS = [2021, 2022, 2023, 2024, 2025]
KST_OFFSET_H = 9
FILL = -9999.9
HALF_HOUR = 0.5                                    # mm/hr → mm (30분)

# native 0.1° subset (한국+버퍼; regrid 보간 여유)
SUB_LAT_MIN, SUB_LAT_MAX = 32.8, 39.2
SUB_LON_MIN, SUB_LON_MAX = 123.8, 130.2

# 출력 0.125° 한국격자 (SM2RAIN/ASOS/AWS 와 동일 격자)
TGT_LAT = np.linspace(39.0, 33.0, 49)              # 내림차순
TGT_LON = np.linspace(124.0, 130.0, 49)            # 오름차순

# 파일명에서 UTC 시작시각 추출: ...3IMERG.20210101-S000000-E002959...
_FN_RE = re.compile(r"3IMERG\.(\d{8})-S(\d{6})-E")

# native subset 인덱스 캐시 (첫 파일에서 1회 계산)
_sub = {"lat": None, "lon": None, "ila": None, "ilo": None}


# ==============================================================================
# 30분 파일 읽기
# ==============================================================================
def _read_30min(nc_path):
    """30분 파일 → native subset 강수율 (lat, lon) mm/hr."""
    with h5py.File(nc_path, "r") as f:
        g = f["Grid"]
        if _sub["lat"] is None:
            lat = g["lat"][:]                       # (1800,) 오름차순
            lon = g["lon"][:]                       # (3600,) 오름차순
            ila = np.where((lat >= SUB_LAT_MIN) & (lat <= SUB_LAT_MAX))[0]
            ilo = np.where((lon >= SUB_LON_MIN) & (lon <= SUB_LON_MAX))[0]
            _sub.update(lat=lat[ila], lon=lon[ilo], ila=ila, ilo=ilo)
        ila, ilo = _sub["ila"], _sub["ilo"]
        # precip: (1, lon, lat) → subset → transpose (lat, lon)
        p = g["precipitation"][0, ilo[0]:ilo[-1] + 1, ila[0]:ila[-1] + 1]
    p = np.asarray(p, dtype="float64").T
    p[p == FILL] = np.nan
    return p


def list_utc_files(years, raw_dir):
    """반환: [(utc_start_timestamp, path), ...] 정렬.
    KST 1/1 경계를 채우기 위해 직전 해 12/31 하루도 포함."""
    need_years = sorted(set(years) | {min(years) - 1})
    files = []
    for yr in need_years:
        yr_dir = os.path.join(raw_dir, str(yr))
        if not os.path.isdir(yr_dir):
            continue
        for mm in sorted(os.listdir(yr_dir)):
            mdir = os.path.join(yr_dir, mm)
            if not os.path.isdir(mdir) or mm.startswith("."):
                continue
            for dd in sorted(os.listdir(mdir)):
                ddir = os.path.join(mdir, dd)
                if not os.path.isdir(ddir) or dd.startswith("."):
                    continue
                # 직전 해는 12/31 하루만
                if int(yr) == min(years) - 1 and not (mm == "12" and dd == "31"):
                    continue
                for fp in glob.glob(os.path.join(ddir, "*3IMERG*.HDF5")):
                    m = _FN_RE.search(os.path.basename(fp))
                    if not m:
                        continue
                    date_s, hms = m.group(1), m.group(2)
                    ts = pd.Timestamp(f"{date_s} {hms[:2]}:{hms[2:4]}:{hms[4:6]}")
                    files.append((ts, fp))
    return sorted(files, key=lambda x: x[0])


# ==============================================================================
# STEP 1. KST 일강수 누적 (native 0.1° subset)
# ==============================================================================
def build_kst_daily_native(years, raw_dir):
    utc_files = list_utc_files(years, raw_dir)
    if not utc_files:
        raise FileNotFoundError(f"입력 30분 파일 없음: {raw_dir} (먼저 GPM_download.py 실행)")
    print(f"대상 30분 파일: {len(utc_files):,}개 "
          f"({utc_files[0][0]} ~ {utc_files[-1][0]} UTC)")

    acc = {}                          # kst_date -> (lat,lon) mm 누적
    count = defaultdict(int)          # kst_date -> 30분 파일 수 (검증용)

    for ts, fp in tqdm(utc_files, desc="GPM +9h KST 누적"):
        kst = ts + pd.Timedelta(hours=KST_OFFSET_H)
        kd = kst.normalize()                        # KST 날짜
        rate = _read_30min(fp)                      # (lat,lon) mm/hr
        val = np.nan_to_num(rate, nan=0.0) * HALF_HOUR
        if kd not in acc:
            acc[kd] = np.zeros(val.shape, dtype="float64")
        acc[kd] += val
        count[kd] += 1

    return acc, count, _sub["lat"].copy(), _sub["lon"].copy()


# ==============================================================================
# 한국 육지마스크 (선택; 참조 nc 의 항상-NaN 픽셀 = 바다)
# ==============================================================================
def _load_ref_mask():
    """반환: (49,49) bool, True=유효(육지). 참조 nc 없으면 None."""
    if not os.path.exists(REF_MASK_NC):
        print(f"  [안내] 참조 마스크 없음({REF_MASK_NC}) → 육지마스크 미적용(전 격자 출력)")
        return None
    ds = xr.open_dataset(REF_MASK_NC)
    var = "precipitation" if "precipitation" in ds else list(ds.data_vars)[0]
    v = ds[var].values
    ds.close()
    if v.ndim == 3:                                 # (lat,lon,time) 가정
        return ~np.isnan(v).all(axis=-1)
    return ~np.isnan(v)


# ==============================================================================
# 분리형 bilinear regrid (scipy 불필요, numpy 만 사용)
# ==============================================================================
def _axis_matrix(src, tgt):
    """1축 선형보간 가중행렬 (len(tgt), len(src)). src 는 오름차순이어야 함."""
    src = np.asarray(src, dtype="float64")
    tgt = np.asarray(tgt, dtype="float64")
    idx = np.clip(np.searchsorted(src, tgt) - 1, 0, len(src) - 2)
    x0, x1 = src[idx], src[idx + 1]
    w = np.clip((tgt - x0) / (x1 - x0), 0.0, 1.0)
    M = np.zeros((len(tgt), len(src)), dtype="float64")
    rows = np.arange(len(tgt))
    M[rows, idx] = 1.0 - w
    M[rows, idx + 1] = w
    return M


def _regrid_bilinear(cube, src_lat, src_lon, tgt_lat, tgt_lon):
    """cube (nlat,nlon,T) → (len(tgt_lat),len(tgt_lon),T)."""
    Mlat = _axis_matrix(src_lat, tgt_lat)
    Mlon = _axis_matrix(src_lon, tgt_lon)
    step1 = np.tensordot(Mlat, cube, axes=([1], [0]))
    step2 = np.tensordot(Mlon, step1, axes=([1], [1]))
    return step2.transpose(1, 0, 2)


# ==============================================================================
# STEP 2. 0.125° 한국격자 regrid + 연도별 저장
# ==============================================================================
def regrid_and_save(acc, count, nat_lat, nat_lon, years, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    dates = sorted(d for d in acc if d.year in years)
    if not dates:
        raise RuntimeError("요청 연도에 해당하는 KST 일자 없음")

    cube = np.stack([acc[d] for d in dates], axis=2)          # (nlat, nlon, T)
    rg = _regrid_bilinear(cube, nat_lat, nat_lon, TGT_LAT, TGT_LON)
    rg = np.clip(rg, 0.0, None)

    land_mask = _load_ref_mask()
    if land_mask is not None:
        rg = np.where(land_mask[:, :, None], rg, np.nan)

    da_rg = xr.DataArray(
        rg, dims=["lat", "lon", "time"],
        coords={"lat": TGT_LAT, "lon": TGT_LON, "time": pd.DatetimeIndex(dates)},
    )

    time_idx = pd.DatetimeIndex(dates)
    for yr in years:
        m = time_idx.year == yr
        if not m.any():
            continue
        sub = da_rg.isel(time=np.where(m)[0])
        ds_out = xr.Dataset({"precipitation": sub})
        ds_out["precipitation"].attrs.update(
            units="mm/day", long_name="GPM Resampled Precipitation (KST daily)")
        ds_out.attrs["time_zone"] = "KST (UTC+9); daily boundary 00-24 KST"
        ds_out.attrs["method"] = ("30-min mm/hr x0.5h -> +9h -> KST daily sum "
                                  "-> 0.125deg regrid")
        path = os.path.join(save_dir, f"GPM_{yr}_KST.nc")
        ds_out.to_netcdf(path)
        # 검증: 해당 연도 KST 일자별 30분 파일 수 (정상=48)
        cnts = [count[d] for d in time_idx[m]]
        n_full = sum(1 for c in cnts if c == 48)
        print(f"  저장: {path}  {tuple(sub.shape)}  "
              f"({time_idx[m][0].date()} ~ {time_idx[m][-1].date()})  "
              f"완전일(48파일) {n_full}/{int(m.sum())}")
    return da_rg


# ==============================================================================
# 메인
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1. GPM 30분 → +9h → KST 일강수 누적 (native 0.1°)")
    print("=" * 60)
    acc, count, nat_lat, nat_lon = build_kst_daily_native(YEARS, GPM_RAW_DIR)

    print("=" * 60)
    print("STEP 2. 0.125° 한국격자 regrid + 연도별 저장")
    print("=" * 60)
    regrid_and_save(acc, count, nat_lat, nat_lon, YEARS, SAVE_DIR)
    print("\n완료. output/GPM_{YYYY}_KST.nc (KST 일경계, 0.125° 한국격자)")
