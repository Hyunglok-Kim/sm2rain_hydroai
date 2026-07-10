"""
assemble.py  (data_pipeline 산출물 → sm2rain_pipeline 입력 조립)
==============================================================
data_pipeline 의 데이터셋별 산출물을 sm2rain_pipeline 이 읽는 두 입력파일로
합친다. 두 산출물은 서로 다른 시점에 필요하므로 단계를 나눠 실행한다.

  --step idw    : IDW/output 의 AWS·ASOS 격자를 합쳐 da_IDWs.nc 생성
                  → sm2rain_pipeline/01_SM2RAIN.py 가 캘리브레이션 기준으로 사용
  --step merge  : SM2RAIN·ERA5·GPM·AWS·ASOS 를 하나로 합쳐 ds_merged_LR.nc 생성
                  → sm2rain_pipeline/02·03·04 가 입력으로 사용
  --step all    : idw → merge 순서로 모두 (기본값. merge 는 SM2RAIN 산출이
                  있어야 하므로, 없으면 안내만 하고 건너뜀)

실행 순서 (전체 파이프라인)
  1) data_pipeline 다운로드·전처리·IDW 완료
  2) python assemble.py --step idw     → assembled/da_IDWs.nc
  3) da_IDWs.nc 를 sm2rain_pipeline/data/ 로 복사 후 01_SM2RAIN.py 실행
  4) 01 산출 SM2RAIN_KST.nc 준비 후
     python assemble.py --step merge   → assembled/ds_merged_LR.nc
  5) ds_merged_LR.nc 를 sm2rain_pipeline/data/ 로 복사 후 02·03·04 실행

격자
  모든 산출물을 0.125° 한국격자(lat 39→33, lon 124→130, 49×49)로 정렬한다.
  GPM·IDW·SM2RAIN 은 이미 이 격자이며(reindex 로 미세오차만 정렬), ERA5 만
  native 격자에서 bilinear regrid 한다.

입력
  IDW/output/Precipitation_IDW_{AWS|ASOS}_{YYYY}.nc
  GPM/output/GPM_{YYYY}_KST.nc
  ERA5_Land/output/era5_land_P_KST.nc
  SM2RAIN nc  (SM2RAIN_PATH; sm2rain_pipeline/01 산출물, merge 단계에만 필요)
출력
  assembled/da_IDWs.nc        (coords y·x 2D, time / "AWS (IDW)", "ASOS (IDW)")
  assembled/ds_merged_LR.nc   (time, lat, lon / SM2RAIN·ERA5·GPM·AWS·ASOS·LON·LAT)

* 이 스크립트는 실제 산출물이 있어야 검증되므로, 처음 돌릴 때는 각 소스의
  격자·시간축이 기대와 맞는지 로그를 확인할 것.
* 의존성: numpy, pandas, xarray, netCDF4
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

# ==============================================================================
# 설정
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
IDW_DIR = os.path.join(HERE, "IDW", "output")
GPM_DIR = os.path.join(HERE, "GPM", "output")
ERA5_PATH = os.path.join(HERE, "ERA5_Land", "output", "era5_land_P_KST.nc")
OUT_DIR = os.path.join(HERE, "assembled")

# SM2RAIN 산출(=sm2rain_pipeline/01 출력). 위치가 다르면 여기만 바꾼다.
SM2RAIN_PATH = os.path.join(HERE, "..", "sm2rain_pipeline", "output", "SM2RAIN_KST.nc")

YEARS = [2021, 2022, 2023, 2024, 2025]

# 표준 0.125° 한국격자 (GPM·IDW 와 동일)
TGT_LAT = np.linspace(39.0, 33.0, 49)          # 내림차순
TGT_LON = np.linspace(124.0, 130.0, 49)        # 오름차순
GRID_TOL = 0.02                                # reindex 최근접 허용오차(도)


# ==============================================================================
# 공통 헬퍼
# ==============================================================================
def _open_years(path_tpl, var, years):
    """연도별 nc 를 time 으로 concat 하여 (time, lat, lon) DataArray 반환."""
    das = []
    for yr in years:
        p = path_tpl.format(yr=yr)
        if not os.path.exists(p):
            continue
        ds = xr.open_dataset(p)
        da = ds[var]
        # (y,x,time) 또는 (time,lat,lon) 등 어떤 순서든 표준화는 뒤에서
        das.append(da.load())
        ds.close()
    if not das:
        raise FileNotFoundError(f"입력 없음: {path_tpl}")
    return xr.concat(das, dim="time").sortby("time")


def _to_latlon_time(da):
    """dim 을 (lat, lon, time) 이름으로 통일하고 1D lat/lon 좌표를 붙인다.
    (y,x) 2D 좌표(lat/lon 또는 y/x)면 1D 로 환원한다."""
    # 위·경도 좌표 후보 찾기
    coord_names = {c.lower(): c for c in list(da.coords) + list(da.dims)}
    lat_c = next((coord_names[k] for k in ("lat", "latitude", "y") if k in coord_names), None)
    lon_c = next((coord_names[k] for k in ("lon", "longitude", "x") if k in coord_names), None)
    if lat_c is None or lon_c is None:
        raise ValueError(f"위경도 좌표를 못 찾음: coords={list(da.coords)} dims={da.dims}")

    lat = np.asarray(da[lat_c].values)
    lon = np.asarray(da[lon_c].values)
    if lat.ndim == 2:
        lat = lat[:, 0]
    if lon.ndim == 2:
        lon = lon[0, :]

    # 공간 dim 이름 파악 (좌표가 2D면 dims 는 보통 y,x)
    spatial_dims = [d for d in da.dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"공간 dim 이 2개가 아님: {da.dims}")
    dy, dx = spatial_dims  # (행=위도, 열=경도) 가정
    da = da.rename({dy: "lat", dx: "lon"})
    da = da.assign_coords(lat=("lat", lat), lon=("lon", lon))
    return da.transpose("lat", "lon", "time")


def _reindex_canonical(da):
    """이미 표준격자에 가까운 소스를 정확히 TGT 좌표로 정렬(최근접 허용오차)."""
    return da.reindex(lat=TGT_LAT, lon=TGT_LON, method="nearest", tolerance=GRID_TOL)


# ── ERA5 native → 표준격자 bilinear regrid (scipy 불필요) ──────────────────
def _axis_matrix(src, tgt):
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
    """cube (nlat,nlon,T) → (len(tgt_lat),len(tgt_lon),T). src 는 오름차순."""
    Mlat = _axis_matrix(src_lat, tgt_lat)
    Mlon = _axis_matrix(src_lon, tgt_lon)
    step1 = np.tensordot(Mlat, cube, axes=([1], [0]))
    step2 = np.tensordot(Mlon, step1, axes=([1], [1]))
    return step2.transpose(1, 0, 2)


def _load_era5_regrid():
    ds = xr.open_dataset(ERA5_PATH)
    da = _to_latlon_time(ds["tp"])
    ds.close()
    src_lat = da["lat"].values
    src_lon = da["lon"].values
    cube = da.values                                    # (lat, lon, time)
    # bilinear 은 오름차순 축 필요 → 정렬 후 regrid
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]; cube = cube[::-1, :, :]
    if src_lon[0] > src_lon[-1]:
        src_lon = src_lon[::-1]; cube = cube[:, ::-1, :]
    tl = np.sort(TGT_LAT); to = np.sort(TGT_LON)
    rg = _regrid_bilinear(cube, src_lat, src_lon, tl, to)   # (49,49,T) on ascending
    out = xr.DataArray(rg, dims=("lat", "lon", "time"),
                       coords={"lat": tl, "lon": to, "time": da["time"].values})
    # TGT 순서(내림차순 lat)로 되돌림
    return out.reindex(lat=TGT_LAT, lon=TGT_LON)


# ==============================================================================
# STEP idw : da_IDWs.nc
# ==============================================================================
def build_da_idws():
    os.makedirs(OUT_DIR, exist_ok=True)
    aws = _open_years(os.path.join(IDW_DIR, "Precipitation_IDW_AWS_{yr}.nc"),
                      "precipitation", YEARS)
    asos = _open_years(os.path.join(IDW_DIR, "Precipitation_IDW_ASOS_{yr}.nc"),
                       "precipitation", YEARS)
    aws = _to_latlon_time(aws)
    asos = _to_latlon_time(asos)
    aws = _reindex_canonical(aws)
    asos = _reindex_canonical(asos)

    # da_IDWs 규약: y/x 2D 좌표 + 변수명 "AWS (IDW)" / "ASOS (IDW)", dims (time,y,x)
    lon2d, lat2d = np.meshgrid(TGT_LON, TGT_LAT)
    time = aws["time"].values
    ds_out = xr.Dataset(
        {
            "AWS (IDW)": (("time", "y", "x"), aws.transpose("time", "lat", "lon").values),
            "ASOS (IDW)": (("time", "y", "x"), asos.transpose("time", "lat", "lon").values),
        },
        coords={
            "y": (("y", "x"), lat2d.astype("float32")),
            "x": (("y", "x"), lon2d.astype("float32")),
            "time": time,
        },
        attrs={"description": "IDW gridded precipitation (AWS, ASOS) for SM2RAIN calibration"},
    )
    out = os.path.join(OUT_DIR, "da_IDWs.nc")
    if os.path.exists(out):
        os.remove(out)
    ds_out.to_netcdf(out)
    print(f"저장: {out}  time {pd.Timestamp(time[0]).date()}~{pd.Timestamp(time[-1]).date()} "
          f"({len(time)}일), 격자 {lat2d.shape}")
    return out


# ==============================================================================
# STEP merge : ds_merged_LR.nc
# ==============================================================================
def build_ds_merged():
    if not os.path.exists(SM2RAIN_PATH):
        print(f"[건너뜀] SM2RAIN 산출 없음: {SM2RAIN_PATH}\n"
              f"  → 먼저 da_IDWs 로 sm2rain_pipeline/01_SM2RAIN.py 를 돌려 SM2RAIN_KST.nc 를 만드세요.")
        return None
    os.makedirs(OUT_DIR, exist_ok=True)

    # 각 멤버 로드 → 표준격자·(lat,lon,time) 정렬
    sm = xr.open_dataset(SM2RAIN_PATH)
    sm2rain = _reindex_canonical(_to_latlon_time(sm["precipitation"])); sm.close()
    gpm = _reindex_canonical(_to_latlon_time(
        _open_years(os.path.join(GPM_DIR, "GPM_{yr}_KST.nc"), "precipitation", YEARS)))
    aws = _reindex_canonical(_to_latlon_time(
        _open_years(os.path.join(IDW_DIR, "Precipitation_IDW_AWS_{yr}.nc"), "precipitation", YEARS)))
    asos = _reindex_canonical(_to_latlon_time(
        _open_years(os.path.join(IDW_DIR, "Precipitation_IDW_ASOS_{yr}.nc"), "precipitation", YEARS)))
    era5 = _load_era5_regrid()

    members = {"SM2RAIN": sm2rain, "ERA5": era5, "GPM": gpm, "AWS": aws, "ASOS": asos}

    # 공통 시간축 (교집합) 으로 정렬
    common = None
    for da in members.values():
        t = pd.DatetimeIndex(da["time"].values).normalize()
        common = t if common is None else common.intersection(t)
    common = common.sort_values()
    if len(common) == 0:
        raise RuntimeError("멤버 간 공통 날짜가 없습니다. 각 소스의 기간을 확인하세요.")
    print(f"공통 시간축: {common[0].date()} ~ {common[-1].date()} ({len(common)}일)")

    data_vars = {}
    for name, da in members.items():
        da = da.assign_coords(time=pd.DatetimeIndex(da["time"].values).normalize())
        da = da.sel(time=common)
        data_vars[name] = (("time", "lat", "lon"),
                           da.transpose("time", "lat", "lon").values.astype("float64"))

    # LON/LAT 을 (time,lat,lon) 으로 브로드캐스트 (원본 ds_merged_LR 규약)
    lon2d, lat2d = np.meshgrid(TGT_LON, TGT_LAT)           # (lat, lon)
    nt = len(common)
    data_vars["LON"] = (("time", "lat", "lon"), np.broadcast_to(lon2d, (nt, 49, 49)).astype("float64"))
    data_vars["LAT"] = (("time", "lat", "lon"), np.broadcast_to(lat2d, (nt, 49, 49)).astype("float64"))

    ds_out = xr.Dataset(
        data_vars,
        coords={"time": common.values, "lat": TGT_LAT, "lon": TGT_LON},
        attrs={"description": "Merged members for TCA/BC (SM2RAIN, ERA5, GPM, AWS, ASOS)",
               "note": "TCA variable is added later by sm2rain_pipeline/02_TCA.py"},
    )
    out = os.path.join(OUT_DIR, "ds_merged_LR.nc")
    if os.path.exists(out):
        os.remove(out)
    ds_out.to_netcdf(out)
    print(f"저장: {out}  변수 {list(ds_out.data_vars)}  격자 (lat 49, lon 49)")
    return out


# ==============================================================================
# 메인
# ==============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="data_pipeline 산출물 조립")
    ap.add_argument("--step", choices=["idw", "merge", "all"], default="all")
    args = ap.parse_args()

    if args.step in ("idw", "all"):
        print("=" * 60); print("STEP idw : da_IDWs.nc"); print("=" * 60)
        build_da_idws()
    if args.step in ("merge", "all"):
        print("=" * 60); print("STEP merge : ds_merged_LR.nc"); print("=" * 60)
        build_ds_merged()
    print("\n완료. assembled/ 의 파일을 sm2rain_pipeline/data/ 로 복사해 사용하세요.")
