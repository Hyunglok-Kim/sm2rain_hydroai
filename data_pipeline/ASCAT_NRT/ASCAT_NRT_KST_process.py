"""
ASCAT_NRT_KST_process.py  (원본 → 한국 0.1° 격자 → KST 일별 스택, 한 파일)
==========================================================================
ASCAT_NRT_download.py 가 받은 원본(.zip 안의 .nat)을 3단계로 처리해
SM2RAIN 파이프라인 입력(ASCAT_daily_stack_KST.nc) 하나를 만든다.

  STEP A. 변환   : raw/*.zip(.nat) → swath .nc          (ascat 패키지)
  STEP B. 격자화 : swath .nc → 한국 0.1° 격자 .nc        (cKDTree 최근접)
  STEP C. 스택   : 전체 concat → +9h(KST) → 일평균 →
                   output/ASCAT_daily_stack_KST.nc

시간대 처리
  swath time 은 실제 UTC 관측시각이므로, 일평균 직전에 +9h 하여
  KST 하루(00~24시 KST) 경계로 재집계한다.
  (이미 일합산된 자료는 라벨만 바뀌므로 반드시 swath 단계에서 처리)

변수
  sm                   : ASCAT 원본 sm (degree of saturation, 0~100 %)
  degree_of_saturation : sm / 100  (0~1)
  sm_volumetric        : (sm/100) × porosity  (porosity 없으면 NaN)
  optimum_water_content: porosity (포화 함수율)

입력 : raw/                     (다운로드 원본 .zip 또는 .nat)
       data/porosity.nc         (선택; TU Wien static layer. 없으면
                                 sm_volumetric = NaN 로 진행)
중간 : work_swath/  work_grid/  (재실행 시 이미 처리된 파일은 건너뜀)
출력 : output/ASCAT_daily_stack_KST.nc   (time, y, x / 2D lat·lon 좌표)

* 출력 격자: lat 33.0–38.7, lon 125.5–130.0, 0.1° (58×46) — 대한민국 본토+제주
* 의존성: ascat, xarray, netCDF4, numpy, pandas, scipy, tqdm
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - 최소 환경 대비 편의 처리
    def tqdm(iterable, *args, **kwargs):
        return iterable

# BLAS/OMP 스레드가 워커 수와 곱해져 폭주하지 않도록 억제
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

# ==============================================================================
# 경로 및 설정
# ==============================================================================
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = HERE / "raw"                 # 다운로드 원본 (.zip / .nat)
SWATH_DIR = HERE / "work_swath"        # STEP A 결과 (swath .nc)
GRID_DIR = HERE / "work_grid"          # STEP B 결과 (격자화 .nc)
OUT_DIR = HERE / "output"
OUT_NAME = "ASCAT_daily_stack_KST.nc"
POROSITY_PATH = HERE / "data" / "porosity.nc"   # 선택 입력

# 한국 격자 범위 (본토 34.3–38.6N/126.0–129.6E + 제주 33.1–33.6N)
LON_MIN, LON_MAX = 125.5, 130.0
LAT_MIN, LAT_MAX = 33.0, 38.7
RES_DEG = 0.1
MARGIN_DEG = 0.5          # 관측 선필터 여유 (격자 경계 근처 관측 포함)
NEAREST_FACTOR = 1.5      # 최근접 허용거리 = RES_DEG × FACTOR

KST_OFFSET_HOURS = 9
N_WORKERS = 5             # STEP A/B 병렬 워커 수
CHUNK_SIZE = 500          # STEP C concat 청크 (메모리 절약)


# ==============================================================================
# STEP A. 원본(.zip/.nat) → swath .nc
# ==============================================================================
def convert_one_raw(raw_path: Path) -> str:
    """zip 이면 풀어서 .nat 추출, ascat 패키지로 읽어 swath .nc 저장."""
    from ascat.eumetsat.level2 import AscatL2File   # 지연 임포트 (STEP A에만 필요)

    out_file = SWATH_DIR / (raw_path.stem + ".nc")
    if out_file.exists() and out_file.stat().st_size > 0:
        return "skip"

    try:
        if raw_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(raw_path, "r") as zf:
                    zf.extractall(tmpdir)
                nats = [f for f in Path(tmpdir).rglob("*.nat")
                        if not f.name.startswith("._")]
                if not nats:
                    return "err: zip 안에 .nat 없음"
                ds, _meta = AscatL2File(str(nats[0])).read(to_xarray=True)
                ds.to_netcdf(out_file)
                ds.close()
        else:  # .nat 직접
            ds, _meta = AscatL2File(str(raw_path)).read(to_xarray=True)
            ds.to_netcdf(out_file)
            ds.close()
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"err: {type(e).__name__}: {e}"


def step_a_convert() -> None:
    SWATH_DIR.mkdir(parents=True, exist_ok=True)
    raws = sorted(
        [p for p in RAW_DIR.glob("*")
         if p.suffix.lower() in (".zip", ".nat") and not p.name.startswith("._")]
    )
    if not raws:
        raise FileNotFoundError(f"원본 없음: {RAW_DIR} (먼저 ASCAT_NRT_download.py 실행)")
    print(f"STEP A. 원본 {len(raws)}개 → swath .nc 변환")

    n = {"ok": 0, "skip": 0, "err": 0}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(convert_one_raw, p): p for p in raws}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  변환"):
            status = fut.result()
            key = status.split(":")[0] if status.startswith("err") else status
            n[key] = n.get(key, 0) + 1
            if status.startswith("err"):
                print(f"  [실패] {futs[fut].name}: {status}")
    print(f"  완료: ok={n.get('ok',0)}, skip={n.get('skip',0)}, err={n.get('err',0)}")


# ==============================================================================
# STEP B. swath .nc → 한국 0.1° 격자 .nc
# ==============================================================================
def build_korea_grid() -> tuple[np.ndarray, np.ndarray]:
    """한국용 정규 격자 — 2D lon/lat 반환 (북→남 정렬, dim 은 (y, x))."""
    lon_1d = np.arange(LON_MIN, LON_MAX + RES_DEG / 2.0, RES_DEG)
    lat_1d = np.arange(LAT_MIN, LAT_MAX + RES_DEG / 2.0, RES_DEG)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    lat_2d = np.flipud(lat_2d)
    return lon_2d.astype("float32"), lat_2d.astype("float32")


def load_porosity_on_grid(lon_2d, lat_2d) -> np.ndarray | None:
    """porosity.nc 를 격자에 최근접 매핑. 없으면 None (sm_volumetric=NaN)."""
    if not POROSITY_PATH.exists():
        print(f"  [안내] porosity 없음({POROSITY_PATH}) → sm_volumetric=NaN 으로 진행")
        return None
    ds = xr.open_dataset(POROSITY_PATH)
    df = ds.to_dataframe().reset_index()
    ds.close()
    col = "por_hwsd" if "por_hwsd" in df.columns else next(
        (c for c in df.columns
         if c not in ("lat", "lon") and np.issubdtype(df[c].dtype, np.number)),
        None,
    )
    if col is None or not {"lat", "lon"} <= set(df.columns):
        print("  [안내] porosity 형식 인식 실패 → 사용 안 함")
        return None
    df = df.dropna(subset=["lat", "lon", col])
    df = df[(df["lon"] > LON_MIN - MARGIN_DEG) & (df["lon"] < LON_MAX + MARGIN_DEG)
            & (df["lat"] > LAT_MIN - MARGIN_DEG) & (df["lat"] < LAT_MAX + MARGIN_DEG)]
    if df.empty:
        return None
    tree = cKDTree(np.column_stack([df["lon"], df["lat"]]))
    gpts = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])
    dist, idx = tree.query(gpts, k=1)
    out = np.full(lon_2d.shape, np.nan, dtype="float32")
    ok = dist <= (RES_DEG * NEAREST_FACTOR)
    out.ravel()[ok] = df[col].to_numpy(dtype="float32")[idx[ok]]
    print(f"  porosity 격자 매핑 완료 (유효 {int(ok.sum())} 격자)")
    return out


def swath_to_grid(lons, lats, values, grid_lon, grid_lat) -> np.ndarray:
    """관측점 → 격자 최근접 매핑 (허용거리 밖은 NaN)."""
    vals = np.asarray(values, dtype="float32")
    valid = np.isfinite(vals) & np.isfinite(lons) & np.isfinite(lats)
    out = np.full(grid_lon.shape, np.nan, dtype="float32")
    if not valid.any():
        return out
    tree = cKDTree(np.column_stack([lons[valid], lats[valid]]))
    gpts = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    dist, idx = tree.query(gpts, k=1)
    ok = dist <= (RES_DEG * NEAREST_FACTOR)
    out.ravel()[ok] = vals[valid][idx[ok]]
    return out


def grid_one_swath(nc_path: str, lon_2d, lat_2d, porosity_grid) -> str:
    try:
        with xr.open_dataset(nc_path) as ds:
            if {"sm", "lat", "lon", "time"} - set(ds.variables):
                return "skip"
            lon = ds["lon"].values
            lat = ds["lat"].values
            m = MARGIN_DEG
            mask = ((lon > LON_MIN - m) & (lon < LON_MAX + m)
                    & (lat > LAT_MIN - m) & (lat < LAT_MAX + m))
            if not mask.any():
                return "skip"                      # 한국 영역 밖 swath
            sm_pct = ds["sm"].values[mask].astype("float32")
            if not np.isfinite(sm_pct).any():
                return "skip"
            lon_k = lon[mask].astype("float64")
            lat_k = lat[mask].astype("float64")
            t_ns = pd.to_datetime(ds["time"].values[mask]).asi8
            t_repr = pd.Timestamp(np.int64(t_ns.mean()), unit="ns")

        grid_sm = swath_to_grid(lon_k, lat_k, sm_pct, lon_2d, lat_2d)
        grid_sat = grid_sm / 100.0
        if porosity_grid is not None:
            grid_vol = (grid_sat * porosity_grid).astype("float32")
            grid_owc = porosity_grid.astype("float32")
        else:
            grid_vol = np.full_like(grid_sat, np.nan)
            grid_owc = np.full_like(grid_sat, np.nan)

        def _var(arr, attrs):
            return (("time", "y", "x"), arr[np.newaxis, :, :], attrs)

        dset = xr.Dataset(
            data_vars={
                "sm": _var(grid_sm, {"long_name": "ASCAT soil moisture (raw)",
                                     "units": "percent"}),
                "degree_of_saturation": _var(grid_sat.astype("float32"),
                                             {"long_name": "sm/100", "units": "1"}),
                "sm_volumetric": _var(grid_vol,
                                      {"long_name": "sat x porosity",
                                       "units": "m3 m-3"}),
                "optimum_water_content": _var(grid_owc,
                                              {"long_name": "porosity",
                                               "units": "m3 m-3"}),
            },
            coords={"time": ("time", [t_repr]),
                    "lat": (("y", "x"), lat_2d),
                    "lon": (("y", "x"), lon_2d)},
            attrs={"description": "ASCAT NRT L2 SMR regridded to 0.1deg Korea grid",
                   "source_file": os.path.basename(nc_path)},
        )
        out_path = GRID_DIR / f"{t_repr.strftime('%Y%m%dT%H%M%S')}__{Path(nc_path).stem}.nc"
        enc = {v: dict(zlib=True, complevel=4, dtype="float32") for v in dset.data_vars}
        dset.to_netcdf(out_path, encoding=enc)
        dset.close()
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"err: {type(e).__name__}: {e}"


def step_b_grid() -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in SWATH_DIR.glob("*.nc") if not p.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"swath .nc 없음: {SWATH_DIR}")

    # 이미 격자화된 swath 는 건너뜀 (파일명 뒤쪽 stem 매칭)
    done = {p.stem.split("__", 1)[-1] for p in GRID_DIR.glob("*.nc")}
    todo = [f for f in files if Path(f).stem not in done]
    print(f"STEP B. 격자화: 전체 {len(files)}, 처리 대상 {len(todo)} (이미 완료 {len(files)-len(todo)})")

    lon_2d, lat_2d = build_korea_grid()
    print(f"  격자 {lon_2d.shape} (lat {LAT_MIN}–{LAT_MAX}, lon {LON_MIN}–{LON_MAX}, {RES_DEG}°)")
    porosity_grid = load_porosity_on_grid(lon_2d, lat_2d)

    n = {"ok": 0, "skip": 0, "err": 0}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(grid_one_swath, f, lon_2d, lat_2d, porosity_grid): f
                for f in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  격자화"):
            status = fut.result()
            key = "err" if status.startswith("err") else status
            n[key] += 1
            if status.startswith("err"):
                print(f"  [실패] {Path(futs[fut]).name}: {status}")
    print(f"  완료: ok={n['ok']}, skip={n['skip']}, err={n['err']}")


# ==============================================================================
# STEP C. 스택 → +9h(KST) → 일평균 → 단일 .nc
# ==============================================================================
def step_c_stack() -> None:
    files = sorted(str(p) for p in GRID_DIR.glob("*.nc") if not p.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"격자화 .nc 없음: {GRID_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    print(f"STEP C. 스택: {len(files)}개 → {out_path}")

    parts = []
    for i in tqdm(range(0, len(files), CHUNK_SIZE), desc="  청크 로드"):
        dss = []
        for f in files[i:i + CHUNK_SIZE]:
            try:
                dss.append(xr.open_dataset(f).load())
            except Exception as e:  # noqa: BLE001
                print(f"  [읽기 실패] {os.path.basename(f)}: {e}")
        if dss:
            parts.append(xr.concat(dss, dim="time", coords="minimal",
                                   compat="override", join="override"))
            for d in dss:
                d.close()

    stacked = xr.concat(parts, dim="time", coords="minimal",
                        compat="override", join="override")
    stacked = stacked.sortby("time")

    # 중복 시각 제거
    _, uniq = np.unique(stacked["time"].values, return_index=True)
    if len(uniq) != stacked["time"].size:
        print(f"  중복 시각 제거: {stacked['time'].size} → {len(uniq)}")
        stacked = stacked.isel(time=np.sort(uniq))

    # UTC → KST (+9h) 후 일평균 → KST 달력일 기준 재집계
    print(f"  UTC → KST(+{KST_OFFSET_HOURS}h) 변환 후 일평균...")
    stacked = stacked.assign_coords(
        time=stacked["time"] + np.timedelta64(KST_OFFSET_HOURS, "h"))
    daily = stacked.resample(time="1D").mean(skipna=True)

    for v in daily.data_vars:   # resample 이 잃는 변수 attrs 복구
        daily[v].attrs = dict(stacked[v].attrs)
    daily.attrs.update({
        "description": "ASCAT NRT L2 SMR, Korea 0.1deg grid, KST daily mean",
        "n_days": int(daily["time"].size),
        "time_aggregation": "daily mean",
        "time_zone": f"KST (UTC+{KST_OFFSET_HOURS}); daily boundary 00-24 KST",
    })

    enc = {v: dict(zlib=True, complevel=4, dtype="float32") for v in daily.data_vars}
    if out_path.exists():
        out_path.unlink()
    daily.to_netcdf(out_path, encoding=enc)
    print(f"  저장 완료: {out_path} (days={int(daily['time'].size)})")
    daily.close()
    stacked.close()


# ==============================================================================
# 메인
# ==============================================================================
if __name__ == "__main__":
    step_a_convert()
    step_b_grid()
    step_c_stack()
    print("\n완료. output/ASCAT_daily_stack_KST.nc 를 sm2rain_pipeline/data/ 로 옮겨 사용.")
