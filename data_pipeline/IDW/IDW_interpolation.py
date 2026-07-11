"""
IDW_interpolation.py  (지점 일강수 → 역거리가중 격자 강수)
========================================================
ASOS/AWS 지점 일강수 CSV를 역거리가중(IDW)으로 0.125° 한국격자 일강수로
보간하여 연도별 nc 로 저장한다. (보간=AWS, 검증=ASOS 규약에 맞춰 두 관측망을
각각 산출)

방법
  1. 지점 좌표·일강수 로드 (AWS는 CSV에 좌표 포함, ASOS는 관측소 메타와 병합)
  2. 셀평균(cell-mean) 전처리: 같은 목표격자 셀에 여러 지점이 몰릴 때(특히
     조밀한 AWS) 셀 안 지점을 먼저 평균내어 과대가중을 막는다.
  3. 역거리가중 보간: 격자점 g 의 값 = Σ(w_i·z_i)/Σw_i,  w_i = 1/d_i^power
     (격자점이 지점과 정확히 겹치면 그 지점값 사용)
  4. 한국 육지마스크(선택; shapefile) 적용 후 연도별 저장

입력 : ../ASOS/output/ASOS_daily_*.csv   (지점, 지점명, 일시, 강수량)
       ../AWS/output/AWS_daily_*.csv     (일시, 지점, 경도, 위도, 고도, 강수량)
       data/Station_ASOS.csv  (ASOS 좌표 메타; '지점'+경도관서/위도관서, 필수)
       data/Korea.shp         (한국 경계; 선택, 없으면 전 격자 출력)
출력 : output/Precipitation_IDW_{AWS|ASOS}_{YYYY}.nc  (y, x, time / mm/day)

* 출력 격자는 SM2RAIN·GPM 파이프라인과 동일한 0.125° 49×49 (lat 39→33,
  lon 124→130) → da_IDWs 로 바로 이어 쓸 수 있다.
* 원본 지점 자료가 KST 일자료이므로 별도 시간대 처리는 없다.
* 의존성: numpy, pandas, xarray, netCDF4, tqdm (shapefile 클립 시 geopandas·shapely)
"""
from __future__ import annotations

import glob
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
DATA_DIR = os.path.join(HERE, "data")          # 관측소 메타·shapefile
OUT_DIR = os.path.join(HERE, "output")

# 관측망별 입력 CSV(다운로드 산출)와 좌표 메타
NETWORKS = {
    "AWS": {
        "csv_glob": os.path.join(HERE, "..", "AWS", "output", "AWS_daily_*.csv"),
        "meta": None,                          # 좌표가 CSV에 포함됨
    },
    "ASOS": {
        "csv_glob": os.path.join(HERE, "..", "ASOS", "output", "ASOS_daily_*.csv"),
        "meta": os.path.join(DATA_DIR, "Station_ASOS.csv"),
    },
}
TARGETS = ["AWS", "ASOS"]                       # 산출할 관측망

SHP_FILE = os.path.join(DATA_DIR, "Korea.shp")  # 선택 (없으면 클립 생략)

YEARS = [2021, 2022, 2023, 2024, 2025]
POWER = 3                                       # IDW 거리 지수
EPS = 1e-12

# 출력 0.125° 한국격자 (SM2RAIN/GPM 과 동일)
TGT_LAT = np.linspace(39.0, 33.0, 49)          # 내림차순
TGT_LON = np.linspace(124.0, 130.0, 49)        # 오름차순


# ==============================================================================
# 입력 로딩
# ==============================================================================
def _read_csv_any(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def _pick_cols(df, lon_names, lat_names):
    lon = next((c for c in lon_names if c in df.columns), None)
    lat = next((c for c in lat_names if c in df.columns), None)
    return lon, lat


def load_network(name, cfg):
    """관측망 일강수를 (일시, 지점, 경도, 위도, 강수량) 표준형으로 반환."""
    files = sorted(glob.glob(cfg["csv_glob"]))
    if not files:
        raise FileNotFoundError(f"{name} 일강수 CSV 없음: {cfg['csv_glob']}")
    df = pd.concat([_read_csv_any(f) for f in files], ignore_index=True)
    df.columns = df.columns.str.strip()
    df["지점"] = pd.to_numeric(df["지점"], errors="coerce").astype("Int64")
    df["일시"] = pd.to_datetime(df["일시"], errors="coerce")
    df["강수량"] = pd.to_numeric(df["강수량"], errors="coerce").fillna(0.0)

    lon_c, lat_c = _pick_cols(df, ["경도", "경도관서", "lon"], ["위도", "위도관서", "lat"])
    if lon_c is None or lat_c is None:
        # 데이터 CSV에 좌표가 없으면 관측소 메타와 병합 (ASOS)
        if not cfg["meta"] or not os.path.exists(cfg["meta"]):
            raise FileNotFoundError(
                f"{name}: 좌표 컬럼이 없고 메타도 없음 → data/Station_{name}.csv 필요")
        meta = _read_csv_any(cfg["meta"])
        meta.columns = meta.columns.str.strip()
        meta["지점"] = pd.to_numeric(meta["지점"], errors="coerce").astype("Int64")
        m_lon, m_lat = _pick_cols(meta, ["경도관서", "경도", "lon"], ["위도관서", "위도", "lat"])
        df = df.merge(meta[["지점", m_lon, m_lat]], on="지점", how="left")
        lon_c, lat_c = m_lon, m_lat

    df = df.rename(columns={lon_c: "경도", lat_c: "위도"})
    df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
    df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
    df = df.dropna(subset=["일시", "경도", "위도"])
    df["date"] = df["일시"].dt.normalize()

    # 같은 지점·날짜 중복은 합산(시간자료가 섞여 들어온 경우 대비)
    df = (df.groupby(["date", "지점"], as_index=False)
            .agg(강수량=("강수량", "sum"), 경도=("경도", "first"), 위도=("위도", "first")))
    print(f"  {name}: {df['지점'].nunique()}개 지점, "
          f"{df['date'].min().date()}~{df['date'].max().date()}")
    return df


# ==============================================================================
# 한국 육지마스크 (선택)
# ==============================================================================
def build_land_mask(grid_lon, grid_lat):
    """shapefile 이 있으면 한국 경계 안=True 마스크, 없으면 전부 True."""
    if not os.path.exists(SHP_FILE):
        print(f"  [안내] shapefile 없음({SHP_FILE}) → 육지마스크 미적용(전 격자 출력)")
        return np.ones(grid_lon.shape, dtype=bool)
    import geopandas as gpd                     # 선택 의존성 (지연 임포트)
    from shapely.geometry import Point
    from shapely.prepared import prep

    gdf = gpd.read_file(SHP_FILE)
    if gdf.crs is None:
        gdf.set_crs("EPSG:5179", inplace=True)
    gdf = gdf.to_crs("EPSG:4326")
    try:
        poly = gdf.union_all()
    except AttributeError:
        poly = gdf.unary_union
    prepared = prep(poly)
    mask = np.zeros(grid_lon.shape, dtype=bool)
    for r in range(grid_lon.shape[0]):
        for c in range(grid_lon.shape[1]):
            mask[r, c] = prepared.contains(Point(grid_lon[r, c], grid_lat[r, c]))
    print(f"  육지마스크 유효 격자 {int(mask.sum())} / {mask.size}")
    return mask


# ==============================================================================
# IDW 보간 (셀평균 전처리 포함)
# ==============================================================================
def _cell_mean(x_obs, y_obs, z_obs, lon_centers, lat_centers):
    """같은 목표격자 셀 안 지점을 평균 → (셀중심 x, y, 평균 z)."""
    dx = np.median(np.diff(lon_centers))
    dy = np.median(np.diff(lat_centers))
    lon_edges = np.concatenate([[lon_centers[0] - dx / 2],
                                (lon_centers[:-1] + lon_centers[1:]) / 2,
                                [lon_centers[-1] + dx / 2]])
    lat_edges = np.concatenate([[lat_centers[0] - dy / 2],
                                (lat_centers[:-1] + lat_centers[1:]) / 2,
                                [lat_centers[-1] + dy / 2]])
    ix = np.digitize(x_obs, lon_edges) - 1
    iy = np.digitize(y_obs, lat_edges) - 1
    ok = ((ix >= 0) & (ix < len(lon_centers))
          & (iy >= 0) & (iy < len(lat_centers)) & np.isfinite(z_obs))
    cells = pd.DataFrame({"ix": ix[ok], "iy": iy[ok], "z": z_obs[ok]})
    if cells.empty:
        return np.array([]), np.array([]), np.array([])
    agg = cells.groupby(["iy", "ix"], as_index=False)["z"].mean()
    return (lon_centers[agg["ix"].values],
            lat_centers[agg["iy"].values],
            agg["z"].values)


def idw_grid(x_obs, y_obs, z_obs, grid_lon, grid_lat, power=POWER):
    """역거리가중 보간 (격자점 전체를 벡터화). 반환 (ny, nx)."""
    gx = grid_lon.ravel()[:, None]              # (G, 1)
    gy = grid_lat.ravel()[:, None]
    d = np.sqrt((gx - x_obs[None, :]) ** 2 + (gy - y_obs[None, :]) ** 2)  # (G, N)
    w = 1.0 / np.maximum(d, EPS) ** power
    zi = (w @ z_obs) / w.sum(axis=1)
    # 격자점이 지점과 정확히 겹치면 그 지점값으로 대체
    hit_rows = np.any(d == 0.0, axis=1)
    if hit_rows.any():
        first_hit = np.argmax(d[hit_rows] == 0.0, axis=1)
        zi[hit_rows] = z_obs[first_hit]
    return zi.reshape(grid_lon.shape)


# ==============================================================================
# 관측망 1종 → 연도별 nc
# ==============================================================================
def interpolate_network(name, cfg, grid_lon, grid_lat, land_mask):
    df = load_network(name, cfg)
    lon_centers = np.sort(grid_lon[0, :])
    lat_centers = np.sort(grid_lat[:, 0])

    os.makedirs(OUT_DIR, exist_ok=True)
    for yr in YEARS:
        dates = pd.date_range(f"{yr}-01-01", f"{yr}-12-31")
        grids = []
        for cur in tqdm(dates, desc=f"IDW {name} {yr}", leave=False):
            day = df[df["date"] == cur]
            if day.empty:
                grids.append(np.full(grid_lon.shape, np.nan))
                continue
            xc, yc, zc = _cell_mean(day["경도"].values, day["위도"].values,
                                    day["강수량"].values, lon_centers, lat_centers)
            if zc.size == 0:
                grids.append(np.full(grid_lon.shape, np.nan))
                continue
            g = idw_grid(xc, yc, zc, grid_lon, grid_lat)
            g[g < 0] = 0.0
            g[~land_mask] = np.nan
            grids.append(g)

        cube = np.stack(grids, axis=2)          # (y, x, time)
        ds_out = xr.Dataset(
            {"precipitation": (["y", "x", "time"], cube,
                               {"units": "mm/day",
                                "long_name": f"IDW interpolated precipitation ({name})"})},
            coords={"time": (["time"], dates),
                    "lat": (["y", "x"], grid_lat,
                            {"units": "degrees_north", "standard_name": "latitude"}),
                    "lon": (["y", "x"], grid_lon,
                            {"units": "degrees_east", "standard_name": "longitude"})},
            attrs={"description": f"IDW precipitation for South Korea ({name}, {yr})",
                   "idw_power": POWER, "method": "cell-mean + inverse distance weighting"},
        )
        path = os.path.join(OUT_DIR, f"Precipitation_IDW_{name}_{yr}.nc")
        ds_out.to_netcdf(path)
        print(f"  저장: {path}  {cube.shape}")


# ==============================================================================
# 메인
# ==============================================================================
if __name__ == "__main__":
    grid_lon, grid_lat = np.meshgrid(TGT_LON, TGT_LAT)   # (49, 49)
    print(f"목표격자 {grid_lon.shape} (lat {TGT_LAT[0]}~{TGT_LAT[-1]}, "
          f"lon {TGT_LON[0]}~{TGT_LON[-1]}, 0.125°)")
    land_mask = build_land_mask(grid_lon, grid_lat)

    for name in TARGETS:
        print("=" * 60)
        print(f"{name} IDW 보간")
        print("=" * 60)
        interpolate_network(name, NETWORKS[name], grid_lon, grid_lat, land_mask)

    print("\n완료. output/Precipitation_IDW_{AWS|ASOS}_{YYYY}.nc")
