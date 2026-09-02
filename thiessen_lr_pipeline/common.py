#!/usr/bin/env python3
"""공통 유틸 — 경로 설정, 표준유역도 읽기, 유역×격자 가중치, 평가지표.

이 폴더의 스크립트가 모두 이 파일을 import 한다.  네 스크립트가 같은
표준유역도 파서와 면적가중 로직을 쓰기 때문에, 파일마다 200줄씩 복사하는
대신 한곳에 모았다.

경로
    입력은 `data/`, 출력은 `output/` 에 둔다.  둘 다 스크립트 위치 기준
    상대경로이고, 다른 곳을 쓰려면 환경변수로 덮어쓴다.

        export TLR_DATA=/my/inputs
        export TLR_OUT=/my/outputs

표준유역도
    shapefile 을 pyproj·geopandas 없이 직접 읽는다.  설치 부담을 줄이려는
    것이고, 폴리곤(5)·멀티폴리곤(15) 두 형식만 다룬다.  파일 이름은
    BASIN_SHP 로 바꿀 수 있다 (기본 data/basin, 확장자 없는 stem).

        export TLR_BASIN_SHP=/my/inputs/std_basin_850
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# ────────────────────────────────────────────────────────────── 경로
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('TLR_DATA', HERE / 'data'))
OUT = Path(os.environ.get('TLR_OUT', HERE / 'output'))
BASIN_SHP = str(os.environ.get('TLR_BASIN_SHP', DATA / 'basin'))   # 확장자 없는 stem

#  입력 파일 이름 (data/ 안에서 찾는다)
F_MERGED = DATA / 'ds_merged_LR.nc'      # SM2RAIN·ERA5·GPM·TCA·AWS·ASOS 조립본
F_BC12 = DATA / 'BC12_fields.nc'         # BC_1·BC_2 (03_BC_LightGBM.py 산출)
AWS_HOURLY = str(DATA / 'AWS_hourly_%d.csv')   # 지점 시자료 (연도별)

#  기간
FIT_YEAR = '2021'                        # 회귀 적합 연도
EVAL0, EVAL1 = '2022-01-01', '2025-05-01'   # 평가 구간

#  격자별 회귀(02, LR-G)의 설명변수.  LightGBM 편의보정(BC-G)과 같은 입력
#  구성이다 — 위성·재분석 네 가지에 같은 날 지상관측 격자장(AWS)을 더한다.
#  목표만 IDW_AWS 에서 티센으로 바뀐다.
X_GRID = ['SM2RAIN', 'ERA5', 'GPM', 'TCA', 'AWS']

#  유역별 회귀(03)의 설명변수.  이미 편의보정된 BC_G 를 티센에 다시 맞춘다
X_BASIN = ['BC_G']

#  BC12_fields.nc 의 변수명 → 이 파이프라인에서 쓰는 이름
BC_MAP = {'BC_G': 'BC_2', 'BC': 'BC_1'}

MIN_AREA_FRAC = 0.5      # 유역 유효면적이 이보다 작은 날은 결측으로 둔다
MIN_FIT = 7              # 계수를 풀 수 있는 최소 표본 (설명변수 5 + 절편 + 1)


def need(*paths) -> None:
    """입력이 있는지 먼저 확인하고, 없으면 무엇을 어디에 두라고 알려준다."""
    missing = [p for p in paths if not Path(str(p)).exists()]
    if missing:
        lines = '\n'.join(f'    {p}' for p in missing)
        raise SystemExit(
            f'다음 입력이 없습니다.\n{lines}\n\n'
            f'  · 입력 폴더는 {DATA} 입니다 (환경변수 TLR_DATA 로 바꿉니다).\n'
            f'  · 표준유역도는 stem 이 {BASIN_SHP} 인 .shp/.dbf/.shx 입니다\n'
            f'    (환경변수 TLR_BASIN_SHP 로 바꿉니다).\n'
            f'  · README.md 의 "데이터 배치" 를 참고하세요.')


def read_dbf(path: str) -> list[dict]:
    """dbf 속성표. std_basin_850 은 cpg 가 UTF-8 이다."""
    with open(path, 'rb') as f:
        nrec, hlen, rlen = struct.unpack('<IHH', f.read(32)[4:12])
        flds = []
        for _ in range((hlen - 33) // 32):
            d = f.read(32)
            flds.append((d[:11].split(b'\x00')[0].decode('latin1'), d[16]))
        f.seek(hlen)
        out = []
        for _ in range(nrec):
            rec = f.read(rlen)
            off, row = 1, {}
            for nm, ln in flds:
                row[nm] = rec[off:off + ln].decode('utf-8', 'replace').strip()
                off += ln
            out.append(row)
    return out


def shp_offsets(path: str) -> list[tuple[int, int]]:
    """shx 로 레코드별 (바이트오프셋, 길이)."""
    with open(path + '.shx', 'rb') as f:
        f.seek(24)
        flen = struct.unpack('>I', f.read(4))[0] * 2
        f.seek(100)
        out = []
        for _ in range((flen - 100) // 8):
            o, l = struct.unpack('>II', f.read(8))
            out.append((o * 2, l * 2))
    return out


def read_polygon(path: str, offset: int, length: int):
    """폴리곤 레코드 → shapely 도형.

    shapefile 규약상 껍질은 시계방향(부호면적 음수), 구멍은 반시계방향이다.
    이 부호로 둘을 가른 뒤 구멍을 제 껍질에 붙인다.
    """
    with open(path + '.shp', 'rb') as f:
        f.seek(offset + 8)
        buf = f.read(length)
    stype = struct.unpack('<i', buf[:4])[0]
    if stype != 5:
        raise ValueError(f'폴리곤이 아닙니다 (shape type {stype})')
    nparts, npts = struct.unpack('<ii', buf[36:44])
    parts = struct.unpack(f'<{nparts}i', buf[44:44 + 4 * nparts])
    p0 = 44 + 4 * nparts
    xy = np.frombuffer(buf[p0:p0 + 16 * npts], dtype='<f8').reshape(npts, 2)
    idx = list(parts) + [npts]

    shells, holes = [], []
    for i in range(nparts):
        ring = xy[idx[i]:idx[i + 1]]
        if len(ring) < 4:
            continue
        a = 0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
        (holes if a > 0 else shells).append(ring)

    polys = []
    for s in shells:
        sp = Polygon(s)
        if not sp.is_valid:
            sp = sp.buffer(0)
        mine = [h for h in holes
                if sp.contains(Polygon(h).representative_point())]
        p = Polygon(s, mine) if mine else sp
        polys.append(p if p.is_valid else p.buffer(0))
    return unary_union(polys) if len(polys) > 1 else polys[0]


# ------------------------------------------------------------------ 가중치


def cell_weights(geom, lat, lon):
    """격자셀 × 유역 교차면적 → [(ilat, ilon, km2), ...]"""
    dlat = float(abs(lat[1] - lat[0]))
    dlon = float(abs(lon[1] - lon[0]))
    minx, miny, maxx, maxy = geom.bounds
    out = []
    for i, la in enumerate(lat):
        if la + dlat / 2 < miny or la - dlat / 2 > maxy:
            continue
        for j, lo in enumerate(lon):
            if lo + dlon / 2 < minx or lo - dlon / 2 > maxx:
                continue
            inter = geom.intersection(
                box(lo - dlon / 2, la - dlat / 2, lo + dlon / 2, la + dlat / 2))
            if inter.is_empty:
                continue
            km2 = inter.area * (111.32 ** 2) * np.cos(np.deg2rad(la))
            if km2 > 1e-6:
                out.append((i, j, float(km2)))
    return out


# ------------------------------------------------------------------ 지표


def to5179(lon, lat):
    """WGS84 → EPSG:5179 (한국 중부원점 TM). pyproj 없이 직접 계산."""
    a, f = 6378137.0, 1 / 298.257222101
    e2 = f * (2 - f)
    k0, lat0, lon0 = 0.9996, np.deg2rad(38.0), np.deg2rad(127.5)
    FE, FN = 1000000.0, 2000000.0
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    e_2 = e2 / (1 - e2)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2
    C = e_2 * np.cos(lat) ** 2
    A = (lon - lon0) * np.cos(lat)

    def M(p):
        return a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * p
                    - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024)
                    * np.sin(2 * p)
                    + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * p)
                    - (35 * e2**3 / 3072) * np.sin(6 * p))

    x = FE + k0 * N * (A + (1 - T + C) * A**3 / 6
                       + (5 - 18 * T + T**2 + 72 * C - 58 * e_2) * A**5 / 120)
    y = FN + k0 * (M(lat) - M(lat0)
                   + N * np.tan(lat) * (A**2 / 2
                                        + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
                                        + (61 - 58 * T + T**2 + 600 * C
                                           - 330 * e_2) * A**6 / 720))
    return x, y


# ────────────────────────────────────────────────────────────── 유역 목록
def basins(shp: str | None = None) -> dict:
    """표준유역도에서 유역별 (격자 교차면적, 이름, 면적, 중심).

    격자 교차면적은 `ds_merged_LR.nc` 의 lat/lon 을 기준으로 잰다.
    한 번 계산해 output/basin_cell_weights.npz 에 두고 다시 쓴다.
    """
    import pickle
    shp = shp or BASIN_SHP
    cache = OUT / 'basin_cell_weights.pkl'
    if cache.exists():
        return pickle.load(open(cache, 'rb'))

    import xarray as xr
    need(F_MERGED, shp + '.shp', shp + '.dbf', shp + '.shx')
    ds = xr.open_dataset(F_MERGED)
    lat, lon = ds.lat.values, ds.lon.values
    ds.close()

    print('유역×격자 교차면적 계산 중...', end=' ', flush=True)
    recs = read_dbf(shp + '.dbf')
    offs = shp_offsets(shp)
    key = next(k for k in recs[0] if 'SBSN_CD' in k.upper()) if any(
        'SBSN_CD' in k.upper() for k in recs[0]) else next(
        k for k in recs[0] if 'CD' in k.upper())
    nm = next((k for k in recs[0] if 'NM' in k.upper()), key)

    W = {}
    for k, r in enumerate(recs):
        try:
            geom = read_polygon(shp, *offs[k])
        except Exception:
            continue
        w = cell_weights(geom, lat, lon)
        if not w:
            continue
        W[r[key].strip()] = {
            'w': w, 'name': r[nm].strip(),
            'cx': float(geom.centroid.x), 'cy': float(geom.centroid.y),
            'area': float(geom.area * 111.32 ** 2
                          * np.cos(np.deg2rad(geom.centroid.y)))}
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump({'W': W, 'lat': lat, 'lon': lon}, open(cache, 'wb'))
    print(f'유역 {len(W)}개  →  {cache}')
    return {'W': W, 'lat': lat, 'lon': lon}


def to_basin(W: dict, arr: np.ndarray, time) -> pd.DataFrame:
    """격자장 → 유역 면적가중 일강수.

    유효 격자 면적 합이 유역 면적의 MIN_AREA_FRAC 미만인 날은 결측이다.
    0.1° 한 칸이 약 100 km2 이고 표준유역 중앙값이 113 km2 라 유역이 격자
    여러 개에 걸치므로, 중심 격자값만 쓰거나 산술평균하면 유역 밖 강수가
    섞인다.
    """
    out = {}
    for code, d in W.items():
        tot = sum(a for _, _, a in d['w'])
        num = np.zeros(len(time))
        den = np.zeros(len(time))
        for i, j, a in d['w']:
            x = arr[:, i, j]
            ok = np.isfinite(x)
            num[ok] += a * x[ok]
            den[ok] += a
        out[code] = np.where(den >= MIN_AREA_FRAC * tot,
                             num / np.where(den > 0, den, 1), np.nan)
    return pd.DataFrame(out, index=time)


# ────────────────────────────────────────────────────────────── 지표
def scores(sim, obs) -> dict:
    """KGE·R·RMSE·편의·누적비.  값이 60일 미만이면 비운다."""
    sim, obs = np.asarray(sim, float), np.asarray(obs, float)
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(s) < 60 or s.std() == 0 or o.std() == 0 or o.mean() == 0:
        return {}
    r = float(np.corrcoef(s, o)[0, 1])
    a, b = float(s.std() / o.std()), float(s.mean() / o.mean())
    return {'n': len(s), 'KGE': 1 - float(np.sqrt((r - 1) ** 2 + (a - 1) ** 2
                                                  + (b - 1) ** 2)),
            'R': r, 'RMSE': float(np.sqrt(((s - o) ** 2).mean())),
            '편의': float(s.mean() - o.mean()),
            '누적': float(s.sum()), '기준누적': float(o.sum()),
            '누적비': b, '최대일': float(s.max()), '기준최대일': float(o.max())}


def fit_ols(X: np.ndarray, y: np.ndarray, train: np.ndarray):
    """train 구간으로 절편 포함 최소제곱 적합 → 전 구간 예측 (음수는 0).

    반환 (예측, 계수, 절편, 적합에 쓴 날 수).  적합할 날이 MIN_FIT 보다
    적으면 예측은 전부 결측이다.
    """
    ok = np.isfinite(X).all(1)
    m = ok & np.isfinite(y) & train
    if m.sum() < MIN_FIT:
        return np.full(len(y), np.nan), None, np.nan, int(m.sum())
    A = np.column_stack([X[m], np.ones(m.sum())])
    c, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    p = np.full(len(y), np.nan)
    p[ok] = np.maximum(np.column_stack([X[ok], np.ones(ok.sum())]) @ c, 0.0)
    return p, c[:-1], float(c[-1]), int(m.sum())


# ────────────────────────────────────────────────────────────── 평가·그림
def compare(THI: pd.DataFrame, S: dict, prods: list, W: dict, order: list):
    """유역별 지표 · 전국 면적가중 평균 · 강우강도 구간별 재현비.

    모든 산출물이 다 있는 날로 표본을 맞춘다.  결측일이 산출물마다 다르면
    서로 다른 날짜를 보고 비교하게 되기 때문이다.
    """
    rows = []
    for c in order:
        o = THI[c].loc[EVAL0:EVAL1]
        v = {k: S[k][c].loc[EVAL0:EVAL1] for k in prods}
        m = o.notna()
        for x in v.values():
            m &= x.notna()
        if m.sum() < 60:
            continue
        o = o[m]
        yp = [gg.idxmax() for _, gg in o.groupby(o.index.year)
              if gg.notna().sum() >= 30 and gg.max() > 0]
        for k in prods:
            sc = scores(v[k][m].to_numpy(), o.to_numpy())
            if not sc:
                continue
            sc['연최대일비'] = float(np.median(
                [v[k].loc[d] / o.loc[d] for d in yp])) if yp else np.nan
            rows.append({'유역코드': c, '유역명': W[c]['name'], '산출': k, **sc})
    tab = pd.DataFrame(rows)

    #  전국 면적가중 평균
    area = np.array([W[c]['area'] for c in order], float)
    sl = slice(EVAL0, EVAL1)
    OK = np.isfinite(THI.loc[sl].to_numpy())
    for k in prods:
        OK &= np.isfinite(S[k][order].loc[sl].to_numpy())
    idx = THI.loc[sl].index
    den = (OK * area).sum(1)
    good = den > 0

    def nat(df):
        x = np.where(OK, np.nan_to_num(df[order].loc[sl].to_numpy()), 0)
        return pd.Series(np.where(good, (x * area).sum(1)
                                  / np.where(good, den, 1), np.nan), index=idx)

    NAT = pd.DataFrame({'THIESSEN': nat(THI), **{k: nat(S[k]) for k in prods}})
    NAT.index.name = 'date'
    o = NAT['THIESSEN'].dropna()
    nrow = []
    for k in prods:
        sc = scores(NAT[k].reindex(o.index).to_numpy(), o.to_numpy())
        nrow.append({'산출': k, **sc, '최대일비': sc['최대일'] / sc['기준최대일']})
    nstat = pd.DataFrame(nrow).set_index('산출')[
        ['n', 'KGE', 'R', 'RMSE', '누적비', '최대일', '기준최대일', '최대일비']]

    #  강우강도 구간별 재현비
    edges = [-.01, .5, 2, 5, 10, 20, 1e4]
    names = ['~0.5', '0.5~2', '2~5', '5~10', '10~20', '20~']
    band = pd.cut(o, edges, labels=names)
    inten = pd.DataFrame(
        {k: [NAT[k].reindex(o.index)[band == b].sum() / o[band == b].sum()
             for b in names] for k in prods}, index=names).T
    inten.columns = [f'{c} ({int((band == c).sum())}일)' for c in names]
    return tab, NAT, nstat, band, inten


def style():
    """한글 그림 설정.  matplotlib 를 import 해 돌려준다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ('AppleGothic', 'NanumGothic', 'Malgun Gothic'):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = f
            break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 12})
    return plt


def savefig(plt, fig, stem):
    p = OUT / f'{stem}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', p)


def fig_distribution(plt, tab, prods, LAB, COL, title, stem):
    """유역별 지표 분포."""
    import numpy as np
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (m, lo, hi, ref) in zip(
            axs, [('KGE', -.2, 1, None), ('R', .6, 1, None),
                  ('누적비', .6, 1.6, 1.0), ('연최대일비', 0, 1.6, 1.0)]):
        p = tab.pivot(index='유역코드', columns='산출', values=m)
        ax.boxplot([p[k].dropna() for k in prods], showfliers=False, widths=.6,
                   patch_artist=True, boxprops=dict(facecolor='#EEF1F4'),
                   medianprops=dict(color='k', lw=2))
        for i, k in enumerate(prods, 1):
            v = p[k].dropna()
            ax.scatter(np.random.normal(i, .07, len(v)), v, s=5, alpha=.15,
                       color=COL[k], zorder=0)
        if ref is not None:
            ax.axhline(ref, color='#C0392B', lw=1.4, ls='--')
        ax.set_xticks(range(1, len(prods) + 1))
        ax.set_xticklabels(prods, rotation=25, ha='right', fontsize=10)
        ax.set_ylim(lo, hi)
        ax.set_title(m)
        ax.grid(axis='y', alpha=.3)
    fig.suptitle(f'{title} — 유역 {tab["유역코드"].nunique()}개 분포   '
                 f'티센 기준   {EVAL0} ~ {EVAL1}', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .93))
    savefig(plt, fig, f'{stem}_basin_distribution')


def fig_national(plt, NAT, nstat, prods, LAB, COL, title, stem):
    """전국 면적가중 월평균과 누적강수."""
    import matplotlib.dates as mdates
    o = NAT['THIESSEN'].dropna()
    loc = mdates.AutoDateLocator(minticks=8, maxticks=16)

    mo = NAT.resample('MS').mean()
    fig, ax = plt.subplots(figsize=(16, 4.4))
    ax.fill_between(mo.index, 0, mo['THIESSEN'], color='#D7DCE1', zorder=0)
    ax.plot(mo.index, mo['THIESSEN'], color='k', lw=2.2, label='티센')
    for k in prods:
        ax.plot(mo.index, mo[k], color=COL[k], lw=1.5, label=LAB[k])
    ax.set_ylabel('월평균 일강수 [mm/일]')
    ax.set_ylim(bottom=0)
    ax.margins(x=.01)
    ax.grid(alpha=.25)
    ax.legend(ncols=3, fontsize=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_title(f'{title} — 전국 면적가중 월평균 일강수', fontweight='bold')
    fig.tight_layout()
    savefig(plt, fig, f'{stem}_national_monthly')

    fig, ax = plt.subplots(figsize=(16, 4.4))
    ax.fill_between(o.index, 0, o.cumsum(), color='#D7DCE1', zorder=0)
    ax.plot(o.index, o.cumsum(), color='k', lw=2.4, label='티센')
    for k in prods:
        v = NAT[k].reindex(o.index).cumsum()
        ax.plot(v.index, v, color=COL[k], lw=1.8, label=LAB[k])
        ax.annotate(f'{nstat.loc[k, "누적비"]:.2f}', (v.index[-1], v.iloc[-1]),
                    xytext=(6, 0), textcoords='offset points', va='center',
                    color=COL[k], fontsize=10, fontweight='bold')
    ax.set_ylabel('누적강수 [mm]')
    ax.margins(x=.03)
    ax.grid(alpha=.25)
    ax.legend(loc='upper left', fontsize=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_title(f'{title} — 전국 면적가중 누적강수', fontweight='bold')
    fig.tight_layout()
    savefig(plt, fig, f'{stem}_national_cumulative')
