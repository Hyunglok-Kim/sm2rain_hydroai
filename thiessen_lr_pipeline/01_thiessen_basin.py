#!/usr/bin/env python3
"""01 — 지점 시자료로 티센 다각형 유역 일강수를 만든다.

이 값이 뒤 단계(02·03)의 **회귀 목표**이자 04 의 **평가 기준**이다.

산정 방법
    지점 가중치   유역을 250 m 격자로 잘게 나눠 각 칸을 최근접 관측지점에
                  배정하고 그 면적 비율을 지점 가중치로 삼는다 (EPSG:5179).
                  폴리곤 교차를 직접 푸는 것과 결과가 같고 훨씬 간단하다.
    유역 일강수   P_b(t) = Σ w_i·P_i(t) / Σ w_i
                  그날 값이 있는 지점만 더하고 가중치를 다시 정규화한다.
                  값이 있는 지점이 하나도 없으면 그날은 결측이다.

입력
    data/AWS_hourly_YYYY.csv   기상청 시자료.  열 이름은 일시·지점·위도·경도·강수량
                               (data_pipeline/AWS/AWS_download.py 산출 형식)
    data/basin.shp .dbf .shx   국가 표준유역도

출력
    output/THIESSEN_basin_daily.csv    행=날짜, 열=유역코드 (mm/day)
    output/THIESSEN_basin_daily.nc     같은 값 (time × basin)
    output/THIESSEN_basin_weights.csv  유역코드 · 지점번호 · 가중치

주의
    하루는 KST 01시 ~ 익일 00시다.  연 파일 경계일(12-31)은 두 파일에
    나뉘어 있으므로 같은 날짜끼리 합친다.

실행
    python3 01_thiessen_basin.py
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

import common as C

YEARS = range(2021, 2026)
START, END = '2021-01-01', '2025-05-01'
CELL = 250.0        # 가중치 산정용 세부 격자 (m)
NEAR_N = 80         # 유역 중심에서 가까운 지점만 후보로 둔다 (속도)


def station_daily():
    """지점별 일강수와 좌표.  한 번 만들어 두고 캐시에서 읽는다."""
    cache = C.OUT / 'station_daily.pkl'
    if cache.exists():
        return pickle.load(open(cache, 'rb'))

    frames, meta = [], []
    for y in YEARS:
        f = C.AWS_HOURLY % y
        C.need(f)
        parts = list(pd.read_csv(f, usecols=['일시', '지점', '위도', '경도', '강수량'],
                                 chunksize=3_000_000))
        d = pd.concat(parts, ignore_index=True)
        #  KST 01시~익일 00시가 하루다.  1시간을 당겨 날짜로 묶는다.
        d['날짜'] = (pd.to_datetime(d['일시'])
                   - pd.Timedelta(hours=1)).dt.normalize()
        frames.append(d.groupby(['날짜', '지점'])['강수량']
                      .sum(min_count=1).unstack())
        meta.append(d.drop_duplicates('지점').set_index('지점')[['위도', '경도']])
        print(f'  {y} 읽음', flush=True)

    S = (pd.concat(frames).groupby(level=0).sum(min_count=1)
         .sort_index().loc[START:END])
    M = pd.concat(meta).groupby(level=0).first()
    C.OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump((S, M), open(cache, 'wb'))
    return S, M


def weights(geom, sx, sy, codes):
    """유역 하나의 티센 가중치 {지점번호: 비율} 와 유효면적 (km2)."""
    x0, y0, x1, y1 = geom.bounds
    cx, cy = C.to5179(np.array([(x0 + x1) / 2]), np.array([(y0 + y1) / 2]))
    cand = np.argsort((sx - cx[0]) ** 2 + (sy - cy[0]) ** 2)[:NEAR_N]

    #  섬을 낀 유역은 MultiPolygon 이므로 조각을 모두 훑는다
    parts = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    shells = []
    for g in parts:
        r = np.asarray(g.exterior.coords)
        rx, ry = C.to5179(r[:, 0], r[:, 1])
        holes = []
        for h in g.interiors:
            hh = np.asarray(h.coords)
            hx, hy = C.to5179(hh[:, 0], hh[:, 1])
            holes.append(np.column_stack([hx, hy]))
        shells.append((np.column_stack([rx, ry]), holes))

    allxy = np.vstack([s for s, _ in shells])
    gx = np.arange(allxy[:, 0].min(), allxy[:, 0].max() + CELL, CELL)
    gy = np.arange(allxy[:, 1].min(), allxy[:, 1].max() + CELL, CELL)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])

    ins = np.zeros(len(pts), bool)
    for shell, holes in shells:
        m = MplPath(shell).contains_points(pts)
        for h in holes:
            m &= ~MplPath(h).contains_points(pts)
        ins |= m
    if not ins.any():
        return {}, 0.0

    px, py = pts[ins, 0], pts[ins, 1]
    d2 = ((px[:, None] - sx[cand][None, :]) ** 2
          + (py[:, None] - sy[cand][None, :]) ** 2)
    cnt = np.bincount(np.argmin(d2, axis=1), minlength=len(cand))
    tot = cnt.sum()
    w = {int(codes[cand[i]]): c / tot for i, c in enumerate(cnt) if c}
    return w, tot * CELL ** 2 / 1e6


def main() -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)

    print('1. 지점 일강수')
    S, M = station_daily()
    codes = M.index.to_numpy()
    sx, sy = C.to5179(M['경도'].to_numpy(float), M['위도'].to_numpy(float))
    print(f'   {len(S)}일 × {S.shape[1]}지점')

    print('2. 유역 목록')
    B = C.basins()
    order = sorted(B['W'])
    print(f'   {len(order)}개')

    print('3. 티센 가중치')
    t0 = time.time()
    W, rows = {}, []
    recs = C.read_dbf(C.BASIN_SHP + '.dbf')
    offs = C.shp_offsets(C.BASIN_SHP)
    key = next(k for k in recs[0] if 'CD' in k.upper())
    idx = {r[key].strip(): i for i, r in enumerate(recs)}
    for n, code in enumerate(order, 1):
        if code not in idx:
            W[code] = {}
            continue
        try:
            g = C.read_polygon(C.BASIN_SHP, *offs[idx[code]])
        except Exception:
            W[code] = {}
            continue
        w, _ = weights(g, sx, sy, codes)
        W[code] = w
        for st, v in sorted(w.items(), key=lambda x: -x[1]):
            rows.append({'유역코드': code, '지점번호': st, '가중치': round(v, 6)})
        if n % 200 == 0:
            print(f'   {n}/{len(order)}   {time.time() - t0:.0f}s', flush=True)
    nst = pd.Series({k: len(v) for k, v in W.items()})
    print(f'   완료 {time.time() - t0:.0f}s   기여지점 중앙값 {nst.median():.0f}개 '
          f'(최소 {nst.min()} 최대 {nst.max()})')

    print('4. 유역 일강수')
    V = S.to_numpy(float)
    have = np.isfinite(V)
    col = {c: i for i, c in enumerate(S.columns)}
    out = np.full((len(S), len(order)), np.nan)
    for j, code in enumerate(order):
        w = W.get(code) or {}
        keys = [k for k in w if k in col]
        if not keys:
            continue
        ii = [col[k] for k in keys]
        ww = np.array([w[k] for k in keys])
        num = np.nansum(np.where(have[:, ii], V[:, ii], 0.0) * ww, axis=1)
        den = (have[:, ii] * ww).sum(axis=1)
        out[:, j] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    T = pd.DataFrame(out, index=S.index, columns=order)
    T.index.name = 'date'

    print('5. 저장')
    p = C.OUT / 'THIESSEN_basin_daily.csv'
    T.to_csv(p, float_format='%.2f')
    print('  ', p)
    pw = C.OUT / 'THIESSEN_basin_weights.csv'
    pd.DataFrame(rows).to_csv(pw, index=False, encoding='utf-8-sig')
    print('  ', pw)

    import xarray as xr
    ds = xr.Dataset(
        {'precipitation': (('time', 'basin'), T.to_numpy(np.float32))},
        coords={'time': T.index.values, 'basin': np.array(order, dtype='U8')},
        attrs={'title': 'Thiessen polygon basin daily precipitation',
               'method': '250 m nearest-station area weights, renormalised',
               'stations': f'{S.shape[1]} gauges (KMA hourly)',
               'day_boundary': 'KST 01h - next 00h', 'units': 'mm/day'})
    ds['precipitation'].attrs['units'] = 'mm/day'
    pn = C.OUT / 'THIESSEN_basin_daily.nc'
    ds.to_netcdf(pn)
    print('  ', pn)

    ok = T.notna().sum()
    print(f'\n유효일수 중앙값 {ok.median():.0f}일 / {len(T)}일')


if __name__ == '__main__':
    main()
