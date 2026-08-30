#!/usr/bin/env python3
"""02 — 격자별 선형회귀로 전국 강우 격자장을 만든다.  [산출물]

편향보정의 목표자료를 IDW_AWS 대신 **티센 유역평균**으로 두면 어떻게 되는지
보는 판이다.  격자 한 칸마다 선형회귀를 따로 적합하고, 그 칸의 목표는 칸이
속한 표준유역의 티센 일강수다.

    설명변수  SM2RAIN, ERA5, GPM, TCA   (LightGBM 편의보정과 같은 입력,
                                         지상관측은 넣지 않는다)
    목표      격자가 속한 표준유역의 티센 일강수
    적합      2021년
    적용      전 기간

격자의 소속 유역은 겹치는 면적이 가장 큰 유역으로 정한다.  0.1° 한 칸이 약
100 km2 이고 표준유역 중앙값이 113 km2 라 한 칸이 여러 유역에 걸치지만,
회귀가 성립하려면 목표를 하나로 정해야 한다.

입력
    data/ds_merged_LR.nc            SM2RAIN·ERA5·GPM·TCA 격자
    output/THIESSEN_basin_daily.nc  01 산출 (회귀 목표)
    data/basin.shp .dbf .shx        국가 표준유역도

출력
    output/LR_grid.nc   LR (time × lat × lon) · coef (term × lat × lon) · n_fit

실행
    python3 02_LR_grid.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xarray as xr

import common as C


def dominant_basin(W, lat, lon):
    """격자마다 겹치는 면적이 가장 큰 유역."""
    best = np.zeros((len(lat), len(lon)))
    owner = np.full((len(lat), len(lon)), '', dtype=object)
    for code, d in W.items():
        for i, j, a in d['w']:
            if a > best[i, j]:
                best[i, j], owner[i, j] = a, code
    return owner


def main() -> None:
    f_thi = C.OUT / 'THIESSEN_basin_daily.nc'
    C.need(C.F_MERGED, f_thi)

    ds = xr.open_dataset(C.F_MERGED)
    t = pd.to_datetime(ds.time.values)
    A = {v: ds[v].values for v in C.X_GRID}
    lat, lon = ds.lat.values, ds.lon.values
    ds.close()

    th = xr.open_dataset(f_thi)
    THI = pd.DataFrame(th['precipitation'].values,
                       index=pd.to_datetime(th.time.values),
                       columns=[str(c) for c in th.basin.values]).reindex(t)
    th.close()

    B = C.basins()
    owner = dominant_basin(B['W'], lat, lon)
    inside = int((owner != '').sum())
    print(f'격자 {len(lat)}×{len(lon)} · {len(t)}일 · 유역 {len(B["W"])}개')
    print(f'유역에 속한 격자 {inside}개')

    train = np.asarray(t.year == int(C.FIT_YEAR))
    P = np.full((len(t), len(lat), len(lon)), np.nan, np.float32)
    K = np.full((len(C.X_GRID) + 1, len(lat), len(lon)), np.nan, np.float32)
    N = np.zeros((len(lat), len(lon)), np.int32)

    t0, done = time.time(), 0
    for i in range(len(lat)):
        for j in range(len(lon)):
            code = owner[i, j]
            if not code or code not in THI.columns:
                continue
            X = np.column_stack([A[v][:, i, j] for v in C.X_GRID])
            p, c, b0, n = C.fit_ols(X, THI[code].to_numpy(), train)
            if c is None:
                continue
            P[:, i, j] = p
            K[:, i, j] = np.append(c, b0)
            N[i, j] = n
            done += 1
    print(f'적합한 격자 {done}개 / {inside}개   ({time.time() - t0:.0f}s)')

    C.OUT.mkdir(parents=True, exist_ok=True)
    out = xr.Dataset(
        {'LR': (('time', 'lat', 'lon'), P),
         'coef': (('term', 'lat', 'lon'), K),
         'n_fit': (('lat', 'lon'), N)},
        coords={'time': t, 'lat': lat, 'lon': lon,
                'term': C.X_GRID + ['intercept']},
        attrs={'title': 'Per-pixel linear regression, target = basin Thiessen',
               'predictors': ', '.join(C.X_GRID),
               'target': 'Thiessen basin daily precipitation',
               'method': f'per-pixel OLS, fit {C.FIT_YEAR}, applied to all days',
               'time_zone': 'KST (UTC+9)', 'units': 'mm/day'})
    out['LR'].attrs = {'units': 'mm/day', 'long_name':
                       'linear-regression precipitation (Thiessen target)'}
    out['coef'].attrs = {'long_name': 'regression coefficients and intercept'}
    out['n_fit'].attrs = {'long_name': 'days used to fit each pixel'}
    p = C.OUT / 'LR_grid.nc'
    out.to_netcdf(p)
    print('저장', p)

    v = P[np.isfinite(P)]
    print(f'값 범위 {v.min():.2f} ~ {v.max():.2f} mm/day   '
          f'유효 {np.isfinite(P).mean():.1%}')


if __name__ == '__main__':
    main()
