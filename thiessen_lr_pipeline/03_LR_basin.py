#!/usr/bin/env python3
"""03 — BC_G 를 유역평균한 뒤 티센에 다시 맞춘다.  [산출물]

02 는 위성·재분석 원자료로 격자마다 회귀를 적합해 격자장을 낸다.  이 스크립트는
이미 편의보정이 끝난 **BC_G 를 출발점**으로 삼는다.  BC_G 를 유역 면적가중
평균한 뒤, 유역마다 티센을 목표로 회귀 하나를 적합한다.

    설명변수  BC_G 를 유역 면적가중 평균한 값   (common.X_BASIN)
    목표      그 유역의 티센 일강수
    적합      2021년
    적용      전 기간

설명변수가 하나뿐인 1차 변환이라 상관계수는 BC_G 와 같고, 바뀌는 것은 크기와
편의다.  BC_G 가 총량은 잘 맞추면서 극값을 놓치는 성질을 티센 쪽으로 되돌리는
것이 목적이고, 실제로 잡히는지는 05 에서 본다.

입력
    data/BC12_fields.nc             BC_2 (= BC_G).  03_BC_LightGBM.py 산출
    output/THIESSEN_basin_daily.nc  01 산출 (회귀 목표)

출력
    output/LR_basin_daily.csv   행=날짜, 열=유역코드 (mm/day)
    output/LR_basin_daily.nc    같은 값 (time × basin)
    output/LR_basin_coef.csv    유역별 회귀계수 · 절편 · 적합일수

실행
    python3 03_LR_basin.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import common as C


def main() -> None:
    f_thi = C.OUT / 'THIESSEN_basin_daily.nc'
    C.need(C.F_MERGED, C.F_BC12, f_thi)

    ds = xr.open_dataset(C.F_MERGED)
    t = pd.to_datetime(ds.time.values)
    ds.close()

    d2 = xr.open_dataset(C.F_BC12)
    A = {}
    for name in C.X_BASIN:
        src = C.BC_MAP.get(name, name)
        if src not in d2.data_vars:
            raise SystemExit(f'{C.F_BC12} 에 {src} 변수가 없습니다.')
        A[name] = d2[src].values
    d2.close()

    th = xr.open_dataset(f_thi)
    THI = pd.DataFrame(th['precipitation'].values,
                       index=pd.to_datetime(th.time.values),
                       columns=[str(c) for c in th.basin.values]).reindex(t)
    th.close()

    B = C.basins()
    W = B['W']
    print(f'유역 {len(W)}개 · {len(t)}일   설명변수 {C.X_BASIN}')

    #  설명변수를 유역 면적가중 평균으로
    XB = {v: C.to_basin(W, A[v], t) for v in C.X_BASIN}
    order = sorted(W)
    train = np.asarray(t.year == int(C.FIT_YEAR))

    pred = np.full((len(t), len(order)), np.nan)
    rows = []
    for j, code in enumerate(order):
        if code not in THI.columns:
            continue
        X = np.column_stack([XB[v][code].to_numpy() for v in C.X_BASIN])
        p, c, b0, n = C.fit_ols(X, THI[code].to_numpy(), train)
        pred[:, j] = p
        rows.append({'유역코드': code, '유역명': W[code]['name'],
                     '면적_km2': round(W[code]['area'], 1),
                     **({v: round(float(cv), 4)
                         for v, cv in zip(C.X_BASIN, c)} if c is not None else {}),
                     '절편': round(b0, 4) if c is not None else np.nan,
                     '적합일수': n})
    L = pd.DataFrame(pred, index=t, columns=order)
    L.index.name = 'date'
    coef = pd.DataFrame(rows)

    C.OUT.mkdir(parents=True, exist_ok=True)
    p = C.OUT / 'LR_basin_daily.csv'
    L.to_csv(p, float_format='%.2f')
    print('저장', p)
    pc = C.OUT / 'LR_basin_coef.csv'
    coef.to_csv(pc, index=False, encoding='utf-8-sig')
    print('저장', pc)

    out = xr.Dataset(
        {'LR_basin': (('time', 'basin'), L.to_numpy(np.float32))},
        coords={'time': t.values, 'basin': np.array(order, dtype='U8')},
        attrs={'title': 'Basin daily precipitation — BC_G rescaled to Thiessen',
               'predictors': ', '.join(C.X_BASIN) + ' (basin area-weighted)',
               'target': 'Thiessen basin daily precipitation',
               'method': f'per-basin OLS, fit {C.FIT_YEAR}, applied to all days',
               'fit': C.FIT_YEAR, 'units': 'mm/day',
               'time_zone': 'KST (UTC+9)'})
    out['LR_basin'].attrs['units'] = 'mm/day'
    pn = C.OUT / 'LR_basin_daily.nc'
    out.to_netcdf(pn)
    print('저장', pn)

    if len(C.X_BASIN) == 1 and C.X_BASIN[0] in coef.columns:
        s = coef[C.X_BASIN[0]].dropna()
        print(f'{C.X_BASIN[0]} 계수  중앙값 {s.median():+.3f}  '
              f'({s.quantile(.1):+.3f} ~ {s.quantile(.9):+.3f})')
    ok = L.notna().sum()
    print(f'유효일수 중앙값 {ok.median():.0f}일 / {len(L)}일')


if __name__ == '__main__':
    main()
