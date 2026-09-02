#!/usr/bin/env python3
"""04 — 격자 산출물끼리 견준다.  [분석용]

02 가 만든 격자 회귀 산출물(`LR_grid.nc`)을 기존 격자 산출물과 나란히 놓는다.
모두 같은 0.1° 격자이므로, 표준유역으로 면적가중 집계한 뒤 티센을 기준으로
평가한다.

    기준     티센 유역 일강수 (01 산출)
    평가구간 2022-01-01 ~ 2025-05-01   (적합에 쓴 2021년은 뺀다)
    대상     LR_grid   격자별 회귀(LR-G), 목표 티센 — 최종 산출물   (02)
             IDW_AWS   지상관측 보간 격자장 (조밀한 관측의 참조선)
             BC_G      LightGBM, 당일 지상관측 입력 (직전 단계 산출물)

유역 시계열 산출물(03)은 여기 넣지 않는다.  격자 자료가 아니고, 극한강우를
보는 관점이 달라 05 에서 따로 견준다.

읽을 때 주의
    LR_grid 는 목표와 평가기준이 같은 자료다.  티센 점수가 좋게 나오기 쉬운
    판이므로 "더 낫다"가 아니라 "제 목표를 얼마나 따라가는가"로 읽는다.
    실제로 볼 것은 유역별 편차와 관측밀도 의존성이다.

    유역 단위 평가와 전국 평균은 다른 것을 잰다.  유역마다 다른 날 오는
    국지 사상이 전국 평균에서는 희석되므로 섞어 쓰지 않는다.

입력
    output/LR_grid.nc               02
    output/THIESSEN_basin_daily.nc  01
    data/ds_merged_LR.nc            TCA
    data/BC12_fields.nc             BC_1 · BC_2

출력
    output/grid_basin_metrics.csv   유역 × 산출물 지표
    output/grid_summary.csv         산출물별 중앙값·평균
    output/grid_national_daily.csv  전국 면적가중 일강수 시계열
    output/grid_*.png               유역별 분포 · 전국 월평균 · 전국 누적

실행
    python3 04_evaluate_grid.py            그림까지
    python3 04_evaluate_grid.py --no-fig   표만
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr

import common as C

PRODS = ['LR_grid', 'IDW_AWS', 'BC_G']
LAB = {'LR_grid': 'LR-G (격자별 회귀, 목표 티센)', 'IDW_AWS': 'IDW_AWS (지상관측 보간)',
       'BC_G': 'BC-G (지상관측 융합)'}
COL = {'LR_grid': '#C0392B', 'IDW_AWS': '#2E86C1', 'BC_G': '#E08A2E'}


def main() -> None:
    f_thi = C.OUT / 'THIESSEN_basin_daily.nc'
    f_grid = C.OUT / 'LR_grid.nc'
    C.need(C.F_MERGED, C.F_BC12, f_thi, f_grid)

    ds = xr.open_dataset(C.F_MERGED)
    t = pd.to_datetime(ds.time.values)
    A = {'IDW_AWS': ds['AWS'].values}
    ds.close()
    g = xr.open_dataset(f_grid)
    A['LR_grid'] = g['LR'].values
    g.close()
    d2 = xr.open_dataset(C.F_BC12)
    for name, src in C.BC_MAP.items():
        if src in d2.data_vars:
            A[name] = d2[src].values
    d2.close()

    th = xr.open_dataset(f_thi)
    THI = pd.DataFrame(th['precipitation'].values,
                       index=pd.to_datetime(th.time.values),
                       columns=[str(c) for c in th.basin.values]).reindex(t)
    th.close()

    W = C.basins()['W']
    prods = [k for k in PRODS if k in A]
    print('격자 산출물:', ', '.join(prods))
    S = {k: C.to_basin(W, A[k], t) for k in prods}
    order = [c for c in sorted(W) if c in THI.columns]
    THI = THI[order]

    tab, NAT, nstat, band, inten = C.compare(THI, S, prods, W, order)
    C.OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(C.OUT / 'grid_basin_metrics.csv', index=False,
               encoding='utf-8-sig')
    cols = ['n', 'KGE', 'R', 'RMSE', '누적비', '연최대일비']
    pd.concat({'중앙값': tab.groupby('산출')[cols].median().reindex(prods),
               '평균': tab.groupby('산출')[cols].mean().reindex(prods)},
              axis=1).to_csv(C.OUT / 'grid_summary.csv', encoding='utf-8-sig')
    NAT.to_csv(C.OUT / 'grid_national_daily.csv', float_format='%.3f')

    print(f'\n■ 유역 {tab["유역코드"].nunique()}개   티센 기준   '
          f'{C.EVAL0} ~ {C.EVAL1}   (중앙값)')
    print(tab.groupby('산출')[cols].median().reindex(prods).round(3).to_string())
    print(f'\n■ 전국 면적가중 일강수   {int(nstat["n"].iloc[0])}일')
    print(nstat.round(3).to_string())

    if '--no-fig' in sys.argv:
        return
    plt = C.style()
    C.fig_distribution(plt, tab, prods, LAB, COL, '격자 산출물', 'grid')
    C.fig_national(plt, NAT, nstat, prods, LAB, COL, '격자 산출물', 'grid')


if __name__ == '__main__':
    main()
