#!/usr/bin/env python3
"""05 — 유역 산출물의 극한강우 재현을 견준다.  [분석용]

03 이 만든 유역 시계열(`LR_basin_daily.nc`)을 같은 유역 면적가중 자료들과
나란히 놓는다.  04 가 일반 성능을 보는 자리라면, 여기는 **큰 비를 얼마나
잡아내는가**에 초점을 둔다.

    기준     티센 유역 일강수 (01 산출)
    평가구간 2022-01-01 ~ 2025-05-01
    대상     LR_basin  BC_G 를 유역평균해 티센에 맞춘 회귀   (03)
             IDW_AWS   지상관측 보간 격자장의 유역평균
             BC_G      LightGBM, 당일 지상관측 입력

    극한 관점  연 최대일 재현비 · 강우강도 구간별 재현비 · 티센 상위 사상

BC 계열은 총량은 잘 맞추면서 극값을 놓치는 경향이 있다.  03 의 회귀는 그 크기를
티센 쪽으로 되돌리는 1차 변환이므로, 상관은 그대로 두고 크기와 편의만 바꾼다.
그래서 여기서 볼 것은 KGE 하나가 아니라 **어느 강우강도 구간에서 무엇이
달라지는가**다.

입력
    output/LR_basin_daily.nc        03
    output/THIESSEN_basin_daily.nc  01
    data/ds_merged_LR.nc            TCA
    data/BC12_fields.nc             BC_1 · BC_2

출력
    output/extreme_basin_metrics.csv   유역 × 산출물 지표
    output/extreme_summary.csv         산출물별 중앙값·평균
    output/extreme_intensity.csv       강우강도 구간별 재현비
    output/extreme_annual_peak.csv     유역 × 연도 최대일 재현비
    output/extreme_national_daily.csv  전국 면적가중 일강수 시계열
    output/extreme_*.png               분포 · 강도별 · 연최대일 · 전국 시계열

실행
    python3 05_evaluate_extreme.py            그림까지
    python3 05_evaluate_extreme.py --no-fig   표만
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr

import common as C

PRODS = ['LR_basin', 'IDW_AWS', 'BC_G']
LAB = {'LR_basin': 'LR (BC-G 를 티센에 맞춤)', 'IDW_AWS': 'IDW_AWS (지상관측 보간)',
       'BC_G': 'BC-G (지상관측 융합)'}
COL = {'LR_basin': '#0F7B8A', 'IDW_AWS': '#2E86C1', 'BC_G': '#E08A2E'}


def annual_peak(THI, S, prods, order):
    """유역 × 연도별로 티센 최대일을 찾아 그날의 재현비."""
    rows = []
    for c in order:
        o = THI[c].loc[C.EVAL0:C.EVAL1]
        v = {k: S[k][c].loc[C.EVAL0:C.EVAL1] for k in prods}
        m = o.notna()
        for x in v.values():
            m &= x.notna()
        if m.sum() < 60:
            continue
        o = o[m]
        for yr, gg in o.groupby(o.index.year):
            if gg.notna().sum() < 30 or gg.max() <= 0:
                continue
            d = gg.idxmax()
            rows.append({'유역코드': c, '연도': yr, '피크일': f'{d:%m-%d}',
                         '티센': round(float(o.loc[d]), 1),
                         **{k: round(float(v[k].loc[d]), 1) for k in prods},
                         **{k + '_비': round(float(v[k].loc[d] / o.loc[d]), 3)
                            for k in prods}})
    return pd.DataFrame(rows)


def main() -> None:
    f_thi = C.OUT / 'THIESSEN_basin_daily.nc'
    f_bas = C.OUT / 'LR_basin_daily.nc'
    C.need(C.F_MERGED, C.F_BC12, f_thi, f_bas)

    ds = xr.open_dataset(C.F_MERGED)
    t = pd.to_datetime(ds.time.values)
    A = {'IDW_AWS': ds['AWS'].values}
    ds.close()
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
    S = {k: C.to_basin(W, A[k], t) for k in A}
    bas = xr.open_dataset(f_bas)
    bcol = [str(c) for c in bas.basin.values]
    S['LR_basin'] = pd.DataFrame(bas['LR_basin'].values, index=t, columns=bcol)
    bas.close()

    prods = [k for k in PRODS if k in S]
    order = [c for c in sorted(W) if c in THI.columns and c in bcol]
    THI = THI[order]
    print('유역 산출물:', ', '.join(prods))

    tab, NAT, nstat, band, inten = C.compare(THI, S, prods, W, order)
    peak = annual_peak(THI, S, prods, order)

    C.OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(C.OUT / 'extreme_basin_metrics.csv', index=False,
               encoding='utf-8-sig')
    cols = ['n', 'KGE', 'R', 'RMSE', '누적비', '연최대일비']
    pd.concat({'중앙값': tab.groupby('산출')[cols].median().reindex(prods),
               '평균': tab.groupby('산출')[cols].mean().reindex(prods)},
              axis=1).to_csv(C.OUT / 'extreme_summary.csv',
                             encoding='utf-8-sig')
    inten.to_csv(C.OUT / 'extreme_intensity.csv', encoding='utf-8-sig')
    peak.to_csv(C.OUT / 'extreme_annual_peak.csv', index=False,
                encoding='utf-8-sig')
    NAT.to_csv(C.OUT / 'extreme_national_daily.csv', float_format='%.3f')

    print(f'\n■ 유역 {tab["유역코드"].nunique()}개   티센 기준   '
          f'{C.EVAL0} ~ {C.EVAL1}   (중앙값)')
    print(tab.groupby('산출')[cols].median().reindex(prods).round(3).to_string())
    print(f'\n■ 전국 면적가중 일강수   {int(nstat["n"].iloc[0])}일')
    print(nstat.round(3).to_string())
    print('\n■ 강우강도 구간별 재현비 (산출/티센)')
    print(inten.round(2).to_string())
    print(f'\n■ 연 최대일 재현비   {len(peak)}건 (유역 × 연도)')
    print(peak[[k + '_비' for k in prods]].describe()
          .loc[['mean', '50%', '25%', '75%']].round(3).to_string())

    if '--no-fig' in sys.argv:
        return
    plt = C.style()
    C.fig_distribution(plt, tab, prods, LAB, COL, '유역 산출물', 'extreme')
    C.fig_national(plt, NAT, nstat, prods, LAB, COL, '유역 산출물', 'extreme')

    names = list(band.cat.categories)
    HI = 2.4
    fig, ax = plt.subplots(figsize=(11, 5.6))
    xx = np.arange(len(names))
    w = .8 / len(prods)
    for i, k in enumerate(prods):
        r = inten.loc[k].to_numpy(float)
        px = xx + (i - (len(prods) - 1) / 2) * w
        ax.bar(px, np.minimum(r, HI), w * .9, color=COL[k], label=LAB[k])
        for xi, rr in zip(px, r):
            if rr > HI:
                ax.annotate(f'{rr:.0f}', (xi, HI), xytext=(0, 3),
                            textcoords='offset points', ha='center',
                            fontsize=9, color=COL[k], fontweight='bold')
    ax.axhline(1, color='#C0392B', lw=1.4, ls='--')
    ax.set_ylim(0, HI * 1.12)
    ax.set_xticks(xx)
    ax.set_xticklabels([f'{b}\n{int((band == b).sum())}일' for b in names])
    ax.set_xlabel('티센 일강수 [mm/일]')
    ax.set_ylabel('산출 / 티센')
    ax.grid(axis='y', alpha=.25)
    ax.legend(fontsize=9)
    ax.set_title('강우강도 구간별 재현비 — 전국 면적가중', fontweight='bold')
    fig.tight_layout()
    C.savefig(plt, fig, 'extreme_intensity')

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.boxplot([peak[k + '_비'].dropna() for k in prods], showfliers=False,
               widths=.6, patch_artist=True,
               boxprops=dict(facecolor='#EEF1F4'),
               medianprops=dict(color='k', lw=2))
    for i, k in enumerate(prods, 1):
        v = peak[k + '_비'].dropna()
        ax.scatter(np.random.normal(i, .06, len(v)), v, s=5, alpha=.12,
                   color=COL[k], zorder=0)
    ax.axhline(1, color='#C0392B', lw=1.4, ls='--')
    ax.set_xticks(range(1, len(prods) + 1))
    ax.set_xticklabels(prods, rotation=20, ha='right')
    ax.set_ylim(0, 2)
    ax.set_ylabel('산출 / 티센')
    ax.grid(axis='y', alpha=.3)
    ax.set_title(f'연 최대일 재현비 — {len(peak)}건 (유역 × 연도)',
                 fontweight='bold')
    fig.tight_layout()
    C.savefig(plt, fig, 'extreme_annual_peak')


if __name__ == '__main__':
    main()
