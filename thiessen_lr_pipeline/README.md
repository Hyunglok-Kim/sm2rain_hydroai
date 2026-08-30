# 티센 목표 선형회귀 강우 산정

편의보정의 **목표자료를 지상관측 내삽장(IDW_AWS)에서 티센 유역평균으로 바꾸면
어떻게 되는지** 확인하는 파이프라인이다. 지점 시자료로 티센 유역 일강수를 만들고,
그것을 목표로 두 가지 선형회귀 산출물을 낸 뒤 각각 다른 관점에서 평가한다.

```
01_thiessen_basin.py     ─┐  지점 시자료 → 티센 유역 일강수      (목표·평가기준)
02_LR_grid.py            ─┼  격자별 회귀 → 전국 격자장          [산출물 ①]
03_LR_basin.py           ─┼  BC-G 유역평균 → 회귀 → 유역 시계열  [산출물 ②]
04_evaluate_grid.py      ─┼  ① 을 다른 격자 산출물과 비교        [분석용]
05_evaluate_extreme.py   ─┘  ② 의 극한강우 재현을 비교           [분석용]
```

기간 구성: **2021 적합 / 2022–2025 평가**. `sm2rain_pipeline` 과 같은 분할이다.

`common.py` 에 경로 설정·표준유역도 파서·면적가중 로직·평가지표·그림 도구를 모아
두었고 다섯 스크립트가 이것을 import 한다. 같은 유틸을 파일마다 복사하지 않으려는
것이고, 그 외 외부 모듈 의존은 없다.

---

## 두 산출물, 두 비교

두 산출물은 **출발점이 다르다.**

| | 출발점 | 회귀 단위 | 결과 |
|---|---|---|---|
| **① `02`** | 위성·재분석 원자료 `SM2RAIN` `ERA5` `GPM` `TCA` | 격자 한 칸씩 | 0.1° 전국 격자장 |
| **② `03`** | 이미 편의보정된 **`BC-G`** | 유역 하나씩 (면적가중 평균 후) | 표준유역 일강수 |

①은 편의보정을 **처음부터 다시** 하되 목표만 티센으로 바꾼 것이다. 격자 자료이므로
04 에서 다른 격자 산출물(`BC-G` `BC` `TCA`)과 견준다.

②는 `BC-G` 를 **출발점으로 두고 크기만 티센에 맞추는** 1차 변환이다. 설명변수가
하나뿐이라 상관계수는 `BC-G` 와 같고 크기와 편의만 바뀐다. `BC-G` 가 총량은 잘
맞추면서 극값을 놓치는 성질을 되돌리려는 것이므로, 05 에서 같은 유역 면적가중
자료들과 **극한강우 관점**으로 견준다.

격자의 소속 유역(02)은 겹치는 면적이 가장 큰 유역으로 정한다. 0.1° 한 칸이 약
100 km² 이고 표준유역 중앙값이 113 km² 라 한 칸이 여러 유역에 걸치지만, 회귀가
성립하려면 목표를 하나로 정해야 한다.

### 비교 대상

무엇을 무엇과 견주는지 정리하면 이렇다. **기준은 두 경우 모두 티센 유역 일강수**이고,
모든 자료를 표준유역 면적가중 평균으로 맞춘 뒤 산출물이 다 있는 날로 표본을 맞춘다.

| | `04_evaluate_grid.py` | `05_evaluate_extreme.py` |
|---|---|---|
| **무엇을** | `LR_grid` (산출물 ①) | `LR_basin` (산출물 ②) |
| **무엇과** | `BC-G` · `BC` · `TCA` | `BC-G` · `BC` · `TCA` |
| **자료 성격** | 0.1° 격자 산출물끼리 | 유역 면적가중 자료끼리 |
| **보는 것** | 일반 성능 — KGE · R · RMSE · 누적비 | 극한강우 — 연 최대일 재현비 · 강우강도 구간별 재현비 |

비교군 셋은 이 저장소가 이미 내는 산출물이다.

| 이름 | 어디서 | 설명 |
|---|---|---|
| `BC-G` | `sm2rain_pipeline/03_BC_LightGBM.py` 의 `BC_2` | LightGBM 편의보정. 같은 날 지상관측을 입력으로 받는 융합 산출물 |
| `BC` | 같은 파일의 `BC_1` | LightGBM 편의보정. 지상관측을 입력으로 쓰지 않는다 |
| `TCA` | `sm2rain_pipeline/02_TCA.py` | 삼중병치분석 병합장. 편의보정 전 |

**LR 산출물끼리는 견주지 않는다.** ①과 ②는 출발점도 자료 단위도 달라서 나란히
놓을 값이 아니다. 각각을 위 비교군과 견주는 것이 이 파이프라인의 목적이다.

---

## 산출물과 분석용 자료

`output/` 에 만들어지는 파일은 넷으로 나뉜다.

### 최종 산출물

| 파일 | 만든 곳 | 내용 |
|---|---|---|
| **`LR_grid.nc`** | 02 | **전국 0.1° 격자 강우장** (time × lat × lon). 격자별 회귀계수 `coef` 와 적합일수 `n_fit` 을 같이 담는다 |
| **`LR_basin_daily.nc`** | 03 | **표준유역 일강수** (time × basin) |
| **`LR_basin_daily.csv`** | 03 | 위와 같은 값. 행=날짜, 열=유역코드 |

### 기준 자료

회귀의 목표이자 평가 기준이다. 산출물이 아니라 **비교 대상**이다.

| 파일 | 만든 곳 | 내용 |
|---|---|---|
| `THIESSEN_basin_daily.nc` `.csv` | 01 | 티센 다각형 유역 일강수 |
| `THIESSEN_basin_weights.csv` | 01 | 유역별 기여지점과 가중치 |

### 분석용 자료

성능을 견주려고 만든 표와 그림이다. 납품 대상이 아니다.

| 파일 | 만든 곳 | 내용 |
|---|---|---|
| `LR_basin_coef.csv` | 03 | 유역별 회귀계수·절편·적합일수 |
| `grid_basin_metrics.csv` `grid_summary.csv` | 04 | 격자 산출물 유역별 지표 |
| `grid_national_daily.csv` | 04 | 격자 산출물 전국 면적가중 시계열 |
| `grid_*.png` | 04 | 유역별 분포 · 전국 월평균 · 전국 누적 |
| `extreme_basin_metrics.csv` `extreme_summary.csv` | 05 | 유역 산출물 지표 |
| `extreme_intensity.csv` | 05 | 강우강도 구간별 재현비 |
| `extreme_annual_peak.csv` | 05 | 유역 × 연도 최대일 재현비 |
| `extreme_national_daily.csv` | 05 | 유역 산출물 전국 면적가중 시계열 |
| `extreme_*.png` | 05 | 분포 · 강도별 · 연최대일 · 전국 시계열 |

### 중간 파일

캐시다. 지우면 다시 만들어진다.

| 파일 | 만든 곳 | 내용 |
|---|---|---|
| `station_daily.pkl` | 01 | 지점 일강수와 좌표 |
| `basin_cell_weights.pkl` | 01–05 | 유역×격자 교차면적 |

---

## 읽을 때 주의

**회귀 산출물은 목표와 평가기준이 같은 자료다.** 티센 점수가 좋게 나오기 쉬운
판이므로 "더 낫다"가 아니라 **"제 목표를 얼마나 따라가는가"** 로 읽는다. 실제로
볼 것은 유역별 편차와 관측밀도 의존성이다.

**유역 단위 평가와 전국 평균은 다른 것을 잰다.** 유역마다 다른 날 오는 국지 사상이
전국 평균에서는 희석되고 광역 사상만 남는다. 04·05 는 둘을 따로 내며 섞어 쓰면
안 된다.

**티센은 지점이 대표하는 면적을 기하학적으로만 정한다.** 유역 안팎에 지점이 없으면
먼 지점 하나가 유역 전체를 대표한다. `THIESSEN_basin_weights.csv` 에서 기여지점이
1–2개인 유역의 값은 그렇게 읽어야 한다.

---

## 데이터 배치

경로는 스크립트 위치 기준 상대경로다. 입력 `.nc` 는 `data/`, 결과는 `output/` 에
저장된다. 둘 다 git 에서 제외된다.

```
thiessen_lr_pipeline/
├── data/                      # 입력 (직접 넣기, git 제외)
│   ├── ds_merged_LR.nc            02/04/05: SM2RAIN·ERA5·GPM·TCA 조립본
│   │                              (sm2rain_pipeline 의 02/03/04 와 같은 파일)
│   ├── BC12_fields.nc             03/04/05: BC_1·BC_2 (03_BC_LightGBM.py 산출)
│   ├── basin.shp .dbf .shx        01–05: 국가 표준유역도
│   │   .prj .cpg
│   └── AWS_hourly_2021.csv …      01: 기상청 시자료 (연도별)
├── output/                    # 결과 (자동 생성, git 제외)
└── *.py
```

다른 곳에 두었다면 환경변수로 덮어쓴다. 파일을 옮기거나 이름을 바꿀 필요가 없다.

```bash
export TLR_DATA=/my/inputs                       # 기본: ./data
export TLR_OUT=/my/outputs                       # 기본: ./output
export TLR_BASIN_SHP=/my/inputs/std_basin_850    # 확장자 없는 stem
```

### 입력 자료

| 파일 | 어디서 | 필요한 것 |
|---|---|---|
| `ds_merged_LR.nc` | `data_pipeline/assemble.py --step merge` | 변수 `SM2RAIN` `ERA5` `GPM` `TCA`, 차원 `(time, lat, lon)` |
| `BC12_fields.nc` | `sm2rain_pipeline/03_BC_LightGBM.py` | 변수 `BC_1` `BC_2` |
| `basin.shp` 일습 | 국가 표준유역도 (환경부) | 속성에 유역코드(`SBSN_CD` 등 `CD` 를 포함하는 열)와 유역명 |
| `AWS_hourly_YYYY.csv` | `data_pipeline/AWS/AWS_download.py` | 열 `일시` `지점` `위도` `경도` `강수량` |

표준유역도는 배포 권한 문제로 저장소에 넣지 않는다. shapefile 은 `pyproj`·
`geopandas` 없이 직접 읽으며 폴리곤(5)·멀티폴리곤(15) 두 형식을 다룬다.

입력이 없으면 스크립트가 **무엇을 어디에 두어야 하는지 알려주고 멈춘다.**

### 설정 바꾸기

기간·설명변수 같은 값은 `common.py` 위쪽에 모여 있다.

```python
FIT_YEAR = '2021'                            # 회귀 적합 연도
EVAL0, EVAL1 = '2022-01-01', '2025-05-01'    # 평가 구간
X_GRID = ['SM2RAIN', 'ERA5', 'GPM', 'TCA']   # 02 격자별 회귀의 설명변수
X_BASIN = ['BC_G']                           # 03 유역별 회귀의 설명변수
BC_MAP = {'BC_G': 'BC_2', 'BC': 'BC_1'}      # BC12_fields.nc 의 변수명 대응
MIN_AREA_FRAC = 0.5    # 유역 유효면적이 이보다 작은 날은 결측
MIN_FIT = 60           # 적합할 날이 이보다 적으면 그 격자·유역은 비운다
```

---

## 실행

```bash
cd thiessen_lr_pipeline
python3 01_thiessen_basin.py         # 20초 + 시자료 읽기
python3 02_LR_grid.py                # 1분 이내
python3 03_LR_basin.py               # 1분 이내
python3 04_evaluate_grid.py          # 표 + 그림
python3 05_evaluate_extreme.py       # 표 + 그림
```

04·05 에 `--no-fig` 를 주면 표만 만든다.

01 은 지점 일강수를 `output/station_daily.pkl` 에, 유역×격자 교차면적을
`output/basin_cell_weights.pkl` 에 캐시한다. 두 번째 실행부터는 그 단계를 건너뛴다.
입력을 바꿨다면 캐시를 지우고 다시 돌린다.

필요한 패키지: `numpy` `pandas` `xarray` `netCDF4` `shapely` `matplotlib`.

---

## 01_thiessen_basin.py — 티센 유역 일강수

02·03 의 **회귀 목표**이자 04·05 의 **평가 기준**이다.

- **지점 가중치** — 유역을 250 m 격자로 잘게 나눠 각 칸을 최근접 관측지점에
  배정하고 그 면적 비율을 가중치로 삼는다. 투영은 EPSG:5179 (한국 중부원점 TM).
  폴리곤 교차를 직접 푸는 것과 결과가 같고 훨씬 간단하다.
- **유역 일강수** — `P_b(t) = Σ wᵢ·Pᵢ(t) / Σ wᵢ`. 그날 값이 있는 지점만 더하고
  가중치를 다시 정규화한다. 값이 있는 지점이 하나도 없으면 그날은 결측이다.
- **하루 경계** — KST 01시 ~ 익일 00시. 연 파일 경계일(12-31)은 두 파일에 나뉘어
  있으므로 같은 날짜끼리 합친다.

출력 `THIESSEN_basin_daily.csv` `.nc` `THIESSEN_basin_weights.csv`

## 02_LR_grid.py — 격자별 선형회귀 　[산출물 ①]

격자 한 칸마다 최소제곱 회귀를 따로 적합한다. 목표는 그 칸이 속한 유역의 티센
일강수, 설명변수는 `SM2RAIN` `ERA5` `GPM` `TCA` 다. 2021년으로 적합해 전 기간에
적용하고 음수는 0으로 자른다.

출력 `LR_grid.nc`

| 변수 | 차원 | 내용 |
|---|---|---|
| `LR` | time × lat × lon | 강우 격자장 (mm/day) |
| `coef` | term × lat × lon | 격자별 회귀계수와 절편 |
| `n_fit` | lat × lon | 격자별 적합에 쓴 날 수 |

## 03_LR_basin.py — BC-G 를 티센에 맞춘 유역 회귀 　[산출물 ②]

`BC-G` 를 유역 면적가중 평균한 뒤, 유역마다 티센을 목표로 회귀를 적합한다. 유효
격자 면적 합이 유역 면적의 절반 미만인 날은 결측으로 둔다.

설명변수가 하나뿐인 1차 변환이라 **상관계수는 `BC-G` 와 같고 크기와 편의만
바뀐다.** 계수가 1보다 크면 `BC-G` 를 키우고 작으면 줄인다.

출력 `LR_basin_daily.csv` `.nc` `LR_basin_coef.csv`

## 04_evaluate_grid.py — 격자 산출물 비교 　[분석용]

`LR_grid` 를 `BC-G` `BC` `TCA` 와 견준다. 모두 표준유역으로 면적가중 집계한 뒤
티센 기준으로 평가하고, 산출물이 모두 있는 날로 표본을 맞춘다.

출력 `grid_basin_metrics.csv` `grid_summary.csv` `grid_national_daily.csv`
`grid_basin_distribution.png` `grid_national_monthly.png`
`grid_national_cumulative.png`

## 05_evaluate_extreme.py — 극한강우 비교 　[분석용]

`LR_basin` 을 같은 유역 면적가중 자료(`BC-G` `BC` `TCA`)와 견주되 **큰 비를 얼마나
잡아내는가**에 초점을 둔다.

- **연 최대일 재현비** — 유역 × 연도마다 티센 최대일을 찾아 그날의 산출/티센
- **강우강도 구간별 재현비** — 전국 면적가중 일강수를 6개 구간으로 나눠 구간마다 산출/티센
- 그 밖에 04 와 같은 유역별 분포·전국 시계열

출력 `extreme_basin_metrics.csv` `extreme_summary.csv` `extreme_intensity.csv`
`extreme_annual_peak.csv` `extreme_national_daily.csv` `extreme_*.png`

---

## References

### 방법
- Thiessen, A. H. (1911). Precipitation averages for large areas.
  *Monthly Weather Review*, 39(7), 1082–1089.
  https://doi.org/10.1175/1520-0493(1911)39<1082b:PAFLA>2.0.CO;2
  *(티센 다각형 유역평균)*
- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
  Decomposition of the mean squared error and NSE performance criteria:
  Implications for improving hydrological modelling. *Journal of Hydrology*,
  377(1–2), 80–91. https://doi.org/10.1016/j.jhydrol.2009.08.003  *(KGE)*

### 좌표계
- 국토지리정보원. 한국 중부원점 TM (EPSG:5179, Korea 2000 / Unified CS).
  티센 가중치 산정에서 면적을 재기 위해 이 좌표계로 투영한다.
