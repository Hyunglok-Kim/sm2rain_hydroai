# 데이터 수집·전처리 파이프라인 (KIHS 과제)

SM2RAIN 강수 파이프라인(`sm2rain_pipeline/`)에 들어가는 입력자료를
데이터셋별로 **다운로드 → (필요시) KST 전처리** 하는 코드 모음.

```
data_pipeline/
├── ASCAT_NRT/
│   ├── ASCAT_NRT_download.py      EUMETSAT에서 ASCAT L2 토양수분(SOMO12) 원본 다운로드
│   └── ASCAT_NRT_KST_process.py   원본 → 한국 0.1° 격자 → KST 일별 스택 (한 파일)
├── ASOS/
│   └── ASOS_download.py           기상청 종관관측 일강수 (이미 KST → 다운로드만)
├── AWS/
│   └── AWS_download.py            기상청 방재관측 일강수 (이미 KST → 다운로드만)
├── GPM/
│   ├── GPM_download.py            GES DISC에서 IMERG 30분 원본(V07) 다운로드
│   └── GPM_KST_process.py         30분 원본 → +9h → KST 일강수 → 0.125° 한국격자
├── ERA5_Land/
│   ├── ERA5_Land_download.py      CDS에서 ERA5-Land 시간별 강수 다운로드
│   └── ERA5_Land_KST_process.py   시간별 누계 해제 + +9h → KST 일강수 (한 파일)
├── IDW/
│   └── IDW_interpolation.py       ASOS/AWS 지점 일강수 → 0.125° 격자 (셀평균 + IDW)
└── assemble.py                   위 산출물 → sm2rain_pipeline 입력으로 조립
                                  (da_IDWs.nc / ds_merged_LR.nc)
```

**IDW 보간**: 지상관측(ASOS/AWS)은 지점 자료라 격자 강수로 바꿔야 하며,
셀평균 전처리(같은 셀 지점 먼저 평균) 후 역거리가중(1/d³)으로 보간한다.
ASOS/AWS 다운로드가 끝난 뒤 실행하며, SM2RAIN·GPM 과 동일한 0.125° 49×49
격자로 산출해 `da_IDWs` 로 바로 이어진다.

**KST 처리 기준**: 위성·재분석 자료(ASCAT, GPM, ERA5)는 원본이 UTC라서
30분/시간별/스와스 원본 단계에서 +9h 후 KST 하루(00–24시 KST)로 재집계한다.
이미 일합산된 자료는 라벨만 바뀌므로 반드시 원본 단계에서 처리해야 한다.
(ERA5는 시간별 누계라 누계 해제까지 함께 처리한다.)
지상관측(ASOS, AWS)은 원래 KST 일자료라 추가 처리가 없다.

**검증 기준에 대하여**: IDW_ASOS 는 학습에 쓰지 않으므로 학습 자료로 채점하는
순환은 피하지만, 96개 종관 지점을 공간보간한 추정장이라 참값이 아니다. 관측이
희소한 지역에서는 자체 보간오차를 포함하므로, 지표는 참값 대비 오차가 아니라
이 비교 기준에 대한 재현도로 읽어야 한다.

## 인증키 설정 (코드에 키를 넣지 않는다)

| 데이터 | 발급처 | 설정 방법 |
|---|---|---|
| ASCAT | https://api.eumetsat.int/api-key/ | `export EUMDAC_CONSUMER_KEY=...`<br>`export EUMDAC_CONSUMER_SECRET=...` |
| ASOS | data.go.kr (ASOS 일자료 조회서비스) | `export KMA_SERVICE_KEY=...` |
| AWS | apihub.kma.go.kr | `export KMA_APIHUB_KEY=...` |
| GPM | urs.earthdata.nasa.gov (GESDISC 앱 승인) | `~/.netrc` 에 계정 등록 (권한 600) |
| ERA5 | cds.climate.copernicus.eu | `~/.cdsapirc` 에 url/key 등록 |

## 실행 순서

각 산출물은 해당 폴더 안의 `output/` 에 저장된다 (공유 폴더 아님).

```bash
# ASCAT (SM2RAIN 입력)
python ASCAT_NRT/ASCAT_NRT_download.py       # → ASCAT_NRT/raw/
python ASCAT_NRT/ASCAT_NRT_KST_process.py    # → ASCAT_NRT/output/ASCAT_daily_stack_KST.nc

# 지상관측 (보정·검증 기준)
python ASOS/ASOS_download.py                 # → ASOS/output/ASOS_daily_*.csv
python AWS/AWS_download.py                   # → AWS/output/AWS_daily_*.csv
python IDW/IDW_interpolation.py              # → IDW/output/Precipitation_IDW_{AWS|ASOS}_{YYYY}.nc

# GPM (편향보정 입력특징)
python GPM/GPM_download.py                   # → GPM/raw/ (30분 HDF5, 용량 큼)
python GPM/GPM_KST_process.py                # → GPM/output/GPM_{YYYY}_KST.nc

# ERA5-Land (TCA 입력)
python ERA5_Land/ERA5_Land_download.py       # → ERA5_Land/raw/ (시간별 강수)
python ERA5_Land/ERA5_Land_KST_process.py    # → ERA5_Land/output/era5_land_P_KST.nc
```

각 폴더의 `raw/`(원본)와 `output/`(산출물)은 자동 생성되며 git에는 올리지
않는다. 다운로드 스크립트는 이미 받은 파일을 건너뛰므로 중단 후
재실행해도 이어받기 된다. IDW 는 ASOS/AWS 다운로드가 끝난 뒤 실행한다
(두 관측망의 `output/` CSV 를 읽는다).

## 조립 (sm2rain_pipeline 입력 만들기)

위 산출물들은 그대로 쓰이지 않고 `assemble.py` 로 두 입력파일로 합친다.
da_IDWs 는 SM2RAIN 산정 전에, ds_merged_LR 은 SM2RAIN 산정 후에 필요하므로
**단계를 나눠** 실행한다 (모든 산출물을 0.125° 49×49 격자로 정렬).

```bash
# 1) IDW 격자 합치기 → da_IDWs.nc  (01_SM2RAIN 캘리브레이션 기준)
python assemble.py --step idw          # → assembled/da_IDWs.nc

#    da_IDWs.nc 를 sm2rain_pipeline/data/ 로 복사 후 01_SM2RAIN.py 실행
#    → SM2RAIN_KST.nc 생성

# 2) 전체 멤버 합치기 → ds_merged_LR.nc  (02·03·04 입력)
python assemble.py --step merge        # → assembled/ds_merged_LR.nc
#    (SM2RAIN 산출이 없으면 안내만 하고 건너뜀)
```

`ds_merged_LR.nc` 의 변수는 SM2RAIN·ERA5·GPM·AWS·ASOS·LON·LAT 이며,
`TCA` 변수는 이후 `02_TCA.py` 가 추가한다. SM2RAIN 산출 경로가 다르면
`assemble.py` 상단 `SM2RAIN_PATH` 만 바꾼다.

## 선택 입력

- `ASCAT_NRT/data/porosity.nc` — TU Wien static layer 공극률.
  없으면 `sm_volumetric`이 NaN으로 저장된다 (SM2RAIN은 픽셀별 0–1 정규화를
  쓰므로 `degree_of_saturation`으로 대체 가능).
- `GPM/data/GPM_ref_mask.nc` — 한국 육지마스크 참조용 기존 GPM nc.
  없으면 마스크 없이 전 격자를 출력한다.
- `ASOS/data/Station_ASOS.csv` — ASOS 관측소 목록('지점' 컬럼 필수).
  기상청 관측지점정보에서 내려받는다.
- `IDW/data/Station_ASOS.csv` — ASOS 좌표 메타(지점+경도관서/위도관서).
  ASOS 다운로드 CSV에는 좌표가 없어 IDW 단계에서 병합에 쓴다.
  (AWS는 다운로드 CSV에 좌표가 포함되어 메타 불필요)
- `IDW/data/Korea.shp` — 한국 경계. 없으면 육지마스크 없이 전 격자 출력.

## sm2rain_pipeline 과의 연결

| 이 폴더의 산출물 | sm2rain_pipeline 에서의 역할 |
|---|---|
| `ASCAT_daily_stack_KST.nc` | `01_SM2RAIN.py` 입력 (토양수분) |
| `Precipitation_IDW_AWS_{YYYY}.nc` | SM2RAIN 보정 목표·TCA 멤버(`da_IDWs`) |
| `Precipitation_IDW_ASOS_{YYYY}.nc` | 비교 검증 기준 (학습에 쓰지 않는다) |
| `era5_land_P_KST.nc` | `02_TCA.py` 입력 멤버(ERA5) |
| `GPM_{YYYY}_KST.nc` | `03_BC_LightGBM.py` 입력특징(GPM) |

이 산출물들은 `sm2rain_pipeline` 에서 곧바로 읽히는 게 아니라, 동일 격자·
기간으로 하나의 `ds_merged_LR.nc` (와 `da_IDWs.nc`)로 조립한 뒤 사용한다.
조립 방식은 `sm2rain_pipeline/README.md` 의 '자료 흐름' 을 참고. (ERA5 는
native 격자로 저장되며 TCA 단계에서 49×49 로 regrid 된다.)

의존성: numpy, pandas, xarray, netCDF4, h5py, scipy, requests, eumdac,
ascat, cdsapi, tqdm  (IDW shapefile 클립 시 geopandas, shapely)
