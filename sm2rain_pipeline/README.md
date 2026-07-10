# SM2RAIN 강수 산정 파이프라인 (KIHS 과제 최종)

ASCAT NRT 토양수분으로 위성 강수(SM2RAIN)를 산정하고, TCA 병합과
LightGBM 편의보정, LOPO 공간 교차검증까지 수행하는 4단계 파이프라인.

입력자료(ASCAT·ASOS·AWS·GPM·ERA5)의 수집·전처리는 `data_pipeline/` 를
참고하세요. 이 폴더는 그 산출물을 받아 강수장을 만드는 단계입니다.

```
01_SM2RAIN.py          ─┐  ASCAT NRT SM → SM2RAIN 강수 (IDW_AWS 로 캘리브레이션)
02_TCA.py              ─┤  SM2RAIN + ERA5 + IDW_AWS → TCA 병합 강수
03_BC_LightGBM.py      ─┤  SM2RAIN/ERA5/GPM/TCA → LightGBM 편의보정 (BC_1, BC_2)
04_spatial_CV_LOPO.py  ─┘  BC_1/BC_2 의 Leave-One-Pixel-Out 공간 교차검증
```

각 스크립트는 필요한 유틸(지표, 데이터 로딩, 모델 팩토리)을 파일 안에
포함하고 있어 별도 모듈 없이 단독 실행됩니다. 03과 04는 반드시 동일한
모델 설정·데이터 분할을 써야 하므로, 한쪽의 유틸 부분을 수정하면
다른 쪽도 똑같이 맞춰 주세요.

기간 구성: **2021 학습(캘리브레이션) / 2022 검증 / 2023–2025 테스트**

### 자료 흐름 (중요)

위 화살표는 개념 순서일 뿐, 스크립트가 서로의 출력을 자동으로 읽지는
않습니다. 실제 입력은 다음과 같습니다.

- **01** 은 `data_pipeline` 산출물(ASCAT 스택 + IDW_AWS)로 SM2RAIN 을 만든다.
- **02·03·04** 는 모든 멤버(SM2RAIN·ERA5·GPM·IDW_AWS·IDW_ASOS)를 하나로
  합친 `ds_merged_LR.nc` 를 입력으로 받는다. 이 조립 파일과 01 이 쓰는
  `da_IDWs.nc` 는 `data_pipeline/assemble.py` 로 만든다 (동일 격자·기간 정렬).
- **02** 가 추가하는 TCA 변수명은 `TCA_SM2RAIN_ERA5_AWS_2021` 이고,
  **03·04** 는 `TCA` 라는 이름의 변수를 특징으로 기대한다. 02 출력을 03 입력에
  쓰려면 이 변수를 `TCA` 로 맞춰 `ds_merged_LR.nc` 에 넣어야 한다.

전체 순서: `data_pipeline` (다운로드·전처리·IDW) → `assemble.py --step idw`
→ **01** → `assemble.py --step merge` → **02 → 03 → 04**.

### 데이터 배치

경로는 모두 스크립트 위치 기준 상대경로예요. 입력 `.nc` 파일은 `data/`에 넣고,
결과는 `output/`에 저장됩니다 (두 폴더 모두 git 에는 올라가지 않음).

```
sm2rain_pipeline/
├── data/                     # 입력 (직접 넣기, git 제외)
│   ├── ASCAT_daily_stack_KST.nc   # 01: ASCAT NRT 토양수분
│   ├── da_IDWs.nc                 # 01: IDW_AWS (캘리브레이션 기준; IDW 산출을
│   │                              #     연도병합·변수명 정리한 파일)
│   └── ds_merged_LR.nc            # 02/03/04: 멤버 조립 파일 (위 '자료 흐름' 참고)
├── output/                   # 결과 (자동 생성, git 제외)
└── *.py
```

---

## 01_SM2RAIN.py — SM2RAIN 강수 역산

- **데이터**: ASCAT NRT 일별 토양수분 stack (`ASCAT_daily_stack_KST.nc`, KST 일경계,
  픽셀별 0–1 정규화), 캘리브레이션 기준 강수 = IDW_AWS (`da_IDWs.nc`)
- **방법**: Brocca et al. (2014) 기본 3-파라미터 SM2RAIN
  - 물수지: `Z·dθ/dt = P − a·θ^b` → 강수 역산 `P = Z·dθ/dt + a·θ^b` (P<0 → 0)
  - 파라미터: a(배수계수), b(배수지수), Z(유효토심, mm)
  - 픽셀별로 2021 IDW_AWS 대비 MSE 최소화 (L-BFGS-B, multi-start) 캘리브레이션
    후 2021–2025 전체에 적용
  - ASCAT 격자(58×46) → AWS 격자(49×49)는 NaN-aware bilinear 보간
- **출력**: `SM2RAIN_KST.nc` (강수 mm/day + a·b·Z 맵) 와
  `SM2RAIN_params_KST.nc` (a·b·Z 맵). 저장된 파라미터가 있으면 재실행 시
  재최적화를 건너뛴다 (`REUSE_SAVED_PARAMS`).

## 02_TCA.py — Triple Collocation 병합

- **데이터**: `ds_merged_LR.nc` 안의 SM2RAIN, ERA5, AWS(IDW) 일강수
- **방법**: Dong-style TCA, 월별 기후값 anomaly 기반
  1. 산출물별 월별 기후값 제거 → anomaly
  2. 2021년 anomaly 로 픽셀별 rescaling 계수(SM2RAIN 기준)와
     TCA 오차분산 `err2_i = <(x_i−x_j)(x_i−x_k)>` 추정
  3. 가중치 `w_i ∝ 1/err2_i` (합 = 1), 비양수 오차분산은 양수 중앙값으로 대체
  4. 전 기간 anomaly 가중 병합 + SM2RAIN 기후값 복원, 음수 0 클리핑
- **출력**: `ds_merged_LR_TCA_SM2RAIN_ERA5_AWS_2021.nc` (병합 강수 변수
  `TCA_SM2RAIN_ERA5_AWS_2021` 추가) 와
  `TCA_weights_LR_SM2RAIN_ERA5_AWS_2021.nc` (가중치·오차분산·스케일 맵)

## 03_BC_LightGBM.py — LightGBM 편의보정

- **데이터**: `ds_merged_LR.nc` (SM2RAIN, ERA5, GPM, TCA, AWS, ASOS)
- **방법**: LightGBM 회귀가 AWS 일강수를 직접 예측 (없으면
  HistGradientBoosting fallback)
  - **BC_1**: features = [SM2RAIN, ERA5, GPM, TCA, lon, lat]
  - **BC_2**: features = BC_1 + [당일 AWS] — 관측이 있을 때의 상한 성능
  - 2021 강수 이벤트(모든 산출물 > 0)만 학습, 예측 음수는 0 클리핑
  - 하이퍼파라미터: n_estimators=500, lr=0.03, num_leaves=31,
    min_child_samples=30, subsample/colsample=0.9, reg_lambda=1.0
- **평가**: 기간별 산점도 + 픽셀별 R/ubRMSD/bias 맵 (모두 AWS 기준, ASOS 참고)
- **출력**: `ds_merged_LR_BC.nc` (`BC_1`, `BC_2` 변수 추가)

## 04_spatial_CV_LOPO.py — Leave-One-Pixel-Out 공간 교차검증

- **방법**: 03과 동일한 특성/모델 구성으로, 픽셀마다 2021년 학습자료에서
  해당 픽셀을 제외하고 재학습 → 그 픽셀을 예측 (미관측 지점 상황 모사)
  - LOPO_BC_1 / LOPO_BC_2, 픽셀별 n_estimators=100, 최소 학습표본 500
- **평가**: 2023–2025 AWS 대비 픽셀별 R/ubRMSD/bias 맵
- **출력**: `LOPO_metrics_BC1_BC2_2023_2025.nc`

---

## 실행

```bash
python 01_SM2RAIN.py       # ASCAT → SM2RAIN
python 02_TCA.py           # TCA 병합
python 03_BC_LightGBM.py   # BC_1, BC_2 학습/평가
python 04_spatial_CV_LOPO.py  # LOPO 교차검증 (오래 걸림; LOPO_MAX_PIXELS 로 테스트)
```

입력 파일은 `data/`에 넣으면 되고, 다른 위치를 쓰려면 각 스크립트 상단
`DATA_DIR`/`INPUT_PATH` 상수만 바꾸면 됩니다.
의존성: numpy, pandas, xarray, netCDF4, scipy, scikit-learn, lightgbm(선택),
matplotlib, tqdm
