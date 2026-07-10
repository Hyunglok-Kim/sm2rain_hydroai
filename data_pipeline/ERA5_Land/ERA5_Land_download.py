"""
ERA5_Land_download.py  (Copernicus CDS에서 ERA5-Land 시간별 강수 다운로드)
=========================================================================
Copernicus Climate Data Store(cdsapi)에서 ERA5-Land 시간별 총강수량(tp)을
일 단위 파일로 병렬 다운로드한다. (원본은 UTC 시간별 누계, 단위 m)

KST 일강수 재집계(ERA5_Land_KST_process.py)에는 하루 24시간이 모두 필요하므로
시간별 자료를 그대로 받는다. 파이프라인에서 ERA5는 강수만 쓰므로 기본 변수는
total_precipitation 하나로 두었다 (VARIABLES 에 추가하면 다른 변수도 받음).

준비 (CDS 인증)
  1) https://cds.climate.copernicus.eu 가입 → 본인 프로필에서 API key 확인
  2) 홈 디렉터리에 ~/.cdsapirc 생성 (키를 코드/깃에 절대 넣지 말 것):
       url: https://cds.climate.copernicus.eu/api
       key: <UID>:<API-KEY>
  3) ERA5-Land 데이터셋 라이선스 동의 (웹에서 1회)
  4) pip install cdsapi

입력 : 없음 (CDS API에서 직접 요청)
출력 : raw/{YYYY}/{YYYY.MM.DD}/ERA5_Land_P_{YYYYMMDD}.nc
       (ERA5_Land_KST_process.py 가 읽는 폴더 구조와 동일)

* 이미 받은 파일(같은 이름, 크기>0)은 건너뛰므로 재실행 시 이어받기 된다.
* KST 경계 재집계를 위해 요청 시작연도 직전 해 12/31 하루도 함께 받는다.
"""
import calendar
import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi

# ==============================================================================
# 설정
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")

DATASET = "reanalysis-era5-land"
VARIABLES = ["total_precipitation"]        # 파이프라인은 강수만 사용
TIME_LIST = [f"{h:02d}:00" for h in range(24)]

# 한반도 영역 [North, West, South, East] — 보고서 격자(32.5–39.5N, 123.5–130.5E)
BOUNDS = [39.5, 123.5, 32.5, 130.5]

YEARS = range(2021, 2026)                  # 2021 ~ 2025
MAX_WORKERS = 8                            # CDS 동시요청 (과하면 큐 지연/차단)


# ==============================================================================
# 다운로드
# ==============================================================================
def target_dates():
    """요청 기간의 모든 날짜 + KST 경계용 직전 해 12/31."""
    dates = []
    y0 = min(YEARS)
    dates.append(datetime.date(y0 - 1, 12, 31))   # KST 1/1 경계 채우기용
    for year in YEARS:
        for month in range(1, 13):
            _, ndays = calendar.monthrange(year, month)
            for day in range(1, ndays + 1):
                dates.append(datetime.date(year, month, day))
    return dates


def download_one(day):
    ymd = day.strftime("%Y%m%d")
    out_dir = os.path.join(RAW_DIR, f"{day.year}", day.strftime("%Y.%m.%d"))
    out_path = os.path.join(out_dir, f"ERA5_Land_P_{ymd}.nc")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "skip"

    os.makedirs(out_dir, exist_ok=True)
    request = {
        "variable": VARIABLES,
        "year": day.strftime("%Y"),
        "month": day.strftime("%m"),
        "day": [day.strftime("%d")],
        "time": TIME_LIST,
        "area": BOUNDS,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    try:
        cdsapi.Client().retrieve(DATASET, request).download(out_path)
        return "ok"
    except Exception as e:  # noqa: BLE001
        print(f"  [실패] {ymd}: {e}")
        return "err"


def main():
    cdsrc = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(cdsrc):
        raise SystemExit(
            "~/.cdsapirc 가 없습니다. CDS url/key 를 먼저 설정하세요.\n"
            "  (https://cds.climate.copernicus.eu 프로필의 API key 참고)"
        )
    os.makedirs(RAW_DIR, exist_ok=True)

    dates = target_dates()
    print(f"ERA5-Land 시간별 강수 다운로드: {dates[0]} ~ {dates[-1]} ({len(dates)}일)")

    n = {"ok": 0, "skip": 0, "err": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(download_one, d): d for d in dates}
        for fut in as_completed(futs):
            n[fut.result()] += 1
    print(f"완료: 신규 {n['ok']}, 건너뜀 {n['skip']}, 실패 {n['err']}  → {RAW_DIR}")
    print("다음 단계: ERA5_Land_KST_process.py 실행 (누계 해제 + KST 일강수)")


if __name__ == "__main__":
    main()
