"""
GPM_download.py  (NASA GES DISC에서 GPM IMERG 30분 원본 다운로드)
================================================================
GES DISC의 GPM IMERG Half-Hourly V07(GPM_3IMERGHH.07) HDF5 파일을
기간 지정으로 내려받는다. 하루 = 30분 파일 48개.

KST 일강수 재집계(GPM_KST_process.py)에 30분 원본이 필요하기 때문에
일자료(3IMERGDF)가 아니라 반드시 Half-Hourly 를 받는다.

준비 (NASA Earthdata 인증)
  1) https://urs.earthdata.nasa.gov 계정 생성
     → 프로필에서 "NASA GESDISC DATA ARCHIVE" 앱 승인(Authorize)
  2) 홈 디렉터리에 ~/.netrc 파일 생성 (권한 600):
       machine urs.earthdata.nasa.gov
           login <아이디>
           password <비밀번호>
     (키/비밀번호를 코드/깃에 절대 넣지 말 것)

입력 : 없음 (서버 디렉터리 목록을 직접 조회)
출력 : raw/{YYYY}/{MM}/{DD}/3B-HHR.MS.MRG.3IMERG.*.HDF5
       (GPM_KST_process.py 가 읽는 폴더 구조와 동일)

* 이미 받은 파일은 건너뛰므로 중단 후 재실행해도 이어받기 된다.
* V07A/V07B 등 버전 접미가 섞여 있어도 서버 목록에서 파일명을 직접
  읽어오므로 상관없다.
"""
import os
import re
from datetime import date, timedelta

import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 최소 환경 대비 편의 처리
    def tqdm(iterable, *args, **kwargs):
        return iterable

# ==============================================================================
# 설정
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")

BASE_URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07"

START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 5, 1)

# KST 재집계 경계용: 시작 전날(UTC 12/31)도 필요하므로 하루 앞서 받는다
INCLUDE_PREV_DAY = True

_HDF5_RE = re.compile(r'href="(3B-HHR\.MS\.MRG\.3IMERG\.\d{8}-S\d{6}-E\d{6}\.\d{4}\.V\d{2}[A-Z]?\.HDF5)"')


# ==============================================================================
# 다운로드
# ==============================================================================
def daterange(d0, d1):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def list_day_files(session, day):
    """해당 UTC 날짜 폴더의 HDF5 파일명 목록을 서버에서 조회."""
    doy = day.timetuple().tm_yday
    day_url = f"{BASE_URL}/{day.year}/{doy:03d}/"
    r = session.get(day_url, timeout=120)
    if r.status_code != 200:
        print(f"  [경고] 목록 조회 실패 {day} (HTTP {r.status_code})")
        return day_url, []
    names = sorted(set(_HDF5_RE.findall(r.text)))
    return day_url, names


def download_file(session, url, out_path):
    """Earthdata 리다이렉트 인증을 따라가며 파일 저장."""
    with session.get(url, stream=True, timeout=300) as r:
        if r.status_code == 401:
            raise SystemExit(
                "인증 실패(401). ~/.netrc 설정과 GES DISC 앱 승인 여부를 확인하세요."
            )
        r.raise_for_status()
        tmp_path = out_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        os.replace(tmp_path, out_path)   # 완료 후 원자적 이동 (부분파일 방지)


def main():
    start = START_DATE - timedelta(days=1) if INCLUDE_PREV_DAY else START_DATE
    n_days = (END_DATE - start).days + 1
    print(f"GPM IMERG 30분(V07) 다운로드: {start} ~ {END_DATE} ({n_days}일, 하루 48파일)")

    # .netrc 인증 사용 (requests 가 자동으로 읽음)
    session = requests.Session()

    n_new = n_skip = 0
    for day in tqdm(list(daterange(start, END_DATE)), desc="일자"):
        out_dir = os.path.join(RAW_DIR, f"{day.year}", f"{day.month:02d}", f"{day.day:02d}")
        os.makedirs(out_dir, exist_ok=True)

        day_url, names = list_day_files(session, day)
        if len(names) < 48:
            print(f"  [안내] {day}: 서버 파일 {len(names)}/48개")

        for name in names:
            out_path = os.path.join(out_dir, name)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                n_skip += 1
                continue
            try:
                download_file(session, day_url + name, out_path)
                n_new += 1
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  [실패] {name}: {e}")

    print(f"완료: 신규 {n_new:,}, 건너뜀 {n_skip:,}  → {RAW_DIR}")
    print("다음 단계: GPM_KST_process.py 실행 (30분 → KST 일강수)")


if __name__ == "__main__":
    main()
