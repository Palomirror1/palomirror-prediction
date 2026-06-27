#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_imu.py
==================
공개 IMU 데이터셋(`final IMU dataset`) → 상지재활 미래-IMU 예측용 IMU 정제.

센서 3개 추출 (오른팔 기준):
  - CHS (Chest,        가슴)   → 몸통 보상작용 감지
  - RU  (Right Upper arm, 위팔) → 어깨 IMU
  - RF  (Right Forearm,  아래팔) → 손목 IMU
  ※ 자력계(Magn)는 자기간섭/경량성 이유로 제외 → 각 센서 Acc+Gyro 6축만 사용 (총 18채널)
  ※ 관절각(elbow_flex/shoulder_elev)은 산출하지 않음 — 동작해석기의 출력이라
     예측모델 입력 feature에 두지 않음. (필요시 U/F에서 언제든 재계산 가능)

파이프라인:
  1) laptop1(_u) CSV에서 CHS/RU/RF의 Acc+Gyro 18채널 추출
  2) 결측치(빈셀/NaN) 선형보간 + 양끝/큰갭 NaN 트림 (최장 연속 유효구간)
  3) 샘플인덱스 → 시간축 → TARGET_HZ 균일 리샘플
  4) 저장: trial별 처리 CSV (time + 18 IMU 채널)

  ※ 미래-IMU 예측용 슬라이딩 윈도우/타깃 생성은 make_a9_phase_forecast.py가 담당.

의존성: numpy, python 표준 라이브러리만 (pandas/scipy 불필요)

사용 예:
  python3 preprocess_imu.py --source-hz 100            # 전체 처리 (소스 100Hz 가정)
  python3 preprocess_imu.py --subjects subject1 subject2 --plot-check  # 일부만 + 검증출력
"""

import argparse
import csv
import glob
import math
import os
import re

import numpy as np

# ----------------------------------------------------------------------------
# 기본 설정 (CLI로 덮어쓰기 가능)
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    data_root="final IMU dataset",
    out_root="processed",
    side="R",            # 'R'(RU,RF) 또는 'L'(LU,LF). CHS(가슴)는 side 무관 항상 포함
    source_hz=100.0,     # ⚠️ 원본 샘플링레이트 — 데이터셋 문서로 반드시 확인! (timestamp는 샘플인덱스라 추정 불가)
    target_hz=50.0,      # 작업 샘플링레이트
    interp_limit=20,     # 연속 결측 보간 허용 최대 샘플 수 (초과 구간은 trial 제외)
    min_samples=30,      # 리샘플 후 이보다 짧으면 trial 제외
)


# ============================================================================
# 신호 처리 유틸 (scipy 없이)
# ============================================================================
def interp_nan_1d(x, limit):
    """1D 배열 내부 NaN 선형보간. 연속 NaN > limit 구간은 NaN 유지(=후속 제외)."""
    x = x.astype(float).copy()
    n = len(x)
    isnan = np.isnan(x)
    if not isnan.any():
        return x
    valid = ~isnan
    if valid.sum() < 2:
        return x
    idx = np.arange(n)
    x_interp = np.interp(idx, idx[valid], x[valid])
    # 연속 NaN 길이가 limit 초과인 구간은 되돌려 NaN 유지
    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            if (j - i) > limit:
                x_interp[i:j] = np.nan
            i = j
        else:
            i += 1
    return x_interp


def resample_uniform(t_src, X, target_hz):
    """비균일/원본 시간축 t_src(초) → 0..T 균일 target_hz 그리드로 채널별 선형보간."""
    if len(t_src) < 2:
        return None, None
    t0, t1 = t_src[0], t_src[-1]
    n_new = int(math.floor((t1 - t0) * target_hz)) + 1
    if n_new < 2:
        return None, None
    t_new = t0 + np.arange(n_new) / target_hz
    Xr = np.empty((n_new, X.shape[1]))
    for c in range(X.shape[1]):
        Xr[:, c] = np.interp(t_new, t_src, X[:, c])
    return t_new, Xr


# ============================================================================
# CSV 입출력
# ============================================================================
CANONICAL_SENSORS = ["CHS", "RF", "RU", "LU", "LF"]  # laptop1 표준 센서 집합
_ACCX_SUFFIX = "_IMU9_Acc_X"


def resolve_sensor_blocks(header_stripped):
    """
    헤더에서 센서별 Acc_X 컬럼 인덱스를 찾아 {센서명: Acc_X_col} 매핑 반환.
    ※ subject마다 (1)센서 순서가 다르고 (2)일부 접두어가 빈 경우(예: '_IMU9_Acc_X')가 있어,
       빈 접두어 블록은 '빠진 표준센서'로 추론해 채운다.
    각 블록은 Acc_X,Y,Z,Gyro_X,Y,Z,... 순으로 연속 배치되어 있다고 가정.
    """
    accx = []  # [(prefix, col_index), ...] 등장순
    for i, h in enumerate(header_stripped):
        if h.endswith(_ACCX_SUFFIX):
            accx.append((h[: -len(_ACCX_SUFFIX)], i))
    canon = set(CANONICAL_SENSORS)
    present = {p for p, _ in accx if p in canon}
    missing = [c for c in CANONICAL_SENSORS if c not in present]
    prefmap, mi = {}, 0
    for pref, ci in accx:
        name = pref
        if name not in canon:     # 빈 접두어 또는 MAC주소 등 깨진 블록 → 빠진 표준센서로 채움
            if mi < len(missing):
                name = missing[mi]
                mi += 1
            else:
                continue
        prefmap[name] = ci
    return prefmap


def read_imu_csv(path, side):
    """
    laptop1 _u CSV에서 {side}U(위팔=어깨IMU), {side}F(아래팔=손목IMU), CHS(가슴=몸통)의
    Acc/Gyro 6축씩 추출.
    반환: idx(N,), data(N,18)  열순서 = [U_acc, U_gyro, F_acc, F_gyro, CHS_acc, CHS_gyro]
          (U/F는 0:12로 유지 → 기존 관절각 계산 호환. CHS는 12:18에 append.)
          (빈셀/NaN은 그대로 NaN). 컬럼명 손상/순서변동에 robust.
    """
    upper = f"{side}U"
    fore = f"{side}F"
    chest = "CHS"           # 가슴(몸통) 센서 — 보상작용 감지용, side 무관
    with open(path, newline="") as fp:
        reader = csv.reader(fp)
        try:
            header = next(reader)
        except StopIteration:
            return None, None
        header_stripped = [h.strip() for h in header]
        name2i = {h: i for i, h in enumerate(header_stripped)}
        if "timestamp" not in name2i:
            return None, None
        ts_i = name2i["timestamp"]
        blocks = resolve_sensor_blocks(header_stripped)
        if upper not in blocks or fore not in blocks or chest not in blocks:
            return None, None
        # 각 블록: Acc_X..Z, Gyro_X..Z = Acc_X 인덱스 +0..+5
        col_i = []
        for seg in (upper, fore, chest):
            a = blocks[seg]
            col_i += [a, a + 1, a + 2, a + 3, a + 4, a + 5]
        ncol = len(header_stripped)
        if max(col_i) >= ncol:
            return None, None
        idx, rows = [], []
        for r in reader:
            if not r:
                continue
            try:
                idx.append(float(r[ts_i]))
            except (ValueError, IndexError):
                continue
            vals = []
            for ci in col_i:
                v = r[ci].strip() if ci < len(r) else ""
                vals.append(float(v) if v not in ("", "nan", "NaN") else np.nan)
            rows.append(vals)
    if not rows:
        return None, None
    return np.asarray(idx), np.asarray(rows, dtype=float)


FNAME_RE = re.compile(r"s(\d+)_a(\d+)(?:\(([a-z]+)\))?_t(\d+)_u", re.IGNORECASE)


def parse_meta(fname):
    """파일명 → (activity, variant, trial). 끝 공백/확장자 누락 모두 허용."""
    m = FNAME_RE.search(fname)
    if not m:
        return None
    _, act, variant, trial = m.groups()
    act_label = f"a{act}" + (f"({variant})" if variant else "")
    return act_label, int(trial)


# ============================================================================
# trial 1개 처리
# ============================================================================
def process_trial(path, side, cfg):
    idx, raw = read_imu_csv(path, side)
    if raw is None or len(raw) < 5:
        return None, "read_fail"

    # 1) 결측치 보간(+ 큰 갭은 NaN 유지)
    for c in range(raw.shape[1]):
        raw[:, c] = interp_nan_1d(raw[:, c], cfg["interp_limit"])
    # 양끝/큰갭 NaN 행 제거 — 내부 큰갭이 남아있으면 가장 긴 연속 유효구간만 사용
    good = ~np.isnan(raw).any(axis=1)
    if good.sum() < 5:
        return None, "too_many_nan"
    # 가장 긴 연속 True 구간 선택
    s, best = 0, (0, 0)
    while s < len(good):
        if good[s]:
            e = s
            while e < len(good) and good[e]:
                e += 1
            if (e - s) > (best[1] - best[0]):
                best = (s, e)
            s = e
        else:
            s += 1
    a, b = best
    raw = raw[a:b]
    idx = idx[a:b]
    if len(raw) < 5:
        return None, "segment_short"

    # 2) 시간축 + 리샘플
    t_src = (idx - idx[0]) / cfg["source_hz"]
    if abs(cfg["source_hz"] - cfg["target_hz"]) < 1e-6:
        t, X = t_src, raw
    else:
        t, X = resample_uniform(t_src, raw, cfg["target_hz"])
    if X is None or len(X) < cfg["min_samples"]:
        return None, "too_short_after_resample"

    # 3) feature = 18 raw IMU(U6+F6+CHS6). 관절각은 산출 안 함(동작해석기 담당).
    return dict(t=t, feats=X), "ok"


FEATURE_NAMES = [
    "U_acc_x", "U_acc_y", "U_acc_z", "U_gyro_x", "U_gyro_y", "U_gyro_z",
    "F_acc_x", "F_acc_y", "F_acc_z", "F_gyro_x", "F_gyro_y", "F_gyro_z",
    "CHS_acc_x", "CHS_acc_y", "CHS_acc_z", "CHS_gyro_x", "CHS_gyro_y", "CHS_gyro_z",
]


# ============================================================================
# 메인
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v, type=type(v))
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="처리할 subject 폴더명(예: subject1 subject2). 미지정=전체")
    ap.add_argument("--plot-check", action="store_true",
                    help="첫 trial 채널 요약 통계 출력")
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in DEFAULTS}
    data_root = cfg["data_root"]
    out_root = cfg["out_root"]
    os.makedirs(out_root, exist_ok=True)

    print("=" * 70)
    print("IMU 전처리 (CHS+RU+RF 18채널 정제)")
    print(f"  side={cfg['side']}  source_hz={cfg['source_hz']}  target_hz={cfg['target_hz']}")
    if abs(cfg["source_hz"] - cfg["target_hz"]) > 1e-6:
        print(f"  ⚠️  원본 {cfg['source_hz']}Hz 가정 → {cfg['target_hz']}Hz 리샘플. source-hz 값을 데이터셋 문서로 꼭 확인!")
    print("=" * 70)

    subj_dirs = (sorted(args.subjects) if args.subjects
                 else sorted(glob.glob(os.path.join(data_root, "subject*"))))
    if args.subjects:
        subj_dirs = [os.path.join(data_root, s) if not s.startswith(data_root) else s
                     for s in args.subjects]

    m_subj, m_act = [], []
    stats = dict(trials=0, ok=0, skipped=0)
    skip_reasons = {}

    for sd in subj_dirs:
        subj = os.path.basename(sd.rstrip("/"))
        imu_dir = os.path.join(sd, "laptop1", "IMU9")
        if not os.path.isdir(imu_dir):
            continue
        files = sorted(os.listdir(imu_dir))
        out_subj = os.path.join(out_root, subj)
        os.makedirs(out_subj, exist_ok=True)
        for fn in files:
            meta = parse_meta(fn)
            if meta is None:
                continue
            act_label, trial = meta
            stats["trials"] += 1
            path = os.path.join(imu_dir, fn)
            res, status = process_trial(path, cfg["side"], cfg)
            if res is None:
                stats["skipped"] += 1
                skip_reasons[status] = skip_reasons.get(status, 0) + 1
                continue
            stats["ok"] += 1
            m_subj.append(subj); m_act.append(act_label)

            # trial별 처리 결과 CSV 저장 (time + 18 IMU 채널)
            out_csv = os.path.join(out_subj, f"{subj}_{act_label}_t{trial}.csv")
            arr = np.column_stack([res["t"], res["feats"]])
            np.savetxt(out_csv, arr, delimiter=",",
                       header="time," + ",".join(FEATURE_NAMES), comments="")

            if args.plot_check and stats["ok"] == 1:
                f = res["feats"]
                print(f"\n[검증] {subj}_{act_label}_t{trial}  (n={len(res['t'])}, {res['t'][-1]:.2f}s)")
                for j, nm in enumerate(FEATURE_NAMES):
                    print(f"  {nm:12} 평균 {f[:,j].mean():8.3f}  범위 {f[:,j].min():8.2f}~{f[:,j].max():7.2f}")
                print()

    print("\n" + "=" * 70)
    print("완료 요약")
    print(f"  trial 총 {stats['trials']}개 | 성공 {stats['ok']} | 제외 {stats['skipped']}")
    if skip_reasons:
        print(f"  제외 사유: {skip_reasons}")
    print(f"  subject 수={len(set(m_subj))}  활동 종류={sorted(set(m_act))}")
    print(f"  저장: {out_root}/<subject>/*.csv  (time + 18 IMU 채널)")
    print("=" * 70)


if __name__ == "__main__":
    main()
