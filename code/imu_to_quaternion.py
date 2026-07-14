#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imu_to_quaternion.py  — [Stage B] 원신호(Acc+Gyro) → 부위별 방향 쿼터니언 12채널
====================================================================================
StrengthSense 데이터셋엔 쿼터니언이 없다(Acc/Gyro/Magn 원신호만). 반면 실기기
Xsens DOT 는 쿼터니언(w,x,y,z)을 직접 출력하므로, **학습입력을 실기기와 같은
형식으로 맞추기 위해** AHRS(Madgwick 6축)로 Acc+Gyro 를 융합해 각 IMU 의 방향
쿼터니언을 산출한다. (자기장 Magn 은 실내 간섭이 커 제외 → 6축 IMU 융합)

  구조: [원신호 Acc+Gyro 3센서] → Madgwick → [쿼터니언 12채널] → (Stage C) TCN 예측

입력 : data/raw/final IMU dataset/<subject>/laptop1/IMU9/*_a9_*_u.csv (원신호)
출력 : data/cache/quat/<subject>/<subject>_a9_t{K}.csv
        컬럼: time, CHS_qw,CHS_qx,CHS_qy,CHS_qz, RU_qw..RU_qz, RF_qw..RF_qz

주의:
  - 표본레이트 FS=60Hz (StrengthSense/Xsens 기준). timestamp 는 샘플인덱스라
    데이터로 확정 불가 → 실기기와 동일한 60Hz 로 통일(시간축 200ms=12샘플 일치).
  - Gyro 단위 deg/s → rad/s 변환(×π/180). Acc 는 m/s²(정규화해 사용).
"""
import os, csv, glob, math
import numpy as np

# 기존 파이프라인의 견고한 CSV 파서 재사용 (컬럼순서/결측/센서명 손상에 robust)
from preprocess_imu import read_imu_csv, parse_meta, interp_nan_1d

RAW_ROOT = "data/raw/final IMU dataset"
OUT_ROOT = "data/cache/quat"
ACT = "a9"
SIDE = "R"                 # RU(오른상완)+RF(오른전완) + CHS(가슴)
FS = 60.0                  # Hz  (Xsens DOT 실기기와 통일)
DT = 1.0 / FS
DEG2RAD = math.pi / 180.0
BETA = 0.08                # Madgwick 수렴 게인 (클수록 가속도 보정 강함)

# read_imu_csv 반환 18채널 순서: [RU(acc3,gyro3), RF(acc3,gyro3), CHS(acc3,gyro3)]
#   acc 시작열: RU=0, RF=6, CHS=12  (gyro = +3)
# 출력 쿼터니언 순서는 CHS, RU, RF (환자 흐름도 표기순).
SENSORS = [("CHS", 12), ("RU", 0), ("RF", 6)]
OUT_COLS = [f"{name}_q{c}" for name, _ in SENSORS for c in ("w", "x", "y", "z")]


def init_q_from_acc(a):
    """첫 가속도(중력)로 roll/pitch 초기 쿼터니언 추정(yaw=0) → 수렴 과도구간 단축."""
    n = np.linalg.norm(a)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax, ay, az = a / n
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])   # yaw=0


def madgwick_imu(gyro, acc, dt=DT, beta=BETA, q0=None):
    """Madgwick 6축(IMU) 융합. gyro(N,3) rad/s, acc(N,3) 임의단위(내부 정규화).
    반환: Q(N,4) 단위쿼터니언 [w,x,y,z]."""
    q = init_q_from_acc(acc[0]) if q0 is None else np.array(q0, float)
    q = q / np.linalg.norm(q)
    Q = np.empty((len(gyro), 4))
    for i in range(len(gyro)):
        gx, gy, gz = gyro[i]
        ax, ay, az = acc[i]
        qw, qx, qy, qz = q
        # 자이로에 의한 쿼터니언 변화율
        qDot = 0.5 * np.array([
            -qx * gx - qy * gy - qz * gz,
             qw * gx + qy * gz - qz * gy,
             qw * gy - qx * gz + qz * gx,
             qw * gz + qx * gy - qy * gx])
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n > 1e-9:
            ax, ay, az = ax / n, ay / n, az / n
            # 중력방향 오차 f 와 야코비안 J → 경사하강 보정
            f = np.array([2 * (qx * qz - qw * qy) - ax,
                          2 * (qw * qx + qy * qz) - ay,
                          2 * (0.5 - qx * qx - qy * qy) - az])
            J = np.array([[-2 * qy,  2 * qz, -2 * qw,  2 * qx],
                          [ 2 * qx,  2 * qw,  2 * qz,  2 * qy],
                          [ 0.0,    -4 * qx, -4 * qy,  0.0]])
            grad = J.T @ f
            gn = np.linalg.norm(grad)
            if gn > 1e-9:
                qDot -= beta * (grad / gn)
        q = q + qDot * dt
        q = q / np.linalg.norm(q)
        if q[0] < 0:                      # 부호 일관성(−q=q) → w≥0 로 고정
            q = -q
        Q[i] = q
    return Q


def longest_valid(data):
    """열별 결측 보간 후 최장 연속 유효구간만 반환(전처리와 동일 정책)."""
    d = data.copy()
    for c in range(d.shape[1]):
        d[:, c] = interp_nan_1d(d[:, c], 20)
    good = ~np.isnan(d).any(axis=1)
    if good.sum() < 5:
        return None
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
    return d[a:b]


def process(path):
    idx, raw = read_imu_csv(path, SIDE)     # (N,), (N,18)  RU,RF,CHS 순
    if raw is None or len(raw) < FS:        # 1초 미만 스킵
        return None
    raw = longest_valid(raw)
    if raw is None or len(raw) < FS:
        return None
    quats = []
    for _, a0 in SENSORS:
        acc = raw[:, a0:a0 + 3]
        gyro = raw[:, a0 + 3:a0 + 6] * DEG2RAD      # deg/s → rad/s
        quats.append(madgwick_imu(gyro, acc))        # (N,4)
    Q = np.concatenate(quats, axis=1)                # (N,12) CHS,RU,RF
    t = np.arange(len(Q)) / FS
    return t, Q


def main():
    files = sorted(glob.glob(os.path.join(RAW_ROOT, "*", "laptop1", "IMU9", f"*_{ACT}_*_u.csv")))
    print(f"{ACT} 원신호 파일: {len(files)}개   FS={FS:.0f}Hz  beta={BETA}")
    os.makedirs(OUT_ROOT, exist_ok=True)
    n_ok = n_skip = 0
    norms = []
    for path in files:
        meta = parse_meta(os.path.basename(path))
        if meta is None:
            n_skip += 1; continue
        act_label, trial = meta
        subj = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        res = process(path)
        if res is None:
            n_skip += 1; continue
        t, Q = res
        norms.append(np.linalg.norm(Q.reshape(-1, 4), axis=1).mean())
        out_dir = os.path.join(OUT_ROOT, subj)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{subj}_{act_label}_t{trial}.csv")
        arr = np.column_stack([t, Q])
        np.savetxt(out_csv, arr, delimiter=",",
                   header="time," + ",".join(OUT_COLS), comments="")
        n_ok += 1
    print(f"완료: 성공 {n_ok}  스킵 {n_skip}")
    if norms:
        print(f"쿼터니언 평균 노름(=1이어야 정상): {np.mean(norms):.6f}")
    print(f"저장: {OUT_ROOT}/<subject>/*.csv  (time + 12채널 쿼터니언)")


if __name__ == "__main__":
    main()
