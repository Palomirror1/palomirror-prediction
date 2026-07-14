#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_quat_forecast.py — [Stage C-1] 쿼터니언 미래예측 데이터셋 빌더
====================================================================
imu_to_quaternion.py 가 만든 12채널 쿼터니언 시계열에서 슬라이딩 윈도우와
**200ms 뒤 Δ쿼터니언 타깃**을 생성한다.

  구조: [과거 1.0s 쿼터니언(60×12)] → TCN → [200ms 뒤 Δ쿼터니언(12)] → (현재⊗Δ) → 역기하학

핵심 설계 (센서 부착방향 차이에 강건 — 가속도판 'Δ+센터링'의 쿼터니언 버전):
  - 입력: 윈도우 내 모든 프레임을 **현재(마지막)프레임 기준 상대회전**으로 변환
          q'_i = q_i ⊗ conj(q_cur)  → 현재프레임=단위(identity), 절대 부착방향 상쇄.
  - 타깃: Δ = q_future ⊗ conj(q_cur)  (곱셈 기반 상대회전, 입력과 같은 기준).
  - 추론 시 최종예측: q_future = Δ예측 ⊗ q_cur.
  - persistence 베이스라인 = "Δ=단위"(200ms 뒤도 지금과 같은 방향) → 오차=실제 회전각.

출력 : data/cache/quat_forecast.npz  (X=(N,60,12), Y=Δ(N,12), subjects, trials, ...)
"""
import os, glob, csv
import numpy as np

QUAT_ROOT = "data/cache/quat"
OUT = "data/cache/quat_forecast.npz"
FS = 60.0
WIN = 60          # 입력 1.0s
HOR = 12          # 예측 200ms (@60Hz)
STRIDE = 6
SENSOR_NAMES = ["CHS", "RU", "RF"]     # 각 4채널(w,x,y,z)


def qmul(a, b):
    """Hamilton 곱 a⊗b. a,b: (...,4)"""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def sign_fix(q):
    """이중덮개(q=−q) 해소: w<0 이면 부호반전 (배열 마지막축이 쿼터니언)."""
    s = np.where(q[..., :1] < 0, -1.0, 1.0)
    return q * s


def load_quat(fn):
    with open(fn) as f:
        r = csv.reader(f); next(r)
        rows = [[float(v) for v in row] for row in r if row]
    a = np.asarray(rows, float)
    return a[:, 1:]        # time 제외 → (N,12)


def subject_of(fn): return os.path.basename(os.path.dirname(fn))
def trial_of(fn):   return os.path.basename(fn)[:-4]


def main():
    files = sorted(glob.glob(os.path.join(QUAT_ROOT, "*", "*.csv")))
    print(f"쿼터니언 trial 파일: {len(files)}개   WIN={WIN}({WIN/FS:.1f}s) HOR={HOR}({HOR/FS*1000:.0f}ms) STRIDE={STRIDE}")
    Xs, Ys, subj, trs = [], [], [], []
    n_ok = n_short = 0
    for fn in files:
        Q = load_quat(fn)                      # (N,12)
        if len(Q) < WIN + HOR + 1:
            n_short += 1; continue
        Qs = Q.reshape(len(Q), 3, 4)           # (N,3센서,4)
        for t in range(WIN - 1, len(Q) - HOR, STRIDE):
            qcur = Qs[t]                        # (3,4) 현재 방향
            cj = qconj(qcur)                    # (3,4)
            win = Qs[t - WIN + 1:t + 1]         # (WIN,3,4)
            rel = qmul(win, cj[None])          # (WIN,3,4) 현재기준 상대회전
            rel = sign_fix(rel)
            fut = Qs[t + HOR]                   # (3,4) 200ms 뒤
            delta = sign_fix(qmul(fut, cj))    # (3,4) Δ 타깃
            Xs.append(rel.reshape(WIN, 12))
            Ys.append(delta.reshape(12))
            subj.append(subject_of(fn)); trs.append(trial_of(fn))
        n_ok += 1

    X = np.asarray(Xs, np.float32); Y = np.asarray(Ys, np.float32)
    subjects = np.asarray(subj); trials = np.asarray(trs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, Y=Y, subjects=subjects, trials=trials,
                        sensor_names=np.asarray(SENSOR_NAMES),
                        config=np.asarray([f"win={WIN}", f"hor={HOR}", f"stride={STRIDE}",
                                           f"fs={FS}", "target=delta_quat(fut*conj(cur))",
                                           "input=rel_to_current"]))

    # persistence(Δ=단위) 회전각 = 실제 200ms 회전량 (센서별, deg)
    w = np.clip(np.abs(Y.reshape(-1, 3, 4)[..., 0]), 0, 1)
    ang = np.degrees(2 * np.arccos(w))         # (N,3)
    print("=== 완료 ===")
    print(f"  사용 trial {n_ok}  (짧아서 제외 {n_short})")
    print(f"  윈도우 {len(Y)}개  subject {len(set(subjects.tolist()))}명")
    print(f"  X={X.shape}  Y={Y.shape}")
    print(f"  저장: {OUT}")
    print("\n=== 실제 200ms 회전량 (persistence 베이스라인 = 이 각도, deg) ===")
    for i, nm in enumerate(SENSOR_NAMES):
        print(f"  {nm:4} 평균 {ang[:, i].mean():5.2f}°  중앙값 {np.median(ang[:, i]):5.2f}°  95%tile {np.percentile(ang[:,i],95):5.2f}°")
    print(f"  전체 평균 {ang.mean():.2f}°")


if __name__ == "__main__":
    main()
