#!/usr/bin/env python3
"""
a9(#9 물병 reach→grasp→pour→복귀) 미래-IMU 예측 데이터셋 빌더 — '표준 동작 구조' phase 라벨 부착판.

표준 동작 템플릿 (총 8초 기준 비율 2:2:3:1):
    정지(rest) 2 → 뻗기+잡기(reach+grasp) 2 → 따르기(pour) 3 → 복귀(return) 1
  * drink(마시기) 구간은 제외 (DRINK_TAIL_FRAC 만큼 끝꼬리를 잘라 버림).

핵심 설계:
  - 신호값은 리샘플링하지 않음 → 가속도 물리단위(m/s²) 왜곡 방지, 원본 timing 유지.
  - 각 trial을 '상대 위치 비율'로 4구간 라벨링 (길이 제각각이어도 동일 비율 적용).
  - 각 forecasting 윈도우에 phase 라벨을 붙여 저장 → 구간별 평가 / 동작해석기 phase 입력용.

구조: [과거 1.0s IMU] -> 모델 -> [0.5s 뒤 가속도 Δ] -> (현재+Δ) -> 동작해석기 -> 팔꿈치동작
타깃이 IMU 신호 자체에서 나와 사람 라벨 불필요 (self-supervised).
  * 타깃 = Δ(미래acc − 현재acc): 중력 오프셋 상쇄로 센서 부착방향 차이에 강건.

출력 : processed/a9_phase_forecast.npz
       (X, Y, phase, subjects, trials, phase_names, in_names, out_names, config)
"""
import os, glob, csv, numpy as np

PROC = "data/cache/processed"
ACT = "a9"
WIN = 50          # 입력 윈도우 (1.0s @50Hz)
HOR = 25          # 예측 지평 (0.5s @50Hz)
STRIDE = 5
SMOOTH = 5        # 가속도 평활화 이동평균 창 (샘플). 0이면 비활성
TARGET_HZ = 50.0

# --- 표준 동작 구조 (정지:reach+grasp:pour:return = 2:2:3:1) ---
PHASE_NAMES = ["rest", "reach_grasp", "pour", "return"]
PHASE_PROPS = np.array([2, 2, 3, 1], float)        # 비율
# 비율 → 누적 컷포인트 (0~1): [0.25, 0.50, 0.875]
PHASE_CUTS = np.cumsum(PHASE_PROPS / PHASE_PROPS.sum())[:-1]
# drink+복귀 끝꼬리: 신호로 검출 불가 → 끝 일정 비율을 잘라 버림 (유일한 가정값, 조절 가능)
DRINK_TAIL_FRAC = 0.15

# 입력 18채널: U(상완)+F(전완)+CHS(가슴) 의 acc+gyro. CHS는 몸통 보상작용 감지용.
IN_NAMES  = ["U_acc_x","U_acc_y","U_acc_z","U_gyro_x","U_gyro_y","U_gyro_z",
             "F_acc_x","F_acc_y","F_acc_z","F_gyro_x","F_gyro_y","F_gyro_z",
             "CHS_acc_x","CHS_acc_y","CHS_acc_z","CHS_gyro_x","CHS_gyro_y","CHS_gyro_z"]
# 예측 타깃 6채널: 팔(U+F) 가속도의 Δ(=0.5s뒤 − 현재). 가슴은 입력 전용(예측 대상 아님).
#   Δ 타깃 이유: 중력 DC오프셋이 빼기에서 상쇄 → 센서 부착방향 차이에 강건(이상 fold 방지).
#   추론 시 최종 예측 = 현재 acc + Δ예측.
OUT_NAMES = ["U_acc_x","U_acc_y","U_acc_z","F_acc_x","F_acc_y","F_acc_z"]
OUT_IDX = [IN_NAMES.index(n) for n in OUT_NAMES]


def moving_avg(a, w):
    if w <= 1:
        return a
    k = np.ones(w) / w
    pad = w // 2
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        ext = np.concatenate([np.full(pad, a[0, j]), a[:, j], np.full(pad, a[-1, j])])
        out[:, j] = np.convolve(ext, k, mode="same")[pad:pad + len(a)]
    return out


def load_csv(fn):
    with open(fn) as f:
        r = csv.reader(f); hdr = next(r)
        idx = {n: i for i, n in enumerate(hdr)}
        if not all(n in idx for n in IN_NAMES):
            return None
        ii = [idx[n] for n in IN_NAMES]
        rows = []
        for row in r:
            try:
                rows.append([float(row[i]) for i in ii])
            except (ValueError, IndexError):
                continue
    return np.asarray(rows, float)


def subject_of(fn): return os.path.basename(os.path.dirname(fn))
def trial_of(fn):   return os.path.basename(fn)[:-4]


def phase_of(t, motion_end):
    """현재 시점 t 의 phase 인덱스 (0~3). t/motion_end 상대위치를 컷포인트로 분류."""
    rel = t / motion_end
    return int(np.searchsorted(PHASE_CUTS, rel, side="right"))


def main():
    files = sorted(glob.glob(os.path.join(PROC, "*", f"*_{ACT}_*.csv")))
    print(f"{ACT} trial 파일: {len(files)}개")
    print(f"phase 컷포인트(상대): {np.round(PHASE_CUTS,3)}  drink 끝꼬리 제외: {DRINK_TAIL_FRAC*100:.0f}%\n")
    Xs, Ys, phs, subj, trs = [], [], [], [], []
    pers_err = []   # persistence 베이스라인: Y_hat = 현재 acc
    n_ok = n_short = n_drop_drink = 0
    for fn in files:
        d = load_csv(fn)
        if d is None or len(d) < WIN + HOR + 1:
            n_short += 1; continue
        acc_sm = d.copy()
        acc_sm[:, OUT_IDX] = moving_avg(d[:, OUT_IDX], SMOOTH)   # 가속도만 평활화
        n = len(d)
        motion_end = int(round(n * (1 - DRINK_TAIL_FRAC)))       # drink+복귀 끝꼬리 경계
        for t in range(WIN - 1, n - HOR, STRIDE):
            if t >= motion_end:                                  # drink 구간 → 제외
                n_drop_drink += 1; continue
            Xs.append(acc_sm[t - WIN + 1:t + 1, :])              # (50,18) 입력
            cur = acc_sm[t, OUT_IDX]                              # (6,) 현재 acc
            fut = acc_sm[t + HOR, OUT_IDX]                        # (6,) 0.5s 뒤 acc
            y = fut - cur                                         # (6,) Δ 타깃
            Ys.append(y)
            phs.append(phase_of(t, motion_end))                  # 현재 시점 phase
            pers_err.append(-y)                                  # persistence(Δ=0) 오차 = 0 − Δ
            subj.append(subject_of(fn)); trs.append(trial_of(fn))
        n_ok += 1

    X = np.asarray(Xs, np.float32); Y = np.asarray(Ys, np.float32)
    phase = np.asarray(phs, np.int64)
    subjects = np.asarray(subj); trials = np.asarray(trs)
    pers_err = np.asarray(pers_err)

    out = os.path.join(PROC, "a9_phase_forecast.npz")
    np.savez_compressed(out, X=X, Y=Y, phase=phase, subjects=subjects, trials=trials,
                        phase_names=np.asarray(PHASE_NAMES),
                        in_names=np.asarray(IN_NAMES), out_names=np.asarray(OUT_NAMES),
                        config=np.asarray([f"win={WIN}", f"hor={HOR}", f"stride={STRIDE}",
                                           f"smooth={SMOOTH}", f"hz={TARGET_HZ}",
                                           f"phase_props=2:2:3:1", f"drink_tail={DRINK_TAIL_FRAC}",
                                           "target=delta(future-current)"]))

    n_subj = len(set(subjects.tolist()))
    print("=== 완료 요약 ===")
    print(f"  사용 trial {n_ok}개 (너무 짧아 제외 {n_short})")
    print(f"  윈도우 총 {len(Y)}개   subject {n_subj}명   (drink로 제외된 윈도우 {n_drop_drink}개)")
    print(f"  X shape={X.shape}  Y shape={Y.shape}  phase shape={phase.shape}")
    print(f"  저장: {out}\n")

    print("=== phase별 윈도우 분포 ===")
    for i, nm in enumerate(PHASE_NAMES):
        m = phase == i
        cnt = int(m.sum())
        prmse = np.sqrt((pers_err[m] ** 2).mean()) if cnt else 0.0
        print(f"  {i} {nm:12} {cnt:6d}개 ({cnt/len(phase)*100:5.1f}%)  persistence RMSE={prmse:.3f}")

    rmse = np.sqrt((pers_err ** 2).mean())
    mae = np.abs(pers_err).mean()
    print("\n=== persistence 베이스라인 (현재 acc → 0.5s 뒤 예측) ===")
    print(f"  전체 RMSE = {rmse:.4f} m/s²   MAE = {mae:.4f} m/s²")


if __name__ == "__main__":
    main()
