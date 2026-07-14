#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_tcn_quat.py — [Stage C-2] 쿼터니언 미래예측 TCN 학습/평가 (LOSO)
======================================================================
입력 : data/cache/quat_forecast.npz  (X=(N,60,12) 현재기준 상대쿼터니언,
        Y=Δ쿼터니언(N,12) = 200ms 뒤 회전)
모델 : dilated causal TCN (train_tcn_reg.TCNReg 재사용), 출력 12 = 3센서×(w,x,y,z)
손실 : 쿼터니언 측지 손실  1 − ⟨q_pred, q_true⟩²  (이중덮개 무관, 정규화 출력)
지표 : 측지각(°) = 2·arccos|⟨q_pred,q_true⟩|  (센서별)  + persistence(Δ=단위) 대비 개선율
검증 : LOSO (Leave-One-Subject-Out)

실행 : /home/jiyul/.venv/bin/python train_tcn_quat.py            # 전체 LOSO
       /home/jiyul/.venv/bin/python train_tcn_quat.py --folds 3  # 빠른 확인
       /home/jiyul/.venv/bin/python train_tcn_quat.py --folds 1 --save data/cache/tcn_quat.pt
"""
import argparse, os, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from train_tcn_reg import TCNReg          # 동일 TCN 아키텍처 재사용

torch.set_num_threads(min(16, os.cpu_count() or 8))   # CPU 코어 최대 활용

NPZ = "data/cache/quat_forecast.npz"
CHANNELS, LEVELS, KERNEL, DROPOUT = 32, 3, 3, 0.10
EPOCHS, BATCH, LR, WEIGHT_DEC, SEED = 40, 256, 1e-3, 1e-4, 0
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], np.float32)


def quat_loss(pred, true):
    """pred,true: (B,12). 센서별 정규화 후 1−dot². 미분가능·이중덮개 무관."""
    p = pred.view(-1, 3, 4)
    p = p / p.norm(dim=2, keepdim=True).clamp_min(1e-8)
    t = true.view(-1, 3, 4)
    dot = (p * t).sum(dim=2)               # (B,3)
    return (1.0 - dot ** 2).mean()


def geodesic_deg(pred, true):
    """센서별 측지각(°). pred,true: (N,12) → (N,3)."""
    p = pred.reshape(-1, 3, 4)
    p = p / np.linalg.norm(p, axis=2, keepdims=True).clip(1e-8)
    t = true.reshape(-1, 3, 4)
    dot = np.abs((p * t).sum(2)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot))  # (N,3)


def train_fold(Xtr, Ytr, Xte, Yte, device):
    # 입력은 이미 '현재기준 상대쿼터니언'(부착방향 상쇄)이라 표준화 없이 그대로 사용.
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model = TCNReg(12, 12, CHANNELS, LEVELS, KERNEL, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = quat_loss(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte).to(device)).cpu().numpy()
    ang_model = geodesic_deg(pred, Yte)                                  # (Nte,3)
    ang_pers = geodesic_deg(np.tile(IDENTITY, (len(Yte), 3)), Yte)       # Δ=단위 예측
    return ang_model, ang_pers, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(NPZ, allow_pickle=True)
    X = d["X"].astype(np.float32); Y = d["Y"].astype(np.float32)
    subjects = d["subjects"]; sensors = list(d["sensor_names"])
    uniq = sorted(set(subjects.tolist()), key=lambda s: int(s.replace("subject", "")))
    if args.folds > 0: uniq = uniq[:args.folds]
    n_param = sum(p.numel() for p in TCNReg(12, 12, CHANNELS, LEVELS, KERNEL, DROPOUT).parameters())
    print(f"X={X.shape} Y={Y.shape} | subject {len(set(subjects.tolist()))}명 | LOSO {len(uniq)} fold | device={device}")
    print(f"TCN 파라미터 ≈ {n_param:,}  (입력12→출력12 쿼터니언)\n")
    print(f"{'subject':>10} {'n_te':>5} {'모델°':>7} {'기준°':>7} {'개선%':>6}")
    print("-" * 44)

    m_ang, p_ang, sens_m, sens_p = [], [], [], []
    for s in uniq:
        te = subjects == s; tr = ~te
        am, ap_, _ = train_fold(X[tr], Y[tr], X[te], Y[te], device)
        mo, po = am.mean(), ap_.mean()
        imp = (po - mo) / po * 100
        m_ang.append(mo); p_ang.append(po)
        sens_m.append(am.mean(0)); sens_p.append(ap_.mean(0))
        print(f"{s:>10} {int(te.sum()):>5} {mo:>7.2f} {po:>7.2f} {imp:>6.1f}")

    m_ang, p_ang = np.array(m_ang), np.array(p_ang)
    sens_m, sens_p = np.array(sens_m).mean(0), np.array(sens_p).mean(0)
    print("-" * 44)
    print(f"LOSO 평균  모델 {m_ang.mean():.2f}°±{m_ang.std():.2f}  "
          f"기준(persistence) {p_ang.mean():.2f}°  "
          f"개선 {(p_ang.mean()-m_ang.mean())/p_ang.mean()*100:.1f}%")
    print("\n센서별 측지각 (모델 vs 기준, °):")
    for nm, mo, po in zip(sensors, sens_m, sens_p):
        print(f"  {nm:4} 모델 {mo:5.2f}°   기준 {po:5.2f}°   개선 {(po-mo)/po*100:5.1f}%")

    if args.save:
        _, _, model = train_fold(X, Y, X[:1], Y[:1], device)
        torch.save({"state_dict": model.state_dict(),
                    "cfg": dict(n_in=12, n_out=12, channels=CHANNELS, levels=LEVELS,
                                kernel=KERNEL, dropout=DROPOUT, win=X.shape[1], hor_ms=200),
                    "sensor_names": sensors,
                    "note": "input=rel_to_current quat; output=delta quat; recon: q_fut=norm(pred)*q_cur"},
                   args.save)
        print(f"\n전체 데이터 학습 모델 저장: {args.save}")


if __name__ == "__main__":
    main()
