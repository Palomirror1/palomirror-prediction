#!/usr/bin/env python3
"""
a9 미래-IMU(가속도 6채널, 0.5s 뒤) 예측용 경량 TCN 회귀 학습/평가.

타깃은 Δ(=미래acc − 현재acc). 추론/평가 시 최종예측 = 현재acc + Δ예측이며,
Δ오차가 곧 절대 acc오차라 RMSE는 그대로 물리단위(m/s²)로 해석됨.
persistence 베이스라인 = "Δ=0"(미래도 현재와 같다) → 절대공간 예측은 현재acc 그대로.

입력 : processed/a9_phase_forecast.npz  (X=(N,50,18), Y=Δ(N,6), subjects, ...)
       + per-window 센터링(윈도우별 채널평균 제거)으로 입력의 중력오프셋도 제거 → 방향불변.
모델 : dilated causal 1D-CNN (TCN) 회귀  — 출력 6, MSE
검증 : LOSO (Leave-One-Subject-Out)
지표 : RMSE / MAE (물리단위 m/s²) + persistence 베이스라인 대비 개선율

실행 : /home/jiyul/.venv/bin/python train_tcn_reg.py            # 전체 LOSO
       /home/jiyul/.venv/bin/python train_tcn_reg.py --folds 3  # 빠른 확인
"""
import os, argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

NPZ        = "data/cache/processed/a9_phase_forecast.npz"   # 18채널(U+F+CHS) 입력, phase 라벨 포함
CHANNELS   = 32
LEVELS     = 3        # dilation 1,2,4
KERNEL     = 3
DROPOUT    = 0.10
EPOCHS     = 40
BATCH      = 256
LR         = 1e-3
WEIGHT_DEC = 1e-4
SEED       = 0


class Chomp1d(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return x[:, :, :-self.c].contiguous() if self.c else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, k, padding=pad, dilation=dilation), Chomp1d(pad),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_out, n_out, k, padding=pad, dilation=dilation), Chomp1d(pad),
            nn.ReLU(), nn.Dropout(dropout))
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()
    def forward(self, x):
        res = x if self.down is None else self.down(x)
        return self.relu(self.net(x) + res)


class TCNReg(nn.Module):
    def __init__(self, n_in, n_out, channels, levels, k, dropout):
        super().__init__()
        layers, c = [], n_in
        for i in range(levels):
            layers.append(TemporalBlock(c, channels, k, 2 ** i, dropout)); c = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, n_out)
    def forward(self, x):                 # (B,T,F)
        h = self.tcn(x.transpose(1, 2))   # (B,C,T)
        return self.head(h[:, :, -1])     # (B,n_out)


def rmse_mae(pred, true):
    err = pred - true
    rmse = np.sqrt((err ** 2).mean())
    mae = np.abs(err).mean()
    ch_rmse = np.sqrt((err ** 2).mean(0))   # 채널별
    return rmse, mae, ch_rmse


def train_fold(Xtr, Ytr, Xte, Yte, n_in, n_out, device):
    # per-window 센터링: 각 윈도우를 자기 채널평균으로 빼서 중력 DC오프셋 제거
    #   → 센서 부착방향 차이가 입력에서 사라짐(이상 fold 방지). 윈도우별 독립이라 leakage 없음.
    Xtr = Xtr - Xtr.mean(axis=1, keepdims=True)
    Xte = Xte - Xte.mean(axis=1, keepdims=True)
    # 입력/타깃 표준화 (train 통계만)
    mx = Xtr.reshape(-1, n_in).mean(0); sx = Xtr.reshape(-1, n_in).std(0) + 1e-6
    my = Ytr.mean(0);                   sy = Ytr.std(0) + 1e-6
    Xtr_ = (Xtr - mx) / sx; Xte_ = (Xte - mx) / sx
    Ytr_ = (Ytr - my) / sy

    ds = TensorDataset(torch.tensor(Xtr_), torch.tensor(Ytr_))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model = TCNReg(n_in, n_out, CHANNELS, LEVELS, KERNEL, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)
    lossfn = nn.MSELoss()
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = lossfn(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred_ = model(torch.tensor(Xte_).to(device)).cpu().numpy()
    pred = pred_ * sy + my                       # Δ 예측 (물리단위 역변환)
    # 모델 RMSE: Δ예측 vs Δ정답 = (현재+Δ예측) vs (현재+Δ정답) 의 절대 acc오차와 동일.
    # persistence: Δ=0 예측 → Δ공간에서 영벡터
    pers = np.zeros_like(Yte)
    return rmse_mae(pred, Yte), rmse_mae(pers, Yte), model, (mx, sx, my, sy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(NPZ, allow_pickle=True)
    X = d["X"].astype(np.float32); Y = d["Y"].astype(np.float32)
    subjects = d["subjects"]; out_names = list(d["out_names"])
    n_in, n_out = X.shape[2], Y.shape[1]
    uniq = sorted(set(subjects.tolist()), key=lambda s: int(s.replace("subject", "")))
    if args.folds > 0: uniq = uniq[:args.folds]
    n_param = sum(p.numel() for p in TCNReg(n_in, n_out, CHANNELS, LEVELS, KERNEL, DROPOUT).parameters())
    print(f"X={X.shape} Y={Y.shape} | subject {len(set(subjects.tolist()))}명 | LOSO {len(uniq)} fold | device={device}")
    print(f"TCN 파라미터 ≈ {n_param:,}\n")
    print(f"{'subject':>10} {'n_te':>5} {'모델RMSE':>8} {'기준RMSE':>8} {'개선%':>6} {'모델MAE':>8}")
    print("-" * 56)

    m_rmse, m_mae, p_rmse, ch_acc = [], [], [], []
    for s in uniq:
        te = subjects == s; tr = ~te
        (mr, mm, mch), (pr, pm, pch), _, _ = train_fold(X[tr], Y[tr], X[te], Y[te], n_in, n_out, device)
        imp = (pr - mr) / pr * 100
        m_rmse.append(mr); m_mae.append(mm); p_rmse.append(pr); ch_acc.append(mch)
        print(f"{s:>10} {int(te.sum()):>5} {mr:>8.3f} {pr:>8.3f} {imp:>6.1f} {mm:>8.3f}")

    m_rmse, m_mae, p_rmse = map(np.array, (m_rmse, m_mae, p_rmse))
    ch_acc = np.array(ch_acc).mean(0)
    print("-" * 56)
    print(f"LOSO 평균  모델 RMSE={m_rmse.mean():.3f}±{m_rmse.std():.3f}  "
          f"기준(persistence) RMSE={p_rmse.mean():.3f}  "
          f"개선={ (p_rmse.mean()-m_rmse.mean())/p_rmse.mean()*100:.1f}%")
    print(f"LOSO 평균  모델 MAE ={m_mae.mean():.3f}±{m_mae.std():.3f} m/s²")
    print(f"\n채널별 모델 RMSE (m/s²):")
    for nm, v in zip(out_names, ch_acc):
        print(f"  {nm:10} {v:.3f}")

    if args.save:
        # 전체 데이터로 재학습 후 배포용 저장
        res = train_fold(X, Y, X[:1], Y[:1], n_in, n_out, device)
        model, (mx, sx, my, sy) = res[2], res[3]
        torch.save({"state_dict": model.state_dict(),
                    "mx": mx, "sx": sx, "my": my, "sy": sy,
                    "cfg": dict(n_in=n_in, n_out=n_out, channels=CHANNELS,
                                levels=LEVELS, kernel=KERNEL, dropout=DROPOUT),
                    "out_names": out_names}, args.save)
        print(f"\n전체 데이터 학습 모델 저장: {args.save}")


if __name__ == "__main__":
    main()
