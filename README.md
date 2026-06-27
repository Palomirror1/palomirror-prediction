# palomirror-prediction

상지재활 서비스용 **경량 미래-IMU 예측 모델**. 3채널 IMU(가슴·오른상완·오른손목)로
0.5초 뒤 팔 가속도를 예측해 재활 로봇팔 보조제어와 보상작용 감지에 활용한다.

```
[과거 1.0s IMU] → 경량 TCN → [0.5s 뒤 가속도 Δ] → (현재+Δ) → 동작해석기 → 팔꿈치동작
```

타깃이 IMU 신호 자체에서 나오므로 사람 라벨이 필요 없다 (self-supervised).

## 파이프라인

| 단계 | 스크립트 | 입력 → 출력 |
|---|---|---|
| 1. 전처리 | `preprocess_imu.py` | 원본 IMU → `processed/<subject>/*.csv` (18채널 정제) |
| 2. 데이터셋 빌드 | `make_a9_phase_forecast.py` | CSV → `processed/a9_phase_forecast.npz` (윈도우·Δ타깃·phase) |
| 3. 학습/평가 | `train_tcn_reg.py` | npz → LOSO 평가 + 배포모델 `tcn_a9_forecast.pt` |

```bash
python preprocess_imu.py --source-hz 100
python make_a9_phase_forecast.py
python train_tcn_reg.py                       # 전체 LOSO 평가
python train_tcn_reg.py --folds 1 --save tcn_a9_forecast.pt   # 배포모델 저장
```

## 입력 / 출력

- **입력 18채널**: U(상완)·F(전완)·CHS(가슴) 각 Acc 3 + Gyro 3. 가슴은 몸통 보상작용 감지용(입력 전용).
- **출력 6채널**: 팔(U+F) 가속도의 **Δ(=0.5s뒤 − 현재)**. 추론 시 최종예측 = 현재 acc + Δ.

### 방향 불변 설계 (센서 부착방향 차이에 강건)
- **출력 Δ 타깃**: 윈도우 내 중력 성분이 빼기에서 상쇄.
- **입력 per-window 센터링**: 각 윈도우를 자기 채널평균으로 빼 중력 오프셋 제거.
- → 두 처리로 입력·출력 모두 방향 불변. LOSO에서 분포이동으로 망가지던 이상 fold 해소.

## 성능 (LOSO, 29-fold)

| 지표 | 값 |
|---|---|
| 모델 RMSE | **2.183 ± 0.332 m/s²** |
| persistence 기준 RMSE | 2.577 m/s² |
| 평균 개선율 | **15.3%** |
| 모델 | TCN, 파라미터 ≈ 18,086 (CPU 실시간 가능) |

## 배포 모델 추론 순서 (`tcn_a9_forecast.pt`)

1. 과거 1초 IMU 18채널 수집 (50×18)
2. per-window 센터링: `X -= X.mean(시간축)`
3. 표준화: `X = (X - mx) / sx`
4. 모델 통과 → `Δ_norm`
5. 역표준화: `Δ = Δ_norm * sy + my`
6. 최종 예측: `미래 팔가속도 = 현재 팔가속도 + Δ`
