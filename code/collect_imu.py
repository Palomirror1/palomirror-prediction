#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_imu.py
==============
실제 실험용 IMU 3채널 수집기 — 센서 3개를 받아 한 trial을 원본 CSV로 저장한다.
저장 형식은 preprocess_imu.py가 읽는 `final IMU dataset` 원본 포맷과 동일하므로,
수집 직후 곧바로 같은 파이프라인(preprocess → make_a9 → train)을 태울 수 있다.

센서 3개 (오른팔 기준):
  - CHS (Chest,           가슴)   → 몸통 보상작용 감지
  - RU  (Right Upper arm, 위팔)   → 어깨 IMU
  - RF  (Right Forearm,   아래팔) → 손목 IMU
  각 센서 Acc(3) + Gyro(3) = 6축씩, 총 18채널. (자력계 Magn은 사용 안 함)

저장:
  final IMU dataset/subject{N}/laptop1/IMU9/s{N}_a{ACT}_t{TRIAL}_u.csv
  컬럼: timestamp, {SENSOR}_IMU9_Acc_X/Y/Z, {SENSOR}_IMU9_Gyro_X/Y/Z  (센서 CHS,RU,RF 순)
    * timestamp = 0,1,2,... 정수 샘플인덱스 (preprocess가 /source_hz 로 시간 환산)
    * Acc 단위 = m/s² (중력 ≈ 9.8), Gyro 단위 = deg/s 권장

────────────────────────────────────────────────────────────────────────
하드웨어: nRF52840-DK 3대 (각 보드 = BLE 페리페럴 1개, 부위별 CHS/RU/RF).
  - 보드 펌웨어가 IMU 칩(LSM6DS3/ICM-20948 등)을 읽어 BLE notify 로 스트리밍.
  - 호스트(이 파일)는 bleak 로 3대에 연결해 notify 패킷을 받아 CSV로 모은다.
  ⚠️ UUID·패킷형식·스케일은 제조사가 아니라 *너희 펌웨어*가 정한다 →
     아래 'nRF52840 BLE 설정' 블록을 펌웨어와 한 줄씩 맞추면 된다.
  센서가 아직 없으면 --simulate 로 전체 배선을 먼저 검증할 수 있다.

설치: pip install bleak   (--simulate 만 쓸 거면 불필요)
────────────────────────────────────────────────────────────────────────

사용 예:
  # 센서 없이 파이프라인 배선 테스트 (가짜 데이터)
  python3 collect_imu.py --subject 1 --trial 1 --simulate
  # 실제 수집 (nRF 3대 켜고 펌웨어 설정 맞춘 뒤)
  python3 collect_imu.py --subject 1 --trial 1 --duration 8
"""

import argparse
import asyncio
import csv
import math
import os
import struct
import threading
import time

# 표준 센서 이름 (preprocess_imu.py 의 CANONICAL/접두어와 일치해야 함)
SENSORS = ["CHS", "RU", "RF"]          # 가슴, 위팔, 아래팔
AXES = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]


def csv_header():
    """preprocess_imu.py 가 읽는 원본 헤더: timestamp + {센서}_IMU9_{축}."""
    cols = ["timestamp"]
    for s in SENSORS:
        for ax in AXES:
            cols.append(f"{s}_IMU9_{ax}")
    return cols


# ============================================================================
# 센서 드라이버 (하드웨어 의존 — 너희 센서에 맞게 구현)
# ============================================================================
class SensorDriver:
    """수집 루프가 기대하는 인터페이스. 실제/시뮬 드라이버가 이걸 구현한다."""
    def connect(self):
        """센서 3개 연결. 실패 시 예외 발생."""
        raise NotImplementedError

    def poll(self):
        """현재 시점 한 샘플 반환: {센서명: (ax, ay, az, gx, gy, gz)}.
        Acc=m/s², Gyro=deg/s. 아직 새 값이 없으면 직전 값을 반환해도 됨."""
        raise NotImplementedError

    def close(self):
        pass


# ────────────────────────────────────────────────────────────────────────
# nRF52840 BLE 설정 — 보드 펌웨어와 반드시 일치시킬 것
# ────────────────────────────────────────────────────────────────────────
# 센서 3개 = nRF 보드 3대. 각 보드의 BLE '광고 이름'으로 어느 부위인지 구분한다.
#   → 펌웨어의 device name(또는 MAC주소)을 여기에 적는다. 주소("XX:XX:..")도 그대로 인식.
DEVICE_NAMES = {"CHS": "IMU-CHS", "RU": "IMU-RU", "RF": "IMU-RF"}

# 보드가 IMU 데이터를 notify 하는 GATT 특성 UUID.
#   기본값 = Nordic UART Service(NUS) 의 TX 특성. 커스텀 서비스를 쓰면 그 UUID로 바꾼다.
NOTIFY_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

SCAN_TIMEOUT = 10.0      # 스캔 대기(초)

# 패킷 → 물리값 스케일. IMU 칩 full-scale 설정에 맞춰 조정 (아래는 ±4g / ±2000dps 예시).
ACC_SCALE  = (4.0 / 32768.0) * 9.80665    # raw int16 → m/s²
GYRO_SCALE = 2000.0 / 32768.0             # raw int16 → deg/s


def parse_packet(data: bytes):
    """펌웨어가 보낸 notify 1패킷 → (ax, ay, az, gx, gy, gz)  [m/s², deg/s].

    기본 가정: 12바이트 = int16 리틀엔디언 6개 [ax,ay,az,gx,gy,gz] (raw count).
    펌웨어 패킷 레이아웃이 다르면 (float, 헤더/타임스탬프 포함 등) 이 함수만 고치면 된다.
    """
    ax, ay, az, gx, gy, gz = struct.unpack("<6h", data[:12])
    return (ax * ACC_SCALE, ay * ACC_SCALE, az * ACC_SCALE,
            gx * GYRO_SCALE, gy * GYRO_SCALE, gz * GYRO_SCALE)


class NrfBleDriver(SensorDriver):
    """nRF52840-DK 3대를 bleak 로 연결해 notify 패킷을 받는 드라이버.

    BLE/bleak 은 비동기라, 백그라운드 스레드에서 asyncio 루프를 돌리며
    notify 콜백이 self.latest[센서] 를 갱신한다. 수집 루프는 poll() 로 최신값만 읽는다.
    """
    def __init__(self):
        self.latest = {s: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for s in SENSORS}
        self._loop = None
        self._thread = None
        self._clients = {}

    def connect(self):
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            raise RuntimeError(
                "bleak 미설치 → `pip install bleak` 후 다시 실행.\n"
                "센서 없이 테스트하려면 --simulate 를 쓰세요.")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(
            self._connect_all(BleakClient, BleakScanner), self._loop)
        fut.result()       # 연결 완료까지 대기 (실패 시 여기서 예외 발생)

    async def _connect_all(self, BleakClient, BleakScanner):
        print(f"  센서 스캔 중... ({SCAN_TIMEOUT:.0f}s)")
        devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT)
        by_name = {d.name: d for d in devices if d.name}
        by_addr = {d.address.upper(): d for d in devices}
        for sensor, want in DEVICE_NAMES.items():
            dev = by_name.get(want) or by_addr.get(want.upper())
            if dev is None:
                raise RuntimeError(
                    f"{sensor} 보드('{want}') 를 못 찾음. "
                    f"발견된 이름: {sorted(by_name)}  "
                    f"(DEVICE_NAMES 를 실제 광고이름/주소로 맞추세요)")
            client = BleakClient(dev)
            await client.connect()
            await client.start_notify(NOTIFY_UUID, self._make_cb(sensor))
            self._clients[sensor] = client
            print(f"  연결됨: {sensor} ← {want}")

    def _make_cb(self, sensor):
        def cb(_char, data):
            try:
                self.latest[sensor] = parse_packet(bytes(data))
            except Exception:
                pass        # 깨진 패킷 1개는 무시(직전 값 유지)
        return cb

    def poll(self):
        return dict(self.latest)

    def close(self):
        if not self._loop:
            return
        async def _disc():
            for c in self._clients.values():
                try:
                    await c.disconnect()
                except Exception:
                    pass
        try:
            asyncio.run_coroutine_threadsafe(_disc(), self._loop).result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


class SimulatedDriver(SensorDriver):
    """센서 없이 파이프라인을 검증하기 위한 가짜 신호 생성기.
    중력(축마다 고정 오프셋) + 저주파 동작 + 소량 노이즈. 절대 실제 데이터 아님(검증용)."""
    def __init__(self, hz):
        self.hz = hz
        self.k = 0
        # 센서별 중력 방향(부착 자세가 제각각이라는 현실 반영) — m/s²
        self.gravity = {
            "CHS": (0.5, -9.6, 1.2),
            "RU":  (-1.0, 9.5, 2.0),
            "RF":  (2.0, -1.5, 9.4),
        }

    def connect(self):
        print("  [simulate] 가짜 센서 3개 연결됨 (CHS/RU/RF)")

    def poll(self):
        t = self.k / self.hz
        self.k += 1
        out = {}
        for i, s in enumerate(SENSORS):
            gx, gy, gz = self.gravity[s]
            # 동작 성분: 센서마다 위상/진폭 다른 저주파 + 잔노이즈(결정적)
            ph = 0.7 * i
            mx = 1.5 * math.sin(2 * math.pi * 0.5 * t + ph)
            my = 1.2 * math.sin(2 * math.pi * 0.4 * t + ph + 1.0)
            mz = 1.0 * math.sin(2 * math.pi * 0.6 * t + ph + 2.0)
            n = 0.05 * math.sin(37.0 * t + i)        # 미세 노이즈(결정적)
            ax, ay, az = gx + mx + n, gy + my + n, gz + mz + n
            # 각속도: 가속도 변화에 대략 비례하는 가짜 값(deg/s)
            wx = 30 * math.cos(2 * math.pi * 0.5 * t + ph)
            wy = 24 * math.cos(2 * math.pi * 0.4 * t + ph + 1.0)
            wz = 20 * math.cos(2 * math.pi * 0.6 * t + ph + 2.0)
            out[s] = (ax, ay, az, wx, wy, wz)
        return out


# ============================================================================
# 수집 루프
# ============================================================================
def collect(driver, hz, duration):
    """target hz 로 duration 초 동안 샘플을 모아 [(timestamp_idx, *18values), ...] 반환."""
    period = 1.0 / hz
    n_total = int(round(hz * duration))
    rows = []
    t_start = time.perf_counter()
    for k in range(n_total):
        sample = driver.poll()
        flat = []
        for s in SENSORS:
            flat.extend(sample[s])           # ax,ay,az,gx,gy,gz
        rows.append([k] + flat)              # timestamp = 정수 샘플인덱스
        # 다음 샘플 시각까지 대기 (시뮬레이션도 실시간성 흉내; 너무 빠르면 sleep 생략돼도 무방)
        next_t = t_start + (k + 1) * period
        dt = next_t - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
    return rows


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(csv_header())
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, required=True, help="피험자 번호 (예: 1)")
    ap.add_argument("--trial", type=int, required=True, help="trial 번호 (예: 1)")
    ap.add_argument("--activity", type=int, default=9, help="동작 번호 (기본 9 = 물병 따르기)")
    ap.add_argument("--duration", type=float, default=8.0, help="수집 길이(초). 기본 8")
    ap.add_argument("--hz", type=float, default=100.0,
                    help="샘플링레이트(Hz). preprocess --source-hz 와 반드시 일치! 기본 100")
    ap.add_argument("--data-root", default="data/raw/final IMU dataset", help="원본 데이터셋 루트")
    ap.add_argument("--simulate", action="store_true",
                    help="센서 없이 가짜 데이터로 전체 파이프라인 배선 테스트")
    ap.add_argument("--countdown", type=int, default=3, help="수집 시작 전 카운트다운(초)")
    args = ap.parse_args()

    driver = SimulatedDriver(args.hz) if args.simulate else NrfBleDriver()

    print("=" * 60)
    print(f"IMU 수집  subject{args.subject}  a{args.activity}  t{args.trial}"
          f"  | {args.duration:.0f}s @ {args.hz:.0f}Hz"
          + ("  [SIMULATE]" if args.simulate else ""))
    print("=" * 60)
    print("센서 연결 중...")
    driver.connect()

    if args.countdown and not args.simulate:
        for c in range(args.countdown, 0, -1):
            print(f"  {c}초 뒤 시작... (동작 준비)")
            time.sleep(1)
    print("▶ 수집 시작")
    rows = collect(driver, args.hz, args.duration)
    driver.close()

    # preprocess_imu.py 가 찾는 경로/파일명 규칙으로 저장
    out_dir = os.path.join(args.data_root, f"subject{args.subject}", "laptop1", "IMU9")
    out_name = f"s{args.subject}_a{args.activity}_t{args.trial}_u.csv"
    out_path = os.path.join(out_dir, out_name)
    save_csv(rows, out_path)

    print(f"■ 저장 완료: {out_path}  ({len(rows)} 샘플, {len(rows)/args.hz:.2f}s)")
    print(f"  다음 단계: python3 preprocess_imu.py --subjects subject{args.subject} --source-hz {args.hz:.0f}")


if __name__ == "__main__":
    main()
