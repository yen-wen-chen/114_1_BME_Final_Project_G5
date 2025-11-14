"""Bridge BrainLink (via NeuroSkyPy) → tightrope game socket."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NeuroSkyPy.NeuroSkyPy import NeuroSkyPy as NSP  # type: ignore

from balance_game.blink_detector import BlinkProfile, EnergyBlinkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use NeuroSkyPy Bluetooth stream to control the game.")
    parser.add_argument("--serial-port", required=True, help="BrainLink / MindWave serial port (e.g. /dev/tty.MindWaveMobile-SerialPort).")
    parser.add_argument("--baud", type=int, default=57600, help="Serial baud rate (default: 57600).")
    parser.add_argument("--game-host", default="127.0.0.1", help="Game socket host (default: 127.0.0.1).")
    parser.add_argument("--game-port", type=int, default=4789, help="Game socket port (default: 4789).")
    parser.add_argument("--profile", default="assets/blink_energy_profile.json", help="Blink energy profile JSON.")
    parser.add_argument("--model-module", help="Optional Python module exposing predict(packet: dict) -> dict for lean control.")
    parser.add_argument("--fallback-blink-threshold", type=int, default=55, help="Blink strength threshold if no profile is present.")
    parser.add_argument("--send-interval", type=float, default=0.05, help="Seconds between socket updates (default: 50 Hz).")
    parser.add_argument("--verbose", action="store_true", help="Print each control payload that is sent to the game.")
    return parser.parse_args()


def load_predict_function(module_name: Optional[str]) -> Optional[Callable[[dict], dict]]:
    if not module_name:
        return None
    module = __import__(module_name, fromlist=["predict"])
    predict = getattr(module, "predict", None)
    if predict is None or not callable(predict):
        raise AttributeError(f"Module {module_name!r} must define predict(packet: dict) -> dict.")
    return predict  # type: ignore[return-value]


class NeuroSkyBridge:
    def __init__(
        self,
        serial_port: str,
        baud: int,
        game_host: str,
        game_port: int,
        detector: Optional[EnergyBlinkDetector],
        fallback_threshold: int,
        predict_fn: Optional[Callable[[dict], dict]],
        send_interval: float,
        verbose: bool,
    ) -> None:
        self.serial_port = serial_port
        self.baud = baud
        self.game_host = game_host
        self.game_port = game_port
        self.detector = detector
        self.fallback_threshold = fallback_threshold
        self.predict_fn = predict_fn
        self.send_interval = send_interval
        self.verbose = verbose

        self.device = NSP(serial_port, baud)
        self.lock = threading.Lock()
        self.latest = {
            "attention": 0,
            "meditation": 0,
            "poorSignal": 0,
        }
        self._pending_jump = False

        self.device.setCallBack("rawValue", self._on_raw)
        self.device.setCallBack("blinkStrength", self._on_blink_strength)
        self.device.setCallBack("poorSignal", self._on_poor_signal)

    def _on_raw(self, value: int) -> None:
        if not self.detector:
            return
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return
        if self.detector.process_sample(fval):
            with self.lock:
                self._pending_jump = True

    def _on_blink_strength(self, value: int) -> None:
        if self.detector:
            return
        try:
            strength = int(value)
        except (TypeError, ValueError):
            return
        if strength >= self.fallback_threshold:
            with self.lock:
                self._pending_jump = True

    def _on_poor_signal(self, value: int) -> None:
        with self.lock:
            self.latest["poorSignal"] = int(value)

    def _consume_jump(self) -> bool:
        with self.lock:
            if self._pending_jump:
                self._pending_jump = False
                return True
        return False

    def _snapshot_packet(self) -> dict:
        return {
            "attention": int(self.device.attention),
            "meditation": int(self.device.meditation),
            "poorSignal": int(self.latest.get("poorSignal", 0)),
            "timestamp": time.perf_counter(),
        }

    def _default_control(self, packet: dict) -> dict:
        attention = packet.get("attention", 50)
        try:
            lean = max(-1.0, min(1.0, (attention - 50.0) / 40.0))
        except TypeError:
            lean = 0.0
        return {"lean": lean}

    def run(self) -> None:
        print(f"[INFO] Connecting to BrainLink on {self.serial_port} @ {self.baud} ...")
        self.device.start()
        print(f"[INFO] Connecting to game socket {self.game_host}:{self.game_port} ...")
        try:
            with socket.create_connection((self.game_host, self.game_port)) as sock:
                print("[INFO] Streaming control data. Press Ctrl+C to stop.")
                while True:
                    packet = self._snapshot_packet()
                    control: dict[str, float | bool] = {}
                    if self.predict_fn:
                        try:
                            result = self.predict_fn(packet)
                            if isinstance(result, dict):
                                control.update(result)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[WARN] model predict failed: {exc}")
                    else:
                        control.update(self._default_control(packet))

                    if self._consume_jump():
                        control["jump"] = True

                    if control:
                        payload = json.dumps(control).encode("utf-8") + b"\n"
                        sock.sendall(payload)
                        if self.verbose:
                            print(f"[STATE] {control}")

                    time.sleep(self.send_interval)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping bridge...")
        finally:
            self.device.stop()


def main() -> None:
    args = parse_args()
    detector: Optional[EnergyBlinkDetector] = None
    profile_path = Path(args.profile).expanduser()
    if profile_path.exists():
        try:
            profile = BlinkProfile.from_json(profile_path)
            detector = EnergyBlinkDetector.from_profile(profile)
            print(f"[INFO] Loaded blink profile from {profile_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not load profile {profile_path}: {exc}")
    else:
        print(f"[WARN] Blink profile {profile_path} not found. Using blinkStrength threshold.")

    predict_fn = load_predict_function(args.model_module)
    bridge = NeuroSkyBridge(
        serial_port=args.serial_port,
        baud=args.baud,
        game_host=args.game_host,
        game_port=args.game_port,
        detector=detector,
        fallback_threshold=args.fallback_blink_threshold,
        predict_fn=predict_fn,
        send_interval=args.send_interval,
        verbose=args.verbose,
    )
    bridge.run()


if __name__ == "__main__":
    main()
