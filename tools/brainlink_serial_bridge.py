"""Bridge BrainLink serial data (BrainLinkParser) to the game socket."""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainlink.parser import BrainLinkSerialClient
from balance_game.blink_detector import BlinkProfile, EnergyBlinkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use BrainLinkParser serial stream to control the game.")
    parser.add_argument("--serial-port", required=True, help="BrainLink serial device (e.g. /dev/cu.BrainLink_Lite).")
    parser.add_argument("--baud", type=int, default=57600, help="Serial baud rate (default 57600).")
    parser.add_argument("--game-host", default="127.0.0.1", help="Game socket host.")
    parser.add_argument("--game-port", type=int, default=4789, help="Game socket port.")
    parser.add_argument("--profile", default="assets/blink_energy_profile.json", help="Blink energy profile JSON path.")
    parser.add_argument("--model-module", help="Optional module exposing predict(packet: dict) -> dict.")
    parser.add_argument("--fallback-blink-threshold", type=int, default=55, help="BlinkStrength threshold if no profile.")
    parser.add_argument("--send-interval", type=float, default=0.05, help="Seconds between socket updates.")
    parser.add_argument("--verbose", action="store_true", help="Print each control payload sent to the game.")
    parser.add_argument("--debug-sensors", action="store_true", help="Periodically print raw EEG stats (poorSignal, attention, etc.).")
    return parser.parse_args()


def load_predict_fn(module_name: Optional[str]) -> Optional[Callable[[dict], dict]]:
    if not module_name:
        return None
    module = importlib.import_module(module_name)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise AttributeError(f"Module {module_name!r} must define predict(packet: dict) -> dict.")
    return predict  # type: ignore[return-value]


class BrainLinkBridge:
    def __init__(
        self,
        port: str,
        baud: int,
        game_host: str,
        game_port: int,
        detector: Optional[EnergyBlinkDetector],
        fallback_threshold: int,
        predict_fn: Optional[Callable[[dict], dict]],
        send_interval: float,
        verbose: bool,
        debug_sensors: bool,
    ) -> None:
        self.client = BrainLinkSerialClient(port, baud)
        self.game_host = game_host
        self.game_port = game_port
        self.detector = detector
        self.fallback_threshold = fallback_threshold
        self.predict_fn = predict_fn
        self.send_interval = send_interval
        self.verbose = verbose
        self.debug_sensors = debug_sensors
        self._pending_jump = False
        self._last_debug_print = 0.0

        self.client.register_callback("poorSignal", self._on_signal)
        self.client.register_callback("blinkStrength", self._on_blink_strength)
        self.client.register_callback("rawValue", self._on_raw_value)
        self.latest_signal = 200

    def _on_signal(self, value: int) -> None:
        self.latest_signal = value

    def _on_blink_strength(self, value: int) -> None:
        if self.detector:
            return
        if value >= self.fallback_threshold:
            self._pending_jump = True

    def _on_raw_value(self, value: int) -> None:
        if not self.detector:
            return
        if self.detector.process_sample(float(value)):
            self._pending_jump = True

    def run(self) -> None:
        print(f"[INFO] Connecting to BrainLink on {self.client.port} @ {self.client.baud} ...")
        self.client.start()
        print(f"[INFO] Connecting to game socket {self.game_host}:{self.game_port} ...")
        try:
            sock = socket.create_connection((self.game_host, self.game_port))
        except OSError as exc:
            print(f"[ERROR] Unable to connect to game socket: {exc}. Did you run main.py --socket-input ?")
            self.client.stop()
            return

        try:
            with sock:
                print("[INFO] Streaming control data. Press Ctrl+C to stop.")
                last_jump_time = 0.0
                while True:
                    packet = self.client.snapshot()
                    packet["poorSignal"] = self.latest_signal
                    if self.debug_sensors:
                        now = time.perf_counter()
                        if now - self._last_debug_print >= 1.0:
                            self._last_debug_print = now
                            print(
                                "[DEBUG]",
                                f"signal={packet.get('poorSignal')} "
                                f"attention={packet.get('attention')} "
                                f"meditation={packet.get('meditation')} "
                                f"blinkStrength={packet.get('blinkStrength')} "
                                f"rawValue={packet.get('rawValue')}",
                            )
                    control: dict[str, float | bool] = {}
                    if self.predict_fn:
                        try:
                            model_output = self.predict_fn(packet)
                            if isinstance(model_output, dict):
                                control.update(model_output)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[WARN] model predict failed: {exc}")
                    else:
                        attention = packet.get("attention", 50)
                        try:
                            attention = float(attention)
                        except (TypeError, ValueError):
                            attention = 50.0
                        lean = max(-1.0, min(1.0, (attention - 50.0) / 40.0))
                        control["lean"] = lean

                    if self._pending_jump:
                        control["jump"] = True
                        self._pending_jump = False
                        last_jump_time = time.perf_counter()
                    elif last_jump_time and time.perf_counter() - last_jump_time > 0.25:
                        control.setdefault("jump", False)
                        last_jump_time = 0.0

                    if control:
                        payload = json.dumps(control).encode("utf-8") + b"\n"
                        sock.sendall(payload)
                        if self.verbose:
                            print(f"[STATE] {control}")

                    time.sleep(self.send_interval)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping bridge...")
        finally:
            self.client.stop()


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
            print(f"[WARN] Failed to load profile {profile_path}: {exc}")
    else:
        print(f"[WARN] Blink profile {profile_path} not found. Using blinkStrength threshold.")

    predict_fn = load_predict_fn(args.model_module)
    bridge = BrainLinkBridge(
        port=args.serial_port,
        baud=args.baud,
        game_host=args.game_host,
        game_port=args.game_port,
        detector=detector,
        fallback_threshold=args.fallback_blink_threshold,
        predict_fn=predict_fn,
        send_interval=args.send_interval,
        verbose=args.verbose,
        debug_sensors=args.debug_sensors,
    )
    bridge.run()


if __name__ == "__main__":
    main()
