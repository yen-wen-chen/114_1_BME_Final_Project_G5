"""Minimal BrainLink / ThinkGear serial parser."""

from __future__ import annotations

import serial
import threading
from collections import defaultdict
from typing import Callable, Optional


class BrainLinkSerialClient:
    """Reads BrainLink/MindWave packets directly from the serial device."""

    def __init__(self, port: str, baud: int = 57600, timeout: float = 1.0) -> None:
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: dict[str, list[Callable[[int], None]]] = defaultdict(list)
        self._values = {
            "attention": 0,
            "meditation": 0,
            "poorSignal": 200,
            "blinkStrength": 0,
            "rawValue": 0,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._serial = serial.Serial(self.port, self.baud, timeout=self.timeout)
        self._running.set()
        self._thread = threading.Thread(target=self._reader_loop, name="BrainLinkSerial", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def register_callback(self, key: str, callback: Callable[[int], None]) -> None:
        self._callbacks[key].append(callback)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._values)

    # internal helpers -----------------------------------------------------
    def _reader_loop(self) -> None:
        assert self._serial is not None
        stream = self._serial
        while self._running.is_set():
            try:
                p1 = stream.read(1)
                if not p1:
                    continue
                if p1[0] != 0xAA:
                    continue
                p2 = stream.read(1)
                if not p2 or p2[0] != 0xAA:
                    continue
                payload_length_bytes = stream.read(1)
                if not payload_length_bytes:
                    continue
                payload_length = payload_length_bytes[0]
                if payload_length > 170:
                    continue  # invalid
                payload = stream.read(payload_length)
                if len(payload) != payload_length:
                    continue
                checksum = stream.read(1)
                if not checksum:
                    continue
                if self._compute_checksum(payload) != checksum[0]:
                    continue
                self._parse_payload(payload)
            except serial.SerialException:
                break
            except Exception:
                continue

    def _compute_checksum(self, payload: bytes) -> int:
        total = sum(payload) & 0xFF
        return (~total) & 0xFF

    def _parse_payload(self, payload: bytes) -> None:
        i = 0
        length = len(payload)
        while i < length:
            code = payload[i]
            i += 1
            if code == 0x02 and i < length:
                self._set_value("poorSignal", payload[i])
                i += 1
            elif code == 0x04 and i < length:
                self._set_value("attention", payload[i])
                i += 1
            elif code == 0x05 and i < length:
                self._set_value("meditation", payload[i])
                i += 1
            elif code == 0x16 and i < length:
                self._set_value("blinkStrength", payload[i])
                i += 1
            elif code == 0x80 and i + 1 < length:
                raw_length = payload[i]
                i += 1
                if raw_length == 0 or i + raw_length > length:
                    i += raw_length
                    continue
                raw_bytes = payload[i : i + raw_length]
                i += raw_length
                if raw_length == 2:
                    value = int.from_bytes(raw_bytes, byteorder="big", signed=True)
                    self._set_value("rawValue", value)
                else:
                    continue
            else:
                # Skip extended codes
                if i >= length:
                    break
                data_len = payload[i]
                i += 1
                i += data_len

    def _set_value(self, key: str, value: int) -> None:
        with self._lock:
            self._values[key] = value
        for callback in self._callbacks.get(key, []):
            try:
                callback(value)
            except Exception:
                continue
