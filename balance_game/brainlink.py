"""Integration utilities for BrainLink/NeuroSky blink detection."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .config import BrainLinkConfig


@dataclass
class BlinkEvent:
    """Represents a single blink detection timestamp."""

    timestamp: float
    strength: int


class BrainLinkClient:
    """Minimal ThinkGear socket client focused on blink detection."""

    def __init__(self, config: BrainLinkConfig) -> None:
        self.cfg = config
        self._socket: Optional[socket.socket] = None
        self._buffer = bytearray()
        self._thread: Optional[threading.Thread] = None
        self._events: Deque[BlinkEvent] = deque()
        self._events_lock = threading.Lock()
        self._running = threading.Event()
        self._connected = threading.Event()
        self._last_blink_time = 0.0
        self._poor_signal_level = 200  # default worst quality

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, name="BrainLinkClient", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._connected.clear()
        self._close_socket()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def consume_blink(self) -> Optional[BlinkEvent]:
        """Return and remove the next queued blink event, if any."""
        with self._events_lock:
            if self._events:
                return self._events.popleft()
        return None

    def _run_loop(self) -> None:
        while self._running.is_set():
            if not self._connect():
                time.sleep(self.cfg.reconnect_backoff)
                continue
            try:
                self._listen()
            except Exception:
                # Intentionally swallow to keep trying; callers can inspect connected flag.
                pass
            finally:
                self._connected.clear()
                self._close_socket()
                time.sleep(self.cfg.reconnect_backoff)

    def _connect(self) -> bool:
        try:
            sock = socket.create_connection((self.cfg.host, self.cfg.port), timeout=5.0)
            sock.settimeout(1.0)
            payload = json.dumps({"enableRawOutput": False, "format": "Json"}).encode("utf-8") + b"\n"
            sock.sendall(payload)
        except OSError:
            return False
        self._socket = sock
        self._connected.set()
        return True

    def _close_socket(self) -> None:
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def _listen(self) -> None:
        assert self._socket is not None
        while self._running.is_set():
            try:
                chunk = self._socket.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            self._buffer.extend(chunk)
            self._process_buffer()

    def _process_buffer(self) -> None:
        while b"\n" in self._buffer:
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            line = line.strip()
            if not line:
                continue
            try:
                packet = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            self._handle_packet(packet)

    def _handle_packet(self, packet: dict) -> None:
        if "poorSignalLevel" in packet:
            try:
                self._poor_signal_level = int(packet["poorSignalLevel"])
            except (TypeError, ValueError):
                self._poor_signal_level = 200
        blink_strength = packet.get("blinkStrength")
        if isinstance(blink_strength, int):
            self._consider_blink(blink_strength)

    def _consider_blink(self, strength: int) -> None:
        now = time.perf_counter()
        if (
            strength >= self.cfg.blink_threshold
            and self._poor_signal_level <= self.cfg.min_signal_quality
            and now - self._last_blink_time >= self.cfg.blink_cooldown
        ):
            self._last_blink_time = now
            with self._events_lock:
                self._events.append(BlinkEvent(timestamp=now, strength=strength))
