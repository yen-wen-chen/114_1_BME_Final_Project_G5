"""Input abstractions for the tightrope game."""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import pygame

from .brainlink import BrainLinkClient
from .config import BrainLinkConfig, SocketInputConfig


@dataclass(frozen=True)
class InputState:
    """Snapshot of player intent."""

    lean: float = 0.0  # -1 (left) .. +1 (right)
    jump: bool = False


class InputProvider(Protocol):
    """Interface for supplying player input to the game loop."""

    def poll(self, dt: float) -> InputState:
        """Return an InputState representing the latest player intent."""


class KeyboardInput(InputProvider):
    """Default keyboard controller (A/D for lean, Space to jump)."""

    def __init__(self, smoothing: float = 8.0) -> None:
        self.smoothing = smoothing
        self._current = 0.0

    def poll(self, dt: float) -> InputState:
        pressed = pygame.key.get_pressed()
        target = 0.0
        if pressed[pygame.K_a] or pressed[pygame.K_LEFT]:
            target -= 1.0
        if pressed[pygame.K_d] or pressed[pygame.K_RIGHT]:
            target += 1.0

        if self.smoothing > 0 and dt > 0:
            blend = min(1.0, self.smoothing * dt)
            self._current += (target - self._current) * blend
        else:
            self._current = target

        jump = bool(pressed[pygame.K_SPACE])
        return InputState(lean=self._current, jump=jump)

    def reset(self) -> None:
        """Reset smoothed input state."""
        self._current = 0.0


class BrainLinkBlinkInput(InputProvider):
    """Augments a base input provider with blink-triggered jumps via BrainLink."""

    def __init__(
        self,
        base: Optional[InputProvider] = None,
        config: Optional[BrainLinkConfig] = None,
    ) -> None:
        self.base = base or KeyboardInput()
        self.cfg = config or BrainLinkConfig()
        self.client = BrainLinkClient(self.cfg)
        self.client.start()
        self._blink_timer = 0.0

    def poll(self, dt: float) -> InputState:
        base_state = self.base.poll(dt)
        blink_event = self.client.consume_blink()
        if blink_event:
            self._blink_timer = self.cfg.blink_hold

        jump_active = base_state.jump
        if self._blink_timer > 0.0:
            self._blink_timer = max(0.0, self._blink_timer - dt)
            jump_active = True

        return InputState(lean=base_state.lean, jump=jump_active)

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()  # type: ignore[attr-defined]
        self._blink_timer = 0.0

    def shutdown(self) -> None:
        self.client.stop()
        if hasattr(self.base, "shutdown"):
            self.base.shutdown()  # type: ignore[attr-defined]

    @property
    def connected(self) -> bool:
        return self.client.connected


class SocketInput(InputProvider):
    """Listens for JSON control messages over TCP to drive the game."""

    def __init__(
        self,
        base: Optional[InputProvider] = None,
        config: Optional[SocketInputConfig] = None,
    ) -> None:
        self.base = base or KeyboardInput()
        self.cfg = config or SocketInputConfig()
        self._lock = threading.Lock()
        self._latest = InputState()
        self._last_update = 0.0
        self._jump_latch = 0.0
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(target=self._run_server, name="SocketInput", daemon=True)
        self._thread.start()

    def poll(self, dt: float) -> InputState:
        base_state = self.base.poll(dt)
        now = time.perf_counter()
        with self._lock:
            state = self._latest
            fresh = now - self._last_update <= self.cfg.max_idle_seconds
            jump_latch = self._jump_latch
            if jump_latch > 0.0:
                jump_latch = max(0.0, jump_latch - dt)
                self._jump_latch = jump_latch

        lean = base_state.lean
        jump = base_state.jump
        if fresh:
            lean = state.lean
            if jump_latch > 0.0:
                jump = True
            else:
                jump = state.jump or jump

        return InputState(lean=lean, jump=jump)

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()  # type: ignore[attr-defined]
        with self._lock:
            self._latest = InputState()
            self._last_update = 0.0
            self._jump_latch = 0.0

    def shutdown(self) -> None:
        self._running.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if hasattr(self.base, "shutdown"):
            self.base.shutdown()  # type: ignore[attr-defined]

    def _run_server(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind((self.cfg.host, self.cfg.port))
                server.listen(self.cfg.backlog)
                server.settimeout(1.0)
            except OSError:
                return

            while self._running.is_set():
                try:
                    client, _ = server.accept()
                    client.settimeout(self.cfg.read_timeout)
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(
                    target=self._handle_client,
                    args=(client,),
                    daemon=True,
                ).start()

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            buffer = bytearray()
            while self._running.is_set():
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buffer.extend(data)
                while b"\n" in buffer:
                    line, _, remainder = buffer.partition(b"\n")
                    buffer = bytearray(remainder)
                    self._process_line(line.strip())

    def _process_line(self, raw: bytes) -> None:
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

        lean = None
        jump = None
        if isinstance(payload, dict):
            if "lean" in payload:
                try:
                    lean_val = float(payload["lean"])
                    lean = max(-1.0, min(1.0, lean_val))
                except (TypeError, ValueError):
                    pass
            if "jump" in payload:
                jump = bool(payload["jump"])
            if payload.get("reset"):
                self.reset()

        if lean is None and jump is None:
            return

        now = time.perf_counter()
        with self._lock:
            current = self._latest
            if lean is None:
                lean = current.lean
            if jump is None:
                jump = current.jump
            else:
                if jump:
                    self._jump_latch = max(self._jump_latch, 0.25)
                else:
                    self._jump_latch = min(self._jump_latch, 0.1)
            self._latest = InputState(lean=lean, jump=jump)
            self._last_update = now
