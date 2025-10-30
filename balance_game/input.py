"""Input abstractions for the tightrope game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pygame


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
