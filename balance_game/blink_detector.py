"""Energy-based blink detection utilities."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional


@dataclass(frozen=True)
class BlinkProfile:
    """Configuration profile derived from training data."""

    sample_rate: int
    window_sec: float
    threshold: float
    min_blink_sec: float = 0.08
    max_blink_sec: float = 0.35
    refractory_sec: float = 0.5

    @classmethod
    def from_dict(cls, data: dict) -> "BlinkProfile":
        return cls(
            sample_rate=int(data["sample_rate"]),
            window_sec=float(data["window_sec"]),
            threshold=float(data["threshold"]),
            min_blink_sec=float(data.get("min_blink_sec", 0.08)),
            max_blink_sec=float(data.get("max_blink_sec", 0.35)),
            refractory_sec=float(data.get("refractory_sec", 0.5)),
        )

    @classmethod
    def from_json(cls, path: Path) -> "BlinkProfile":
        with path.open() as f:
            return cls.from_dict(json.load(f))


class EnergyBlinkDetector:
    """Sliding-window detector for short high-energy bursts (blinks)."""

    def __init__(
        self,
        threshold: float,
        sample_rate: int,
        window_sec: float = 0.15,
        min_blink_sec: float = 0.08,
        max_blink_sec: float = 0.35,
        refractory_sec: float = 0.5,
    ) -> None:
        self.threshold = float(threshold)
        self.sample_rate = int(sample_rate)
        self.window_samples = max(1, int(window_sec * self.sample_rate))
        self.min_blink_samples = max(1, int(min_blink_sec * self.sample_rate))
        self.max_blink_samples = max(self.min_blink_samples, int(max_blink_sec * self.sample_rate))
        self.refractory_samples = max(1, int(refractory_sec * self.sample_rate))

        self.buffer: Deque[float] = deque(maxlen=self.window_samples)
        self.sum_sq = 0.0
        self.sample_index = 0

        self.state = "idle"  # idle | above
        self.above_start: Optional[int] = None
        self.last_blink_sample: int = -self.refractory_samples

    @classmethod
    def from_profile(cls, profile: BlinkProfile) -> "EnergyBlinkDetector":
        return cls(
            threshold=profile.threshold,
            sample_rate=profile.sample_rate,
            window_sec=profile.window_sec,
            min_blink_sec=profile.min_blink_sec,
            max_blink_sec=profile.max_blink_sec,
            refractory_sec=profile.refractory_sec,
        )

    def reset(self) -> None:
        self.buffer.clear()
        self.sum_sq = 0.0
        self.sample_index = 0
        self.state = "idle"
        self.above_start = None
        self.last_blink_sample = -self.refractory_samples

    def process_sample(self, value: float) -> bool:
        """Return True when a blink is detected."""

        self.sample_index += 1
        if len(self.buffer) == self.window_samples:
            oldest = self.buffer.popleft()
            self.sum_sq -= float(oldest) * float(oldest)
        self.buffer.append(value)
        self.sum_sq += float(value) * float(value)

        if len(self.buffer) < self.window_samples:
            return False

        energy = math.sqrt(self.sum_sq / self.window_samples)
        if energy >= self.threshold:
            if self.state == "idle":
                self.state = "above"
                self.above_start = self.sample_index
            return False

        if self.state == "above" and self.above_start is not None:
            duration = self.sample_index - self.above_start
            self.state = "idle"
            self.above_start = None
            if (
                self.min_blink_samples <= duration <= self.max_blink_samples
                and self.sample_index - self.last_blink_sample >= self.refractory_samples
            ):
                self.last_blink_sample = self.sample_index
                return True

        return False
