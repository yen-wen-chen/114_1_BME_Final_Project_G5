"""Verify blink detection against recorded open/close cycles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balance_game.blink_detector import BlinkProfile, EnergyBlinkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate blink energy profile on recorded datasets.")
    parser.add_argument(
        "--profile",
        default="assets/blink_energy_profile.json",
        help="Blink profile JSON produced by train_blink_energy.py.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="One or more 3.txt recordings to evaluate.",
    )
    parser.add_argument("--skip-buffer-sec", type=float, default=20.0, help="Initial buffer to ignore (sec).")
    parser.add_argument("--open-sec", type=float, default=20.0, help="Open segment length (sec).")
    parser.add_argument("--closed-sec", type=float, default=20.0, help="Closed segment length (sec).")
    parser.add_argument("--cycles", type=int, default=10, help="Number of open/closed cycles expected.")
    return parser.parse_args()


def load_signal(path: Path) -> np.ndarray:
    with path.open() as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return np.asarray(data, dtype=np.float32)


def evaluate_file(
    path: Path,
    detector: EnergyBlinkDetector,
    fs: int,
    skip_buffer_sec: float,
    open_sec: float,
    closed_sec: float,
    cycles: int,
) -> list[tuple[float, str]]:
    data = load_signal(path)
    detector.reset()
    events: list[tuple[float, str]] = []
    for value in data:
        if detector.process_sample(value):
            t = detector.sample_index / fs
            stage = classify_stage(
                t,
                skip_buffer_sec=skip_buffer_sec,
                open_sec=open_sec,
                closed_sec=closed_sec,
                cycles=cycles,
            )
            events.append((t, stage))
    return events


def classify_stage(
    timestamp: float,
    *,
    skip_buffer_sec: float,
    open_sec: float,
    closed_sec: float,
    cycles: int,
) -> str:
    if timestamp < skip_buffer_sec:
        return "buffer"
    relative = timestamp - skip_buffer_sec
    cycle_duration = open_sec + closed_sec
    cycle = int(relative // cycle_duration)
    if cycle >= cycles:
        return "tail"
    phase = relative % cycle_duration
    if phase < open_sec:
        return f"cycle{cycle+1}-open"
    return f"cycle{cycle+1}-closed"


def summarize(events: list[tuple[float, str]]) -> None:
    if not events:
        print("  No blink events detected.")
        return
    for t, stage in events:
        print(f"  t={t:7.2f}s  stage={stage}")
    stages = {}
    for _, stage in events:
        stages[stage] = stages.get(stage, 0) + 1
    print("  Counts by stage:")
    for stage, count in sorted(stages.items()):
        print(f"    {stage}: {count}")


def main() -> None:
    args = parse_args()
    profile = BlinkProfile.from_json(Path(args.profile).expanduser())
    detector = EnergyBlinkDetector.from_profile(profile)

    for dataset in args.datasets:
        path = Path(dataset).expanduser()
        if path.is_dir():
            files: Sequence[Path] = sorted(path.glob("S*/3.txt"))
            if not files:
                print(f"[WARN] No 3.txt files under {path}")
                continue
        else:
            files = [path]

        for file_path in files:
            if not file_path.exists():
                print(f"[WARN] Missing {file_path}")
                continue
            print(f"Evaluating {file_path} ...")
            events = evaluate_file(
                file_path,
                detector,
                fs=profile.sample_rate,
                skip_buffer_sec=args.skip_buffer_sec,
                open_sec=args.open_sec,
                closed_sec=args.closed_sec,
                cycles=args.cycles,
            )
            summarize(events)
            print()


if __name__ == "__main__":
    main()
