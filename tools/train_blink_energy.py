"""Train an energy threshold profile for blink detection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_DATASETS = [
    "~/Downloads/BME_Lab_BCI_training/bci_dataset_114-1",
    "~/Downloads/BME_Lab_BCI_training/bci_dataset_113-2",
]


@dataclass
class EnergyStats:
    mean: float
    std: float
    median: float
    p10: float
    p90: float

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EnergyStats":
        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            median=float(np.median(arr)),
            p10=float(np.percentile(arr, 10)),
            p90=float(np.percentile(arr, 90)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute blink energy threshold from EEG datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Directories containing S*/3.txt recordings (default: known BrainLink training folders).",
    )
    parser.add_argument("--output", default="assets/blink_energy_profile.json", help="Output JSON path.")
    parser.add_argument("--sample-rate", type=int, default=500, help="Sampling rate of the recordings (Hz).")
    parser.add_argument("--window-sec", type=float, default=0.06, help="Sliding window size in seconds.")
    parser.add_argument("--hop-sec", type=float, default=0.02, help="Hop size between windows (sec).")
    parser.add_argument("--min-blink-sec", type=float, default=0.08, help="Minimum energy burst length (sec).")
    parser.add_argument("--max-blink-sec", type=float, default=0.35, help="Maximum energy burst length (sec).")
    parser.add_argument("--refractory-sec", type=float, default=0.5, help="Refractory period between blinks (sec).")
    parser.add_argument("--skip-buffer-sec", type=float, default=20.0, help="Initial buffer to discard (sec).")
    parser.add_argument("--open-sec", type=float, default=20.0, help="Open-eye segment duration (sec).")
    parser.add_argument("--closed-sec", type=float, default=20.0, help="Closed-eye segment duration (sec).")
    parser.add_argument("--cycles", type=int, default=10, help="Number of open/closed cycles in each file.")
    return parser.parse_args()


def load_signal(path: Path) -> np.ndarray:
    with path.open() as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return np.asarray(data, dtype=np.float32)


def iter_files(root_dirs: Sequence[str]) -> Iterable[Path]:
    for raw_root in root_dirs:
        root = Path(raw_root).expanduser()
        if not root.exists():
            continue
        for path in sorted(root.glob("S*/3.txt")):
            yield path


def collect_energy_windows(
    data: np.ndarray,
    fs: int,
    window_samples: int,
    hop_samples: int,
    skip_buffer_sec: float,
    open_sec: float,
    closed_sec: float,
    cycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    buffer_samples = int(skip_buffer_sec * fs)
    open_samples = int(open_sec * fs)
    closed_samples = int(closed_sec * fs)
    cycle_samples = open_samples + closed_samples

    if len(data) <= buffer_samples + window_samples:
        raise ValueError("Recording shorter than expected.")

    available_cycles = min(
        cycles,
        max(0, (len(data) - buffer_samples - window_samples) // cycle_samples),
    )
    if available_cycles == 0:
        raise ValueError("Recording shorter than expected.")

    open_windows: list[float] = []
    closed_windows: list[float] = []
    offset = buffer_samples

    for cycle in range(available_cycles):
        open_start = offset + cycle * cycle_samples
        close_start = open_start + open_samples
        open_seg = data[open_start : open_start + open_samples]
        close_seg = data[close_start : close_start + closed_samples]

        for seg, target in ((open_seg, open_windows), (close_seg, closed_windows)):
            if len(seg) < window_samples:
                continue
            for idx in range(0, len(seg) - window_samples + 1, hop_samples):
                window = seg[idx : idx + window_samples]
                energy = math.sqrt(float(np.mean(window * window)))
                target.append(energy)

    return np.asarray(open_windows, dtype=np.float32), np.asarray(closed_windows, dtype=np.float32)


def compute_threshold(open_stats: EnergyStats, closed_stats: EnergyStats) -> float:
    between = (open_stats.p90 + closed_stats.p10) / 2.0
    mean_std = open_stats.mean + 1.5 * open_stats.std
    return float(max(between, mean_std))


def main() -> None:
    args = parse_args()
    fs = args.sample_rate
    window_samples = max(1, int(args.window_sec * fs))
    hop_samples = max(1, int(args.hop_sec * fs))

    open_energies: list[np.ndarray] = []
    closed_energies: list[np.ndarray] = []
    used_files = 0

    files = list(iter_files(args.datasets))
    if not files:
        print("No 3.txt recordings found. Check dataset paths.", file=sys.stderr)
        raise SystemExit(1)

    for path in files:
        try:
            data = load_signal(path)
            open_vals, closed_vals = collect_energy_windows(
                data,
                fs=fs,
                window_samples=window_samples,
                hop_samples=hop_samples,
                skip_buffer_sec=args.skip_buffer_sec,
                open_sec=args.open_sec,
                closed_sec=args.closed_sec,
                cycles=args.cycles,
            )
            if open_vals.size == 0 or closed_vals.size == 0:
                print(f"[WARN] Insufficient windows in {path}")
                continue
            open_energies.append(open_vals)
            closed_energies.append(closed_vals)
            used_files += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Skipping {path}: {exc}")

    if not open_energies or not closed_energies:
        print("Not enough data to compute statistics.", file=sys.stderr)
        raise SystemExit(1)

    open_arr = np.concatenate(open_energies)
    closed_arr = np.concatenate(closed_energies)

    open_stats = EnergyStats.from_array(open_arr)
    closed_stats = EnergyStats.from_array(closed_arr)
    threshold = compute_threshold(open_stats, closed_stats)

    profile = {
        "sample_rate": fs,
        "window_sec": args.window_sec,
        "threshold": threshold,
        "min_blink_sec": args.min_blink_sec,
        "max_blink_sec": args.max_blink_sec,
        "refractory_sec": args.refractory_sec,
        "stats": {
            "open": asdict(open_stats),
            "closed": asdict(closed_stats),
            "recordings": used_files,
            "windows_open": int(open_arr.size),
            "windows_closed": int(closed_arr.size),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(profile, fh, indent=2)

    print(f"Profile saved to {output_path}")
    print(f"Suggested threshold: {threshold:.4f}")
    print(f"Open mean {open_stats.mean:.4f} ± {open_stats.std:.4f}")
    print(f"Closed mean {closed_stats.mean:.4f} ± {closed_stats.std:.4f}")


if __name__ == "__main__":
    main()
