"""Entry point for the G5 tightrope balance game."""

from __future__ import annotations

import argparse
import random

from balance_game import GameConfig, TightropeGame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tightrope balance prototype.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for deterministic wind patterns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed()

    config = GameConfig()
    game = TightropeGame(config=config)
    game.run()


if __name__ == "__main__":
    main()
