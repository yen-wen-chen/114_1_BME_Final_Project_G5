"""Entry point for the G5 tightrope balance game."""

from __future__ import annotations

import argparse
import random
from dataclasses import replace

from balance_game import (
    BrainLinkConfig,
    BrainLinkBlinkInput,
    GameConfig,
    KeyboardInput,
    SocketInput,
    SocketInputConfig,
    TightropeGame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tightrope balance prototype.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for deterministic wind patterns.",
    )
    parser.add_argument(
        "--brainlink",
        action="store_true",
        help="Enable BrainLink blink-to-jump input (requires ThinkGear socket service).",
    )
    parser.add_argument(
        "--brainlink-host",
        help="Override BrainLink ThinkGear socket host (default: config value).",
    )
    parser.add_argument(
        "--brainlink-port",
        type=int,
        help="Override BrainLink ThinkGear socket port (default: config value).",
    )
    parser.add_argument(
        "--blink-threshold",
        type=int,
        help="Override blink strength threshold for BrainLink integration.",
    )
    parser.add_argument(
        "--socket-input",
        action="store_true",
        help="Enable JSON-over-TCP control interface for external pipelines.",
    )
    parser.add_argument(
        "--socket-host",
        help="Override socket input bind host (default: config value).",
    )
    parser.add_argument(
        "--socket-port",
        type=int,
        help="Override socket input port (default: config value).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed()

    config = GameConfig()

    brainlink_cfg: BrainLinkConfig = config.brainlink
    brainlink_overrides = {}
    if args.brainlink_host:
        brainlink_overrides["host"] = args.brainlink_host
    if args.brainlink_port is not None:
        brainlink_overrides["port"] = args.brainlink_port
    if args.blink_threshold is not None:
        brainlink_overrides["blink_threshold"] = args.blink_threshold

    if brainlink_overrides:
        brainlink_cfg = replace(brainlink_cfg, **brainlink_overrides)
        config = replace(config, brainlink=brainlink_cfg)

    socket_cfg: SocketInputConfig = config.socket_input
    socket_overrides = {}
    if args.socket_host:
        socket_overrides["host"] = args.socket_host
    if args.socket_port is not None:
        socket_overrides["port"] = args.socket_port

    if socket_overrides:
        socket_cfg = replace(socket_cfg, **socket_overrides)
        config = replace(config, socket_input=socket_cfg)

    input_provider = KeyboardInput()
    if args.brainlink:
        input_provider = BrainLinkBlinkInput(base=input_provider, config=brainlink_cfg)
    if args.socket_input:
        input_provider = SocketInput(base=input_provider, config=socket_cfg)

    game = TightropeGame(config=config, input_provider=input_provider)
    game.run()


if __name__ == "__main__":
    main()
