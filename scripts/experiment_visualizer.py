"""Compatibility CLI for DARP experiment replay visualization."""

# TODO(visualization): Keep this wrapper stable while replay internals move into src/darp/visualization.

from __future__ import annotations

from darp.visualization.replay import main


if __name__ == "__main__":
    raise SystemExit(main())
