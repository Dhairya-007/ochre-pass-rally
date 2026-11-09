#!/usr/bin/env python3
"""Ochre Pass Rally.

    python main.py                 play
    python main.py --windowed      force a window
    python main.py --quality low   graphics preset: low | medium | high | ultra
    python main.py --host          skip the menus and open a multiplayer lobby
    python main.py --join 1.2.3.4  skip the menus and join a lobby

First run? Download the free art and audio once:

    python tools/fetch_assets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ochre-pass-rally", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windowed", action="store_true", help="run in a window")
    parser.add_argument("--fullscreen", action="store_true", help="run fullscreen")
    parser.add_argument("--quality", choices=("low", "medium", "high", "ultra"))
    parser.add_argument("--size", metavar="WxH", help="window size, e.g. 1920x1080")
    parser.add_argument("--host", action="store_true", help="open a multiplayer lobby")
    parser.add_argument("--join", metavar="ADDRESS", help="join a lobby at ADDRESS[:PORT]")
    parser.add_argument("--fps", action="store_true", help="show the frame-rate counter")
    parser.add_argument("--no-vsync", action="store_true", help="uncap the frame rate")
    args = parser.parse_args(argv)

    from rally.engine.assets import MissingAssets
    from rally.settings import Settings

    settings = Settings.load()
    if args.windowed:
        settings.fullscreen = False
    if args.fullscreen:
        settings.fullscreen = True
    if args.quality:
        settings.quality = args.quality
    if args.size:
        try:
            w, h = (int(v) for v in args.size.lower().split("x"))
            settings.width, settings.height = w, h
        except ValueError:
            parser.error("--size must look like 1920x1080")
    if args.fps:
        settings.show_fps = True
    if args.no_vsync:
        settings.vsync = False

    try:
        from rally.engine.app import RallyApp
        app = RallyApp(settings)
    except MissingAssets as error:
        print(f"\n{error}")
        return 1

    if args.host:
        app.open_multiplayer()
        app.screen.host()
    elif args.join:
        app.open_multiplayer()
        app.screen.address.entry.set(args.join)
        app.screen.join()

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
