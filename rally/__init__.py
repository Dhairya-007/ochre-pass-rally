"""Ochre Pass Rally - a 3D rally game with a career mode and online multiplayer.

The package splits in two, and the split is load-bearing:

  * Simulation core (`rally.*`): physics, tracks, terrain, AI, collisions,
    the race referee, career, networking. Python and numpy only - no engine
    imports - so all of it is unit-tested headlessly.

  * Presentation (`rally.engine.*`): Panda3D rendering, audio, UI. Only this
    part needs a window.

Run the game with `python main.py` from the project root.
"""

__version__ = "2.0.0"
