# Ochre Pass Rally

A 3D rally game in Python, with a three-tier **career**, **AI rivals**,
**online multiplayer**, four stages on four surfaces, five cars with upgrades,
physically based rendering with HDR skies and real shadows, and procedural
engine audio.

Everything is free: the code and libraries are open source, and every model,
texture, sky, sound and font is CC0 or SIL OFL. See `assets/CREDITS.md` after
downloading.

```bash
pip install -r requirements.txt
python tools/fetch_assets.py        # one-off, ~105 MB of free assets
python main.py
```

---

## Playing

| Key | Racing | Menus |
| --- | --- | --- |
| `W` / `↑` | Throttle | Move up |
| `S` / `↓` | Brake, then reverse | Move down |
| `A` `D` / `←` `→` | Steer | Change a setting |
| `Space` | Handbrake | |
| `R` | Recover to the stage | |
| `C` | Camera (chase / far / bonnet) | |
| `Esc` | Pause | Back |
| `Enter` | | Select |

Gamepads work too: left stick steers, triggers are throttle and brake, A is the
handbrake, Y recovers, Start pauses, the d-pad drives the menus.

### Career

Three championships: **Rookie Cup**, **National Series** and the **World Rally
Championship**. Each season is a run of events against a fixed roster of AI
rivals who carry points from event to event (25-18-15-12-10-8-6-4). Finish a
season in the top three to unlock the next tier; win it for a title bonus.
Seasons can be replayed to earn money.

Prize money goes on the garage: five cars from $0 to $52,000 and four upgrade
lines per car (engine, tyres, suspension, brakes), three levels each. The
career saves automatically after every event and garage purchase.

### Multiplayer

One player picks **Multiplayer → Host a lobby**; the screen shows their IP
address. Everyone else enters it under **Join**. The host picks the stage,
laps and how many AI bots fill the grid, and starts the race once everyone is
ready. Up to 8 cars.

- **Same Wi-Fi / LAN:** works as is.
- **Over the internet:** the host must forward **TCP port 47800** on their
  router to their computer, and give friends their public IP.

Shortcuts: `python main.py --host`, `python main.py --join 192.168.1.20`.

### Options

| Flag | |
| --- | --- |
| `--windowed` / `--fullscreen` | Window mode |
| `--quality low/medium/high/ultra` | Graphics preset (shadows, MSAA, sky and scenery detail) |
| `--size 1920x1080` | Window size |
| `--no-vsync` | Uncap the frame rate |
| `--fps` | Frame-rate counter |

Settings, the career save and caches live in your user data folder
(`~/Library/Application Support/OchrePassRally` on macOS, `%APPDATA%` on
Windows, `~/.local/share` on Linux) - not next to the code.

---

## The stages

| Stage | Surface | Character |
| --- | --- | --- |
| **Pine Ridge**, Finland | Forest gravel | Fast and flowing through dense pine, blind crests |
| **Red Mesa**, Morocco | Red laterite | Wide-open and very fast, little grip off the line |
| **Frostfell**, Sweden | Packed snow | 70% grip on the line, much less off it |
| **Dusk Valley**, Wales | Mud | Twisty forestry track at sunset, five hairpins |

Each track is **generated**, not hand-placed: control points sit on a closed
curve whose radius is a sum of a few sine waves around the loop. A wave with k
cycles and amplitude a tightens the sharpest corner by roughly a·(k²−1), so
that is the knob for how twisty a stage is - and because every radius stays
positive the loop is star-shaped and cannot cross itself.

## The cars

| Car | Price | Character |
| --- | --- | --- |
| Kestrel R2 | starter | Light, forgiving, rotates easily |
| Vanta GT-S | $18,000 | More power, stable, lazier turn-in |
| Ridgeback T1 | $26,000 | Heavy 4×4 - slower on gravel, **faster on snow** |
| Torque RS 4×4 | $34,000 | Big turbo, all-wheel grip |
| Apex Proto R5 | $52,000 | Savage, sharp, unforgiving |

Cars differ in the physics, not just the bars. Two traits do most of it:
`loose_traction` (a 4×4 claws back some of the grip a loose surface takes
away) and `rough_terrain` (how much going off the stage slows you). Measured
with the AI driving, the Ridgeback is ~1% slower than the Kestrel on Pine
Ridge's dry gravel and ~2% faster on Frostfell's snow.

---

## How it works

### Driving

The car keeps longitudinal and lateral velocity separately, in its own frame.
Steering rotates the car; lateral grip bleeds the sideways velocity away. The
key non-linearity is that **grip falls off as slip angle rises**, so once the
back steps out it keeps coming round and you hold the slide on the throttle.
The handbrake drops rear grip to 17%; full throttle above 12 m/s costs 20%;
the verge and the scrub cost more.

### AI

Each rival follows a **racing line** - inside at the apex, wide on entry and
exit - with a **speed profile** computed once per stage and car: the cornering
limit at every point, a backward pass so it can brake in time, a forward pass
so it only accelerates as fast as the engine allows. Every frame it steers by
pure pursuit towards a look-ahead point, counter-steers into slides, and
dodges cars ahead.

The cornering limit was **measured, not guessed**: driving the physics model
round 25, 50 and 90 m skidpans showed this grip model gives lateral
acceleration a ≈ 2.8·√v at full grip (it is a decay rate, not a friction
coefficient), so the corner speed is v = (2.8·r)^(2/3). Skill scales how close
to that limit a rival drives; lower-skill rivals also wander off line and
occasionally misjudge a braking point.

### Multiplayer

TCP, length-prefixed JSON. Each player's own machine simulates their own car,
so driving never feels laggy; the host is the referee for the lobby, the grid,
the countdown, AI bots and the finishing order, and relays every car's state
at 20 Hz. Everyone renders the other cars 100 ms in the past, interpolating
between snapshots, so they move smoothly. Everything received is
size-capped and validated - a NaN, a 10⁹ coordinate or a garbage frame is
coerced or rejected rather than crashing anyone.

### Rendering

[simplepbr](https://github.com/Moguri/panda3d-simplepbr) on Panda3D:
physically based materials with normal maps, image-based lighting from the
stage's HDR sky, a shadow-mapped sun that follows the player, fog and filmic
tone mapping. Its shaders are GLSL 1.20, which matters on macOS, where the
default OpenGL context cannot compile newer GLSL.

Two corrections make the free assets look right: Kenney's models store sRGB
colours in a field glTF defines as linear, so they are converted on load (and
re-tinted per stage); and Poly Haven's colour maps are sRGB while its normal
and roughness maps are data, so each gets the right texture format.

The sky's image-based lighting is integrated by simplepbr in pure Python;
fed the 512 px display cubemap it took 8.7 s per stage. Lighting only needs
the low frequencies, so it gets a 64 px cube and the result is cached to disk:
the first visit to a stage takes ~14 s, every later one 0.02 s.

---

## Layout

```
rally-racer/
├── main.py                     entry point and command-line flags
├── tools/fetch_assets.py       downloads the free assets, writes CREDITS.md
├── rally/                      ── simulation core: no engine imports ──
│   ├── vehicle.py              handling model
│   ├── track.py  terrain.py    centreline, height / grip / distance fields
│   ├── stages.py               the four stages, generated tracks
│   ├── scenery.py              where every tree and rock goes
│   ├── collision.py            car-vs-scenery and car-vs-car
│   ├── ai.py                   racing line, speed profile, drivers
│   ├── race.py                 the race referee: grid, timing, gaps, results
│   ├── cars.py  career.py      roster, upgrades, championships, save file
│   ├── net/                    protocol, transport, host and client sessions
│   ├── sound.py                procedural engine / tyre / surface loops
│   └── engine/                 ── presentation: Panda3D ──
│       ├── app.py              window, screens, input, loop
│       ├── race_controller.py  runs one race in any mode
│       ├── renderer.py         PBR pipeline and per-stage lighting
│       ├── world.py  car_view.py  particles.py  camera.py  audio.py
│       └── ui/                 kit, HUD, menus, career, multiplayer
└── tests/                      127 headless tests
```

## Tests

```bash
python -m unittest discover -s tests -v
```

127 tests, no window, about six seconds. Beyond the physics and timing they
cover: every stage generating a sane, non-crossing loop with drivable
hairpins; AI lapping every stage on the road and getting faster with skill;
cars not passing through trees and conserving momentum in car-to-car hits;
whole races producing ordered results and DNFs; the career's points, prizes,
unlocks, garage rules and save/load round-trip; the network protocol against
chunked, oversized, garbage and hostile input; and a real loopback host with
clients joining, readying, starting and relaying state.

A full two-process multiplayer race - one game hosting, another joining over
the network, both driven to the flag - was also run end to end; both machines
produced identical results.

## Known limitations

- **The cars are low-poly.** They are Kenney's CC0 Car Kit, the best freely
  licensed rally-style cars available; realistic free rally cars do not exist
  under CC0. The scenery, ground, roads and skies are photographic.
- **No car damage** - cars bounce off trees and each other but don't deform.
- **Internet play needs port forwarding** on the host's router; there is no
  matchmaking server.
- The first visit to each stage takes ~14 s while its sky lighting is
  computed and cached.
