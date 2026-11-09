"""Wire format for multiplayer.

Every message is one JSON object, framed with a 4-byte big-endian length. The
`t` field names the message type. JSON is not the smallest encoding, but a
car-state message is ~120 bytes and eight cars at 20 Hz is ~20 KB/s, which is
nothing on a LAN - and every byte on the wire is readable when debugging.

Anything received from the network is untrusted: `FrameDecoder` caps frame
size, and the `clean_*` helpers coerce fields to the right type and range so a
malformed or hostile message can't crash the game or teleport a car to NaN.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any, Dict, List, Optional

from ..cars import CARS_BY_ID

PROTOCOL_VERSION = 1
DEFAULT_PORT = 47800
MAX_PLAYERS = 8
MAX_FRAME = 64 * 1024
STATE_HZ = 20

_HEADER = struct.Struct(">I")


class ProtocolError(Exception):
    pass


def encode(message: Dict[str, Any]) -> bytes:
    body = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(body) > MAX_FRAME:
        raise ProtocolError(f"message too large ({len(body)} bytes)")
    return _HEADER.pack(len(body)) + body


class FrameDecoder:
    """Accumulates bytes from a stream socket and yields whole messages."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> List[Dict[str, Any]]:
        self._buffer.extend(data)
        out = []
        while len(self._buffer) >= _HEADER.size:
            (length,) = _HEADER.unpack_from(self._buffer)
            if length > MAX_FRAME:
                raise ProtocolError(f"frame of {length} bytes exceeds the {MAX_FRAME} limit")
            if len(self._buffer) < _HEADER.size + length:
                break
            body = bytes(self._buffer[_HEADER.size:_HEADER.size + length])
            del self._buffer[:_HEADER.size + length]
            try:
                message = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ProtocolError(f"undecodable frame: {error}") from error
            if not isinstance(message, dict) or not isinstance(message.get("t"), str):
                raise ProtocolError("frame is not a typed message")
            out.append(message)
        return out


# --------------------------------------------------------------------------- #
# Validation of untrusted fields
# --------------------------------------------------------------------------- #
def clean_float(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(low, min(high, number))


def clean_int(value: Any, low: int, high: int, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def clean_name(value: Any, default: str = "Driver") -> str:
    if not isinstance(value, str):
        return default
    name = "".join(ch for ch in value if ch.isprintable()).strip()[:20]
    return name or default


def clean_car(value: Any) -> str:
    return value if isinstance(value, str) and value in CARS_BY_ID else "kestrel"


def clean_upgrades(value: Any) -> Dict[str, int]:
    from ..cars import UPGRADES_BY_ID
    if not isinstance(value, dict):
        return {}
    return {k: clean_int(v, 0, UPGRADES_BY_ID[k].max_level)
            for k, v in value.items() if k in UPGRADES_BY_ID}


WORLD_LIMIT = 2_000.0


def clean_state(value: Any) -> Optional[Dict[str, float]]:
    """A car state as sent by its owner, coerced to sane ranges."""
    if not isinstance(value, dict):
        return None
    return {
        "x": clean_float(value.get("x"), -WORLD_LIMIT, WORLD_LIMIT),
        "y": clean_float(value.get("y"), -500.0, 500.0),
        "z": clean_float(value.get("z"), -WORLD_LIMIT, WORLD_LIMIT),
        "yaw": clean_float(value.get("yaw"), -1e4, 1e4),
        "vl": clean_float(value.get("vl"), -100.0, 100.0),
        "vt": clean_float(value.get("vt"), -100.0, 100.0),
        "st": clean_float(value.get("st"), -1.0, 1.0),
        "rpm": clean_float(value.get("rpm"), 0.0, 1.2),
        "br": clean_float(value.get("br"), 0.0, 1.0),
        "pitch": clean_float(value.get("pitch"), -1.0, 1.0),
        "roll": clean_float(value.get("roll"), -1.0, 1.0),
    }


def state_of(vehicle, braking: float = 0.0) -> Dict[str, float]:
    """The compact state of a local car, for sending."""
    return {
        "x": round(vehicle.x, 3), "y": round(vehicle.y, 3), "z": round(vehicle.z, 3),
        "yaw": round(vehicle.yaw, 4), "vl": round(vehicle.v_long, 3),
        "vt": round(vehicle.v_lat, 3), "st": round(vehicle.steer_angle, 3),
        "rpm": round(vehicle.rpm, 3), "br": round(braking, 2),
        "pitch": round(vehicle.pitch, 3), "roll": round(vehicle.roll, 3),
    }
