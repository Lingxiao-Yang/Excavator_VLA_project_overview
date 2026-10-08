"""Protocol v2: the simulator <-> policy bridge.

Transport : plain TCP (default 127.0.0.1:5555), one client at a time
Framing   : 4-byte big-endian length prefix + UTF-8 JSON body (max 128 MB)
Images    : uint8 HxWx3 -> zlib -> base64 string
Timing    : deterministic lockstep. The sim timeline is *paused* between requests,
            so model latency never shows up as simulated time.

    client                                   server (Isaac Sim)
      |-- handshake {schema, names, fps, normalization_hash} -->|
      |<------------------- handshake_ack {canonical_to_raw} ----|
      |-- step {joint_velocities[4]} --------------------------->|  6 x (q_tgt += v*dt; world.step)
      |<--- obs {state[28], effort[4], cameras{0,1,2}, loads} ---|
      ...                                                         (repeat 300-350x)

A step request may carry only `joint_velocities` (rad/s, canonical order). Fields from the
earlier tick-based protocol (`ticks`, `joint_positions`, `joint_efforts`) are rejected, so an
outdated client can never silently drive the server.
"""

from __future__ import annotations

import base64
import json
import math
import socket
import struct
import zlib
from typing import Any

import numpy as np

from . import contract as C

HEADER = struct.Struct("!I")
MAX_MESSAGE_BYTES = 128 * 1024 * 1024
FORBIDDEN_STEP_FIELDS = ("ticks", "joint_positions", "joint_efforts")


# ---------------------------------------------------------------- framing

def send_msg(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(HEADER.pack(len(body)) + body)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("bridge peer closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    (length,) = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(f"bridge message too large: {length} bytes")
    return json.loads(_recv_exact(sock, length).decode("utf-8"))


def encode_rgb(img: np.ndarray) -> dict[str, Any]:
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return {
        "rgb_shape": list(img.shape),
        "rgb_dtype": "uint8",
        "rgb_zlib_b64": base64.b64encode(zlib.compress(img.tobytes(), 1)).decode("ascii"),
    }


def decode_rgb(msg: dict[str, Any]) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(msg["rgb_zlib_b64"]))
    return np.frombuffer(raw, dtype=np.uint8).reshape(msg["rgb_shape"])


# ---------------------------------------------------------------- handshake

def build_handshake(training_fps: float, normalization_hash: str) -> dict[str, Any]:
    return {
        "type": "handshake",
        "protocol_version": C.PROTOCOL_VERSION,
        "training_fps": float(training_fps),
        "observation_schema": C.STATE_SCHEMA,
        "state_names": list(C.STATE_NAMES),
        "effort_names": list(C.EFFORT_NAMES),
        "action_names": list(C.ACTION_NAMES),
        "action_units": "rad/s",
        "camera_keys": list(C.CAMERA_KEYS),
        "camera_shape": [1, *C.CAMERA_SHAPE],
        # sha256 of the normalizer/unnormalizer safetensors: catches "right model, wrong stats"
        "normalization_hash": normalization_hash,
    }


def validate_handshake(msg: dict[str, Any]) -> None:
    if msg.get("protocol_version") != C.PROTOCOL_VERSION:
        raise ValueError(f"protocol mismatch: {msg.get('protocol_version')} != {C.PROTOCOL_VERSION}")
    if msg.get("observation_schema") != C.STATE_SCHEMA:
        raise ValueError(f"unsupported observation schema {msg.get('observation_schema')!r}")
    C.validate_feature_names("state", C.STATE_NAMES, msg["state_names"])
    C.validate_feature_names("effort", C.EFFORT_NAMES, msg["effort_names"])
    C.validate_feature_names("action", C.ACTION_NAMES, msg["action_names"])
    if not math.isclose(float(msg["training_fps"]), C.POLICY_FPS):
        raise ValueError("training fps differs from the simulator control interval")


def validate_step_request(msg: dict[str, Any]) -> np.ndarray:
    bad = [k for k in FORBIDDEN_STEP_FIELDS if k in msg]
    if bad:
        raise ValueError(f"protocol v2 step rejects legacy fields {bad}")
    vel = np.asarray(msg["joint_velocities"], dtype=np.float64)
    if vel.shape != (len(C.DOF_ORDER),) or not np.all(np.isfinite(vel)):
        raise ValueError("joint_velocities must be 4 finite floats in canonical order")
    return vel


def resolve_canonical_dof_indices(raw_dof_names: list[str]) -> list[int]:
    """Map canonical [swing, boom, arm, bucket] onto the articulation's raw DOF order by name."""
    lookup = {name.lower(): i for i, name in enumerate(raw_dof_names)}
    out = []
    for j in C.DOF_ORDER:
        hits = [i for name, i in lookup.items() if j in name]
        if len(hits) != 1:
            raise ValueError(f"cannot uniquely resolve joint {j!r} in {raw_dof_names}")
        out.append(hits[0])
    return out


# ---------------------------------------------------------------- tick scheduling

class PhysicsTickScheduler:
    """Distribute physics ticks across an arbitrary policy FPS without drift.

    60 Hz / 10 fps -> exactly 6 ticks each step. For FPS values that don't divide 60
    (e.g. 7 fps -> 8.571 ticks), the fractional remainder is carried forward so the
    long-run simulated time matches wall-clock policy time exactly.
    """

    def __init__(self, policy_fps: float, physics_hz: float = C.PHYSICS_HZ):
        if policy_fps <= 0 or policy_fps > physics_hz:
            raise ValueError("policy fps must be in (0, physics_hz]")
        self._exact = physics_hz / policy_fps
        self._remainder = 0.0

    def next_ticks(self) -> int:
        target = self._exact + self._remainder
        ticks = max(1, int(math.floor(target + 1e-9)))
        self._remainder = target - ticks
        return ticks
