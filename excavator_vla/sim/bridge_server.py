"""Protocol-v2 Isaac Sim bridge server (sketch of `run_simulation.py`).

Threading model
---------------
Isaac Sim requires `world.step()` on the main thread. A network thread accepts a single TCP
client and forwards each request to the main loop through a queue; the main loop steps
physics and returns the observation through a second queue. The timeline is paused between
requests, which makes the loop *deterministic lockstep*: whether inference takes 3 ms or
300 ms, each action is applied for exactly 0.1 s of simulated time.

Per step
--------
    ticks = scheduler.next_ticks()                 # 6 at 60 Hz / 10 fps
    repeat ticks:
        q_target += v * dt ; clip to joint limits  # velocity command -> position target
        robot.apply_action(joint_positions=q_target)
        world.step(render=True)
    read q, dq, effort, bucket/truck particle counts
    render 3 cameras at 1024^2 -> LANCZOS downsample to 256^2 -> zlib+b64

Timing breakdown on an L40S (final report, Table 5-1): physics ~33 ms, 3-camera
render 200-400 ms (the bottleneck, 70-80 %), encoding 5-15 ms, TCP 2-5 ms,
model 2-3 ms  ->  2.1-4.0 Hz closed loop.

Launch (illustrative):
    /isaac-sim/python.sh -m excavator_vla.sim.bridge_server --scene-seed 2 --sand-amount 10
"""

from __future__ import annotations

import argparse
import queue
import socket
import threading
import time

import numpy as np

from .. import contract as C
from ..protocol import (PhysicsTickScheduler, build_handshake, encode_rgb, recv_msg,
                        resolve_canonical_dof_indices, send_msg, validate_handshake,
                        validate_step_request)

PHYSICS_DT = 1.0 / C.PHYSICS_HZ


def network_thread(host: str, port: int, cmd_q: queue.Queue, resp_q: queue.Queue):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[bridge] TCP server listening on {host}:{port}", flush=True)  # launchers wait for this line
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                cmd_q.put(recv_msg(conn))
                send_msg(conn, resp_q.get())          # strict request/reply
        except ConnectionError:
            print(f"[bridge] client {addr} disconnected", flush=True)


class BridgeServer:
    def __init__(self, world, robot, cameras, scene, normalization_hash: str = ""):
        self.world, self.robot, self.cameras, self.scene = world, robot, cameras, scene
        self.scheduler = PhysicsTickScheduler(C.POLICY_FPS)
        self.canonical_idx = resolve_canonical_dof_indices(list(robot.dof_names))
        lim = np.radians([C.JOINT_LIMITS_DEG[j] for j in C.DOF_ORDER])
        self.lo, self.hi = lim[:, 0], lim[:, 1]
        self.q_target = robot.get_joint_positions()[self.canonical_idx].astype(np.float64)
        self.fill_rate = C.FillRateEMA()
        self.normalization_hash = normalization_hash
        self.handshake_done = False

    # -------------------------------------------------------------- request handlers

    def handle(self, msg: dict) -> dict:
        try:
            if msg.get("type") == "handshake":
                return self._handshake(msg)
            if not self.handshake_done:
                return {"type": "error", "error": "handshake required"}
            return self._step(validate_step_request(msg), bool(msg.get("record_overview_camera")))
        except Exception as exc:                      # fail closed, never move on bad input
            return {"type": "error", "error": f"{type(exc).__name__}: {exc}"}

    def _handshake(self, msg: dict) -> dict:
        validate_handshake(msg)
        if self.normalization_hash and msg["normalization_hash"] != self.normalization_hash:
            raise ValueError("normalization statistics differ from the deployment manifest")
        self.scene.latch_initial_context()            # base pose, truck yaw, initial-pile particle mask
        self.handshake_done = True
        ack = build_handshake(C.POLICY_FPS, msg["normalization_hash"])
        ack.update(type="handshake_ack", ok=True, physics_hz=C.PHYSICS_HZ,
                   timing_mode="deterministic_lockstep",
                   raw_dof_names=list(self.robot.dof_names),
                   canonical_dof_names=list(C.DOF_ORDER),
                   canonical_to_raw=self.canonical_idx)
        return ack

    def _step(self, vel: np.ndarray, record_overview: bool) -> dict:
        from isaacsim.core.utils.types import ArticulationAction  # Isaac Sim 4.x

        timing = {}
        t0 = time.perf_counter()
        ticks = self.scheduler.next_ticks()
        self.world.play()
        try:
            for _ in range(ticks):
                self.q_target = np.clip(self.q_target + vel * PHYSICS_DT, self.lo, self.hi)
                self.robot.apply_action(ArticulationAction(
                    joint_positions=self.q_target.astype(np.float32),
                    joint_indices=np.asarray(self.canonical_idx)))
                self.world.step(render=True)
        finally:
            self.world.pause()
        timing["physics_ms"] = 1e3 * (time.perf_counter() - t0)

        t1 = time.perf_counter()
        cams = {name: encode_rgb(cam.get_rgb_256()) for name, cam in self.cameras.items() if name in "012"}
        rec = {"overview": encode_rgb(self.cameras["overview"].get_rgb_native())} if record_overview else {}
        timing["camera_encode_ms"] = 1e3 * (time.perf_counter() - t1)

        bucket_n = self.scene.bucket_particle_count()   # particles inside the closed `bucket_cut` mesh
        # tracking error and previous action use the commanded target / velocity of this step
        fill = bucket_n / C.BUCKET_CAPACITY_PARTICLES
        state = self.scene.build_state_28d(self.robot, self.q_target, vel, bucket_n,
                                           self.fill_rate.update(fill, ticks * PHYSICS_DT))
        timing["total_server_ms"] = 1e3 * (time.perf_counter() - t0)

        return {
            "type": "observation",
            "observation_state": list(map(float, state)),
            "observation_state_schema": C.STATE_SCHEMA,
            "observation_effort": list(map(float, self.robot.get_measured_joint_efforts()[self.canonical_idx])),
            "joint_positions": list(map(float, self.robot.get_joint_positions()[self.canonical_idx])),
            "bucket_load_metrics": {"count": int(bucket_n)},
            "truck_bed_load_metrics": {"count": int(self.scene.truck_bed_new_particles())},
            "cameras": cams,
            "recording_cameras": rec,
            "task_text": self.scene.task_text,
            "bridge_timing": {**timing, "physics_ticks": ticks, "simulated_seconds": ticks * PHYSICS_DT},
        }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--scene-seed", type=int, default=2)
    p.add_argument("--sand-amount", type=float, default=10.0)
    p.add_argument("--idle-ui-hz", type=float, default=5.0)
    a = p.parse_args(argv)

    # Isaac Sim must be started before any omni.* import
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    from .scene import build_scene   # loads USD, excavator, truck, particle sand, cameras

    world, robot, cameras, scene = build_scene(seed=a.scene_seed, sand_amount=a.sand_amount)
    server = BridgeServer(world, robot, cameras, scene)

    cmd_q, resp_q = queue.Queue(), queue.Queue()
    threading.Thread(target=network_thread, args=(a.host, a.port, cmd_q, resp_q), daemon=True).start()

    while app.is_running():
        try:
            msg = cmd_q.get(timeout=1.0 / a.idle_ui_hz)
        except queue.Empty:
            app.update()                               # keep the UI alive, timeline stays paused
            continue
        resp_q.put(server.handle(msg))
    app.close()


if __name__ == "__main__":
    main()
