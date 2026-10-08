"""SmolVLA closed-loop policy client (condensed `smolvla_policy_client.py`).

    connect -> handshake -> loop 300 steps:
        send joint_velocities, receive observation           (sim advances exactly 0.1 s)
        success check (optional early stop)
        decode 3 cameras, normalise state/effort (clamp to ±8 sigma, reject NaN)
        every `replan_interval` steps: predict a 50x4 action chunk + stage logits
        pick this step's action from the chunk (optionally temporal-ensembled)
        optional bounded supervisor
        log model vs executed action

The client applies NO gain or clipping to model actions; the only limit is the server's
joint-limit clip on the integrated position target. Model latency on an L40S: forward pass
median 2.37 ms, full client pipeline median 6.47 ms.

Run (illustrative, inside the `smolvla` env):
    python -m excavator_vla.policy.client --ckpt outputs/.../checkpoints/018450/pretrained_model \
        --vlm checkpoints/SmolVLM2-500M-Video-Instruct --replan-interval 3 --steps 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import numpy as np

from .. import contract as C
from ..protocol import build_handshake, decode_rgb, recv_msg, send_msg
from .ensemble import ChunkHistory, needs_replan
from .success import SuccessDetector
from .supervisor import ExcavationSupervisor

MAX_ABS_SIGMA = 8.0


def normalization_hash(ckpt: Path) -> str:
    h = hashlib.sha256()
    for name in ("policy_preprocessor_step_5_normalizer_processor.safetensors",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        p = ckpt / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def load_policy(ckpt: Path, vlm: Path, device: str = "cuda"):
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    policy.config.vlm_model_name = str(vlm)          # point at the local, offline VLM copy
    policy.to(device).eval()
    stats = {k: torch.as_tensor(v, device=device) for k, v in _load_stats(ckpt).items()}
    return policy, stats


def _load_stats(ckpt: Path) -> dict:
    from safetensors.numpy import load_file
    return load_file(str(ckpt / "policy_preprocessor_step_5_normalizer_processor.safetensors"))


def build_batch(reply: dict, stats: dict, task_tokens: dict, device: str = "cuda") -> dict:
    import torch

    def norm(x, key):
        z = (torch.as_tensor(x, device=device, dtype=torch.float32) - stats[f"{key}.mean"]) / (stats[f"{key}.std"] + 1e-8)
        if not torch.isfinite(z).all():
            raise ValueError(f"non-finite {key}")
        return z.clamp(-MAX_ABS_SIGMA, MAX_ABS_SIGMA)[None]   # clip & log out-of-distribution inputs

    batch = {
        "observation.state": norm(reply["observation_state"], "observation.state"),
        "observation.effort": norm(reply["observation_effort"], "observation.effort"),
        **task_tokens,
    }
    for cam, key in zip("012", C.CAMERA_KEYS):
        img = decode_rgb(reply["cameras"][cam])                  # 256x256x3 uint8
        batch[key] = torch.as_tensor(img, device=device).permute(2, 0, 1)[None].float() / 255.0
    return batch


def predict_chunk_and_stage(policy, batch, stats):
    """Flow-matching sampling: 50x4 normalised actions -> unnormalised rad/s + 50 stage ids.

    The original hand-writes the SmolVLA sampler so it can read the (patched-in) stage head
    on the suffix tokens at the final denoising step. Here we use the public entry point.
    """
    import torch
    with torch.inference_mode():
        out = policy.predict_action_chunk(batch)                # [1, 50, max_action_dim]
        stage_logits = getattr(policy, "last_stage_logits", None)
    act = out[0, :, :4] * stats["action.std"] + stats["action.mean"]
    stages = stage_logits[0].argmax(-1).cpu().numpy() if stage_logits is not None else np.zeros(len(act), int)
    return act.float().cpu().numpy(), stages


def run(args) -> dict:
    policy, stats = load_policy(Path(args.ckpt), Path(args.vlm))
    task_tokens = policy.tokenize_task(args.task) if hasattr(policy, "tokenize_task") else {"task": [args.task]}

    sock = socket.create_connection((args.host, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_msg(sock, build_handshake(C.POLICY_FPS, normalization_hash(Path(args.ckpt))))
    ack = recv_msg(sock)
    if not ack.get("ok"):
        raise RuntimeError(f"handshake rejected: {ack}")

    history = ChunkHistory(args.temporal_ensemble_width, args.temporal_ensemble_decay)
    supervisor = ExcavationSupervisor() if args.enable_supervisor else None
    success = SuccessDetector()
    chunk_origin, chunk_len, stages = None, 0, None
    cmd = np.zeros(4, dtype=np.float32)                         # first reply is "free"
    trace = open(args.trace_log, "w") if args.trace_log else None
    t_start = time.time()

    for step in range(args.steps):
        send_msg(sock, {"joint_velocities": cmd.tolist(), "record_overview_camera": args.record_overview})
        reply = recv_msg(sock)
        if reply.get("type") == "error":
            raise RuntimeError(reply["error"])

        q = reply["joint_positions"]
        if success.update(reply["truck_bed_load_metrics"]["count"], q[3]) and args.early_stop_on_success:
            break

        t0 = time.perf_counter()
        if needs_replan(step, args.replan_interval, chunk_origin, chunk_len):
            batch = build_batch(reply, stats, task_tokens)
            chunk, stages = predict_chunk_and_stage(policy, batch, stats)
            history.add(step, chunk)
            chunk_origin, chunk_len = step, len(chunk)
        model_action = history.action_at(step)
        executed = model_action
        reasons: list[str] = []
        if supervisor is not None:
            s = dict(zip(C.STATE_NAMES, reply["observation_state"]))
            obs = {"q": q,
                   "tip_above_dig_target": -s["dig_delta_up"],       # dig_delta = target - tip
                   "load_above_unload": -s["unload_delta_up"],       # unload_delta = landing - load point
                   "fill_fraction": s["bucket_fill_fraction"]}
            executed, reasons = supervisor(model_action, obs, int(stages[step - chunk_origin]))
        cmd = np.asarray(executed, dtype=np.float32)
        if not np.all(np.isfinite(cmd)):
            raise ValueError("non-finite action")

        if trace:
            trace.write(json.dumps({
                "step": step, "chunk_origin": chunk_origin, "model_action": model_action.tolist(),
                "executed_action": cmd.tolist(), "supervisor_reasons": reasons,
                "client_ms": 1e3 * (time.perf_counter() - t0), "server": reply.get("bridge_timing"),
            }) + "\n")

    send_msg(sock, {"joint_velocities": [0.0] * 4})              # always park the machine
    recv_msg(sock)
    return {"steps": step + 1, "success": success.update(reply["truck_bed_load_metrics"]["count"], q[3]),
            "wall_s": time.time() - t_start}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vlm", required=True)
    p.add_argument("--task", default="Excavate one scoop of sand from the sand pile in front of the "
                   "excavator's initial base pose, then carry and dump the collected material into the "
                   "truck bed to the right of the excavator's initial base pose.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--replan-interval", type=int, default=3)
    p.add_argument("--temporal-ensemble-width", type=int, default=1)
    p.add_argument("--temporal-ensemble-decay", type=float, default=0.35)
    p.add_argument("--enable-supervisor", action="store_true")
    p.add_argument("--early-stop-on-success", action="store_true")
    p.add_argument("--record-overview", action="store_true")
    p.add_argument("--trace-log", default=None)
    print(json.dumps(run(p.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
