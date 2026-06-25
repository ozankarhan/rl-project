"""Reproduce the Role B results tables.

Loads the trained policies and prints (1) the dispatch comparison table vs the
random / greedy_nearest / milp_rolling baselines on DroneDispatch-v0 (primary
metric: cost_per_order, lower is better), and (2) the DDPG vs GoStraight table on
the continuous DroneControl-v0 sub-env.

    python run_all.py --config drone_dispatch_env/configs/eval_standard.yaml --seeds 0,1,2,3,4

Both --config and --seeds are overridable so this runs unchanged on a held-out
config / held-out seeds. The learned dispatch net is dimension-robust, so the same
weights load even if the config changes k_max / grid size.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))

from drone_dispatch_env.config import Config                      # noqa: E402
from drone_dispatch_env.evaluate import evaluate                  # noqa: E402
from drone_dispatch_env.baselines import make_baseline           # noqa: E402
from drone_dispatch_env.env_control import DroneControlEnv        # noqa: E402

from role_b.adapters import load_dispatch_agent, GoStraight       # noqa: E402
from role_b.networks import DDPGActor                             # noqa: E402
from role_b.ddpg import _clip_action                              # noqa: E402

DISPATCH_COLS = ["cost_per_order", "success_rate", "ontime_rate",
                 "depletion_events", "energy_per_order", "n_delivered",
                 "n_dropped", "episode_return"]


def _canonical(method: str) -> str | None:
    """Return weights/<method>.pt, creating it from the best seed file if absent."""
    canon = f"weights/{method}.pt"
    if os.path.exists(canon):
        return canon
    seedfiles = sorted(glob.glob(f"weights/{method}_seed*.pt"))
    if not seedfiles:
        return None
    key = "return" if method == "ddpg" else "cost_per_order"
    lower_better = method != "ddpg"
    best, best_val = None, None
    for f in seedfiles:
        ck = torch.load(f, map_location="cpu")
        val = ck.get(key) if isinstance(ck, dict) else None
        if val is None:
            continue
        if best_val is None or (val < best_val if lower_better else val > best_val):
            best_val, best = val, f
    best = best or seedfiles[0]
    shutil.copyfile(best, canon)
    print(f"  [selected {os.path.basename(best)} -> {canon}]")
    return canon


def dispatch_table(cfg: Config, seeds: list[int]) -> dict:
    rows = {}
    for name in ["random", "greedy_nearest", "milp_rolling"]:
        rows[name] = evaluate(make_baseline(name, cfg), cfg, seeds)["mean"]
    for method in ["reinforce", "a2c"]:
        canon = _canonical(method)
        if canon is None:
            print(f"  [no weights for {method} yet - skipping]")
            continue
        agent = load_dispatch_agent(canon, cfg)
        rows[method] = evaluate(agent, cfg, seeds)["mean"]
    return rows


@torch.no_grad()
def _run_control(act_fn, cfg: Config, seeds: list[int]) -> dict:
    env = DroneControlEnv(cfg)
    rets, succ, steps = [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done, ret, n, last_term = False, 0.0, 0, False
        while not done:
            obs, r, term, trunc, _ = env.step(_clip_action(act_fn(obs)))
            ret += r
            n += 1
            last_term = term
            done = term or trunc
        rets.append(ret)
        succ.append(1.0 if (last_term and obs[2] > 0.0) else 0.0)
        steps.append(n)
    return {"return": float(np.mean(rets)), "success_rate": float(np.mean(succ)),
            "mean_steps": float(np.mean(steps))}


def control_table(cfg: Config, seeds: list[int]) -> dict:
    rows = {"go_straight": _run_control(GoStraight(cfg).act, cfg, seeds)}
    canon = _canonical("ddpg")
    if canon is not None:
        ck = torch.load(canon, map_location="cpu")
        actor = DDPGActor(ck["obs_dim"], ck.get("hidden", 256))
        actor.load_state_dict(ck["state_dict"])
        actor.eval()
        act_fn = lambda o: actor(torch.as_tensor(o, dtype=torch.float32)).numpy()
        rows["ddpg"] = _run_control(act_fn, cfg, seeds)
    else:
        print("  [no weights for ddpg yet - skipping]")
    return rows


def _print_dispatch(rows: dict):
    print("\n=== DroneDispatch-v0 : cost_per_order is PRIMARY (lower is better) ===")
    head = f"{'policy':<16}" + "".join(f"{c[:11]:>13}" for c in DISPATCH_COLS)
    print(head)
    print("-" * len(head))
    for name, m in rows.items():
        line = f"{name:<16}" + "".join(f"{m[c]:>13.3f}" for c in DISPATCH_COLS)
        print(line)
    if "greedy_nearest" in rows:
        g = rows["greedy_nearest"]["cost_per_order"]
        print(f"\nTarget to beat (greedy_nearest cost_per_order): {g:.3f}")
        for m in ("reinforce", "a2c"):
            if m in rows:
                c = rows[m]["cost_per_order"]
                verdict = "BEATS greedy" if c < g else "does NOT beat greedy"
                print(f"  {m:<10} cost_per_order = {c:.3f}  -> {verdict}")


def _print_control(rows: dict):
    print("\n=== DroneControl-v0 : DDPG vs GoStraight (higher return / success better) ===")
    head = f"{'policy':<14}{'return':>12}{'success_rate':>14}{'mean_steps':>12}"
    print(head)
    print("-" * len(head))
    for name, m in rows.items():
        print(f"{name:<14}{m['return']:>12.2f}{m['success_rate']:>14.3f}{m['mean_steps']:>12.1f}")
    if "ddpg" in rows and "go_straight" in rows:
        d, gs = rows["ddpg"], rows["go_straight"]
        verdict = "BEATS go-straight" if d["return"] > gs["return"] else "does NOT beat go-straight"
        print(f"\n  DDPG return {d['return']:.2f} vs go-straight {gs['return']:.2f} -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="drone_dispatch_env/configs/eval_standard.yaml")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    print(f"config: {args.config}   seeds: {seeds}")

    _print_dispatch(dispatch_table(cfg, seeds))
    _print_control(control_table(cfg, seeds))


if __name__ == "__main__":
    main()
