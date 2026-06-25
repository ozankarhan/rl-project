"""Generate Ozan Karhan's mixed-quality offline-RL dataset for the joint Ch.20 task.

Reuses the team's canonical generator in drone_dispatch_env/offline.py (a mixed behavior
policy: ~60% greedy_nearest + ~40% noisy/eps-random over DroneDispatch-v0 on the standard
Config). That function writes the exact d4rl-style schema the team validator checks, so the
output conforms to the shared format by construction. Output is a compressed .npz.

    python gen_offline.py                          # -> offline_ozan_karhan.npz (~120k transitions)
    python gen_offline.py --out probe.npz --n 5000 # small probe to measure speed / size
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))

from drone_dispatch_env.config import Config                       # noqa: E402
from drone_dispatch_env.offline import generate_offline_dataset    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="offline_ozan_karhan.npz")
    ap.add_argument("--n", type=int, default=120_000,
                    help="minimum transitions to collect (team floor is 100k)")
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    t0 = time.time()
    # Standard Config() => obs flattens to 181 dims (8*10 drones + 20*5 orders + 1 time),
    # which is what the team format / checker expects. Do NOT pass a modified config here,
    # or the observation width changes and the dataset fails validation.
    generate_offline_dataset(args.out, config=Config(),
                             min_transitions=args.n, base_seed=args.base_seed)
    dt = time.time() - t0

    d = np.load(args.out)
    n = len(d["actions"])
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {n} transitions, {mb:.1f} MB, in {dt:.1f}s "
          f"({n / max(dt, 1e-9):.0f} transitions/s, {os.path.getsize(args.out) / n:.1f} bytes/tx)")


if __name__ == "__main__":
    main()
