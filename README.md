# IE 306 Term Project — Role B: Policy-Based Methods (REINFORCE + GAE → A2C, + DDPG)

**Owner:** Ozan Karhan · **Chapters 16–17** · operational drone dispatch inside the frozen
`drone_dispatch_env` simulator (**not modified**).

Role B owns the policy-gradient / actor–critic family: **REINFORCE + GAE → A2C** on the discrete,
action-masked dispatcher (`DroneDispatch-v0`) and **DDPG** on the continuous control sub-env
(`DroneControl-v0`). Project bar — *beat `greedy_nearest` on mean cost per delivered order, on the
instructor's held-out seeds/config.* **Met on all three seeds** (see Results).

## Quickstart — the grader runs exactly this
```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # the only setup step (installs the editable simulator + RL stack)
python run_all.py --config drone_dispatch_env/configs/eval_standard.yaml --seeds 0,1,2,3,4
```
`run_all.py` loads the saved weights and prints both baseline-comparison tables. `--config` and
`--seeds` are inputs you can change, so it runs unchanged on **held-out** seeds/config; nothing is
tuned to a specific seed. Add `--per-seed` for the ≥3-seed **mean ± std** view.

## Results (standard eval config, seeds 0–4)
| policy | cost_per_order ↓ | beats greedy? |
|--------|-----------------:|:-------------:|
| random | 18.50 | |
| greedy_nearest | **4.31** | — |
| milp_rolling | 4.28 | |
| REINFORCE + GAE | **2.57** | ✅ |
| A2C | **1.74** | ✅ |

3-seed robustness: REINFORCE **2.65 ± 0.09**, A2C **1.55 ± 0.21** — both beat greedy on every seed.
DDPG (control sub-env): return **+20.7 / success 1.00** vs go-straight **−417 / 0.80** → ✅.

## Deliverables → where they are (project spec §4 & §6)
| Required | Location |
|----------|----------|
| Single command + config | `run_all.py` · `configs/` |
| Trained weights + exact config | `weights/<method>.pt` (+ per-seed) · `configs/*.yaml` |
| Learning curves, ≥3 seeds, mean ± std | `logs/<method>_seed{0,1,2}.csv` · `figures/` · `run_all.py --per-seed` |
| Ablation (GAE-λ sweep, 3 seeds) | `configs/ablation_gae.yaml` · `code/role_b/run_ablation_gae.py` · `figures/ablation_gae.png` |
| Baseline table (random / greedy_nearest / milp_rolling) | `run_all.py` |
| Report (methods, results, ablation, citations, engineering log) | `REPORT_roleB.md` |
| Roles · AI-use declaration | `ROLES.md` · `AI_USE.md` |
| Reproducibility / held-out seeds | `run_all.py --config … --seeds …` (seed hygiene: `REPORT_roleB.md` §5) |
| Joint offline-RL dataset contribution | `gen_offline.py` · `offline_ozan_karhan.npz` · `check_offline_npz.py` |

## Retrain (optional)
```bash
# one command per method + its config
python code/role_b/train_reinforce.py --config configs/reinforce.yaml --seed 0
python code/role_b/train_a2c.py       --config configs/a2c.yaml       --seed 0
python code/role_b/train_ddpg.py      --config configs/ddpg.yaml      --seed 0
# all methods × 3 seeds, then the ablation + figures
python code/role_b/run_training.py     --methods reinforce,a2c,ddpg --seeds 0,1,2
python code/role_b/run_ablation_gae.py --config configs/ablation_gae.yaml
python code/role_b/plot_curves.py
```

## Layout
```
code/role_b/   features · networks (factored actor-critic + DDPG/TD3) · trainers · CLIs · plotting
configs/       reinforce · a2c · ddpg · ddpg_td3 · ablation_gae   (every hyperparameter — nothing buried in code)
weights/       trained models (one <method>.pt + per-seed files)
logs/          per-seed learning-curve CSVs
figures/       learning curves + ablation figure
run_all.py     prints the comparison tables (the grader runs this)
drone_dispatch_env/   the course simulator (installed via requirements.txt; NOT modified)
REPORT_roleB.md · ROLES.md · AI_USE.md
```
See **`REPORT_roleB.md`** for full method descriptions, the complete results, the ablation, the
half-page method-origin citations, and the "what broke & how we diagnosed it" engineering log.
