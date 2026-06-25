# IE 306 Term Project — Drone Dispatch RL (Role B)

Policy-based methods for the operational drone-dispatch problem: **REINFORCE + GAE → A2C** on the
discrete masked dispatcher, plus **DDPG** on the continuous control sub-env. Built on the frozen
`drone_dispatch_env` simulator (not modified).

## Setup (one step)
```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt # installs the editable simulator + torch
```

## Reproduce the results tables
```bash
python run_all.py --config drone_dispatch_env/configs/eval_standard.yaml --seeds 0,1,2,3,4
```
Prints the dispatch comparison (vs random / greedy_nearest / milp_rolling; primary metric
`cost_per_order`, lower is better) and the DDPG-vs-go-straight table. `--config` and `--seeds`
are overridable, so this runs unchanged on a held-out config / held-out seeds. The learned
dispatch policy is dimension-robust and loads even if the config changes `k_max` / grid size.

## Retrain (optional)
```bash
python code/role_b/train_reinforce.py --config configs/reinforce.yaml --seed 0
python code/role_b/train_a2c.py       --config configs/a2c.yaml       --seed 0
python code/role_b/train_ddpg.py      --config configs/ddpg.yaml      --seed 0
# all 3 methods x 3 seeds in parallel:
python code/role_b/run_training.py --methods reinforce,a2c,ddpg --seeds 0,1,2
# GAE-lambda ablation + figures:
python code/role_b/run_ablation_gae.py --config configs/ablation_gae.yaml
python code/role_b/plot_curves.py
```

## Layout
```
code/role_b/      features, networks (factored actor-critic + DDPG), trainers, CLIs, plotting
configs/          reinforce.yaml, a2c.yaml, ddpg.yaml, ablation_gae.yaml  (every hyperparameter)
weights/          trained models (one <method>.pt per method, + per-seed files)
logs/             per-seed learning-curve CSVs
figures/          learning curves + ablation figure
run_all.py        prints the comparison tables (the grader runs this)
drone_dispatch_env/   the course simulator (installed via requirements.txt; NOT modified)
REPORT_roleB.md ROLES.md AI_USE.md
```

See `REPORT_roleB.md` for method descriptions, results, the ablation, citations, and the
engineering log.
