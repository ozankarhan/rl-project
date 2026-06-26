# Role B — Policy-Based Methods (REINFORCE+GAE → A2C, + DDPG)

IE 306 Term Project — operational drone dispatch. This document covers the **Role B**
deliverables; it is written to fold into the team `REPORT.docx`.

## 1. Problem and success criterion

The operational layer decides, in real time, which drone serves which order and when drones
charge, inside the frozen `drone_dispatch_env` simulator. Role B owns the **policy-gradient /
actor–critic** family:

- **REINFORCE + GAE → A2C** on the discrete, action-masked dispatcher `DroneDispatch-v0`.
- **DDPG** on the continuous single-drone control sub-env `DroneControl-v0`.

**Primary metric:** `cost_per_order = (energy + late + drop + depletion costs) / orders_delivered`
on `DroneDispatch-v0` — **lower is better**. The bar for the dispatch methods is to beat
`greedy_nearest`. DDPG (a different env with no greedy baseline) is graded against a
**go-straight** controller.

A subtlety we exploit throughout: `cost_per_order` is **not** the RL return (the return adds the
+10 delivered / +5 on-time bonuses that the cost metric excludes). We therefore **select model
checkpoints on validation `cost_per_order`, not on return.**

## 2. Baselines (standard config, seeds 0–4)

| policy | cost_per_order | success | on-time | depletions/ep | dropped/ep | delivered/ep |
|--------|---------------:|--------:|--------:|--------------:|-----------:|-------------:|
| random | 18.498 | 0.659 | 0.897 | 8.00 | 21.2 | 40.0 |
| **greedy_nearest** | **4.309** | 0.858 | 0.906 | 3.60 | 19.8 | 120.0 |
| milp_rolling | 4.282 | 0.853 | 0.910 | 3.20 | 20.6 | 120.6 |

**Where the headroom is:** even the strong `greedy_nearest` depletes a drone **3.6 times per
episode** (each depletion costs +50) and drops ~20 orders/episode. Its charge rule is a fixed
0.30-SoC threshold and it never reasons about whether a drone can *finish* a job before its
battery dies. A policy that charges proactively and refuses battery-infeasible assignments can
remove most of those +50 depletion hits — this is the main lever Role B targets.

For the control sub-env, the go-straight baseline scores **return −417 (seeds 0–4) / −891
(seeds 200–204)** with 60–80% success: it reaches targets but repeatedly crashes into no-fly
cells (each collision −25), so its return is deeply negative.

## 3. Methods

### 3.1 Shared representation — a dimension-robust factored policy
The action space scales with the config: `n_actions = n_drones·k_max + n_drones + 1` (= 169 on
the standard config). Because grading uses a **held-out config we have not seen**, a flat MLP with
a 169-wide output layer would fail to load if `k_max` or the grid size change. We therefore use a
**factored, permutation-invariant** policy whose parameters do not depend on the action count:

- **Actor:** a shared `assign_head` scores every (drone, order) candidate from
  `[drone features, order features, routed distance, deadline-feasibility]`; a shared `charge_head`
  scores each drone; a state-conditioned `noop` bias. Logits are concatenated in action-index
  order, the env action mask sets invalid logits to −1e8, then softmax.
- **Critic:** a Deep Sets encoder — shared per-drone and per-order MLPs, masked mean+max pooling,
  concatenated with global features → V(s). Also dimension-agnostic.

**Routed-distance features.** For each (drone, order) we feed the **no-fly-aware BFS routed
distance** (drone→pickup, pickup→dropoff trip length), the steps-to-deadline, and battery- and
deadline-feasibility flags — computed only from `obs["grid"]` via a self-contained BFS router
(`code/role_b/routing.py`, with no dependency on simulator internals), memoised once per source
cell per episode. This gives the policy the *same* distance
information `greedy_nearest` uses, which is what makes beating it feasible.

### 3.2 REINFORCE + GAE
Monte-Carlo policy gradient with a learned value baseline; the return is replaced by a GAE(λ)
advantage. Collect complete episodes, compute GAE per episode, one gradient step per batch with a
value (Huber) term and an entropy bonus.

**Behavior-cloning warm-start.** Plain REINFORCE is high-variance: from a random init it
(seed-dependently) gets stuck in a no-/half-delivery local optimum, so only ~1/3 of seeds reach a
greedy-beating policy. We fix this by first **cloning `greedy_nearest`** into the policy (a short
supervised phase — cross-entropy to greedy's masked action over a fixed greedy dataset), which
puts *every* seed in the delivering regime; REINFORCE then reliably refines the policy **below**
greedy. With the clone plus lr decay and an extra critic-fit per update, all three seeds converge
to cost ≈ 2.6 (§6).

### 3.3 A2C
The synchronous advantage actor–critic: identical net, but instead of full episodes it collects
fixed n-step rollouts and **bootstraps** the advantage with the critic's value of the state after
the rollout (GAE-λ). Lower variance, more sample-efficient.

### 3.4 DDPG (continuous control sub-env)
Deterministic actor (speed∈[0.3,1] via sigmoid+floor, heading-delta∈[−1,1] via tanh) + Q-critic,
target networks with Polyak averaging, a replay buffer, Ornstein–Uhlenbeck exploration that decays
over training, and a random warm-up phase. **Checkpoints are selected on validation success-rate
(then return)**, so the policy that actually reaches the target is the one we keep. We also
implement the **TD3** stabilisation (clipped double-Q, target-policy smoothing, delayed actor
updates; Fujimoto et al. 2018) as a config-gated option for seed robustness. Graded against
go-straight on return / success.

## 4. Method-origin note (citations)

- **REINFORCE** — Williams (1992), *Simple statistical gradient-following algorithms for
  connectionist reinforcement learning*. Chosen as the canonical likelihood-ratio policy gradient
  and the conceptual base of the family.
- **GAE** — Schulman et al. (2016), *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation*. Chosen for the explicit bias/variance λ knob — which is exactly our
  ablation.
- **A2C** — Mnih et al. (2016), *Asynchronous Methods for Deep RL* (A3C); we use the synchronous
  A2C variant for reproducibility on a single machine.
- **DDPG** — Lillicrap et al. (2016), *Continuous control with deep reinforcement learning*.
  Chosen because the control sub-env has a genuinely continuous action (speed, heading), which is
  DDPG's native setting.
- **TD3** — Fujimoto et al. (2018), *Addressing Function Approximation Error in Actor-Critic
  Methods*. Added as a config-gated stabiliser (clipped double-Q, target-policy smoothing, delayed
  actor updates) to reduce DDPG's across-seed variance.

## 5. Experimental setup

- One command per method + a YAML config, e.g.
  `python code/role_b/train_a2c.py --config configs/a2c.yaml --seed 0`.
- **Seed hygiene / generalization:** training draws fresh episode seeds from a high disjoint pool
  (≥10 000); checkpoints are selected on **validation seeds 200–202**; the reported tables use
  `--seeds` (default 0–4). The dispatch hyperparameters are shared across all seeds; DDPG's exploration noise is the one seed-sensitive exception (§7). Grading runs on held-out
  seeds/config via `run_all.py --config ... --seeds ...`.
- ≥3 seeds per method; learning curves are mean ± std (see `figures/`).
- Reproduce everything: `python run_all.py --config drone_dispatch_env/configs/eval_standard.yaml --seeds 0,1,2,3,4`.

## 6. Results — dispatch (standard eval config, seeds 0–4)

Reproduce: `python run_all.py --config drone_dispatch_env/configs/eval_standard.yaml --seeds
0,1,2,3,4`. Primary metric `cost_per_order` (lower better). The reported model per method is the
best of 3 seeds, selected on validation seeds 200–202.

| policy | cost_per_order | success | on-time | depletions/ep | delivered/ep | dropped/ep |
|--------|---------------:|--------:|--------:|--------------:|-------------:|-----------:|
| random | 18.498 | 0.659 | 0.897 | 8.00 | 40.0 | 21.2 |
| greedy_nearest | 4.309 | 0.858 | 0.906 | 3.60 | 120.0 | 19.8 |
| milp_rolling | 4.282 | 0.853 | 0.910 | 3.20 | 120.6 | 20.6 |
| **REINFORCE+GAE** | **2.565** | 0.907 | 0.941 | 1.80 | 128.8 | 13.4 |
| **A2C** | **1.735** | 0.955 | 0.847 | 1.60 | 134.0 | 6.4 |

Both policy-gradient methods beat greedy_nearest and MILP. **A2C wins decisively (1.735 — a 60%
lower cost per order than greedy):** it delivers *more* orders (134 vs 120), drops 3× fewer
(6.4 vs 19.8), and cuts the depletions that dominate greedy's cost (1.6 vs 3.6) — exactly the
headroom identified in §2. **REINFORCE+GAE also beats greedy decisively (2.565)** and — thanks to
the behavior-cloning warm-start (§3.2) — now does so *reliably on every seed* (2.65 ± 0.09, §9),
not just its best one. A2C still reaches the lowest cost with the least compute — the textbook
payoff of bootstrapping over high-variance Monte-Carlo returns.

**Seed robustness: mean ± std over the 3 trained seeds (eval seeds 0-4)**, reproduced with
`run_all.py --per-seed`. The headline table reports the *best* of 3 seeds (selected on validation);
the full spread is:

| method | cost_per_order (mean ± std) | per-seed |
|--------|----------------------------:|----------|
| REINFORCE+GAE | 2.65 ± 0.09 | 2.77 / 2.61 / 2.57 |
| A2C | 1.55 ± 0.21 | 1.26 / 1.64 / 1.74 |

Both methods now beat greedy (4.31) on **all three seeds**. The behavior-cloning warm-start turned
REINFORCE from a 1/3-reliable, high-variance learner (previously **48.98 ± 50.36**, per-seed
25.3 / 119.0 / 2.6) into a tight **2.65 ± 0.09**; A2C remains the most sample-efficient.

**Robustness to a held-out config.** Because the policy is factored / permutation-invariant, the
*same* standard-config weights load and run on a structurally different config — verified on a
24×24 grid with k_max=28 (vs trained 20×20 / k_max=20) via
`run_all.py --config configs/eval_stress.yaml`. On held-out *seeds* of the training config it beats
greedy (table above); on the much harder stress config it still runs end-to-end but trails greedy
(it is trained only on the standard distribution) — an honest generalization limit. Notably DDPG
generalizes better there (return −44.6 vs go-straight −1511).

## 7. Results — DDPG vs go-straight (DroneControl-v0, seeds 0–4)

| policy | return | success_rate | mean_steps |
|--------|-------:|-------------:|-----------:|
| go_straight | −417.4 | 0.80 | 28.4 |
| **DDPG** (best seed) | **+20.7** | **1.00** | 17.0 |

DDPG now **beats go-straight on both metrics decisively**: it reaches the target on **every** eval
episode (success 1.00 vs 0.80) while go-straight crashes into no-fly cells (−25 each), so DDPG's
return is *positive* (+20.7) against go-straight's −417.4. Selecting the checkpoint on validation
**success** (§3.4) rather than return is what captures the target-reaching policy — it appears late
in training and was previously discarded.

**Seed robustness: mean ± std over the 3 trained seeds (eval seeds 0-4)**, via `run_all.py
--per-seed`: **success 1.00 ± 0.00** (per-seed 1.00 / 1.00 / 1.00 — every seed reaches the target
on every held-out episode), return **−21.4 ± 62.6** (per-seed +24.9 / −109.8 / +20.7; all far above
go-straight's −417.4). DDPG is exploration-noise sensitive across seeds (a known DDPG trait, §9):
seed 0 trains best with lower OU noise (0.2), seeds 1–2 with higher (0.3); the TD3 variant reaches
the same 1.00 success. All three checkpoints generalise to 100% success on the held-out eval.

## 8. Ablation — GAE λ sweep on A2C (required)

We sweep λ ∈ {0.0, 0.9, 0.95, 0.99, 1.0} on A2C (3 seeds each, 40k steps). We run the sweep on A2C
rather than REINFORCE because A2C is stable, so the λ effect is not swamped by Monte-Carlo seed
variance. Best validation `cost_per_order` per λ (`figures/ablation_gae.png`):

| GAE λ | mean best cost_per_order |
|-------|-------------------------:|
| 0.0 | 13.85 |
| 0.9 | 0.75 |
| 0.95 | **0.69** |
| 0.99 | 0.80 |
| 1.0 | 1.11 |

This is the textbook bias/variance tradeoff. **λ = 0** uses the one-step TD advantage
`r + γV(s′) − V(s)` — minimal variance but maximal bias, so credit is assigned too myopically and the
policy never beats greedy (13.8 ≫ 4.31). **λ = 0.9–0.95** balances bias and variance and is optimal
(≈0.69). Pushing to **λ = 1.0** (the Monte-Carlo advantage — unbiased but high-variance) slightly
degrades it (1.11). This validates the GAE(0.95) default used for the headline A2C/REINFORCE runs.

## 9. Engineering log — what broke and how we diagnosed it

- **Held-out-config fragility (caught in design review).** A flat 169-wide policy head would not
  load if the grading config changed `k_max`/grid size → zero score. Diagnosed by reading the
  action-index math in `config.py`; fixed with the factored/Deep-Sets architecture (§3.1), which
  we verify loads under a modified config.
- **Namespace-package shadowing.** Running from the repo root, `from drone_dispatch_env import
  evaluate` failed ("unknown location") because the vendored `drone_dispatch_env/` folder shadows
  the installed package as a PEP-420 namespace (the top-level `__init__` re-exports don't run).
  Diagnosed from the `unknown location` import error; fixed by importing submodules directly
  (`from drone_dispatch_env.evaluate import evaluate`), which we use everywhere.
- **Throughput / CPU contention.** The dispatcher runs at ~88 env-steps/s (the auto-advance ticks
  + BFS dominate), dropping to ~32/s under 9 parallel processes. Diagnosed with a timed probe;
  addressed by pinning 1 torch thread per process, vectorising the feature extractor, shrinking
  the net to hidden=64, and right-sizing the step budgets.
- **Value-function divergence — the key bug.** Early A2C/REINFORCE delivered almost nothing and
  depleted all 8 drones every episode (cost_per_order ≈ 14–50). The smoking gun was in the logs:
  `value_loss ≈ 25 000`. With unscaled rewards (+10 delivered, −50 depletion) over a
  hundreds-of-steps horizon, the critic's value targets were huge, so the value-loss gradient
  dominated the shared objective and drowned the policy gradient — the policy never learned to
  charge. Fix: scale rewards ×0.1 and use a Huber (smooth-L1) value/critic loss. `value_loss`
  dropped to ≈ 5 and A2C immediately learned to charge (zero depletions) and beat greedy within
  ~10k steps. This single fix turned a non-working agent into the headline result.
- **DDPG "don't-move" optimum, success-selection, and seed-variance.** DDPG first converged to
  barely moving (0% success): with a −25 wall penalty, hovering beats risky movement and i.i.d.
  noise on a *turn-rate* action just spins in place. Fixes: reward-scaling (the Q-targets were also
  exploding), Ornstein–Uhlenbeck temporally-correlated exploration (decaying over training), and a
  minimum-speed floor (0.3). The decisive reporting fix was **selecting the checkpoint on validation
  success rather than return** — the high-success policy appears late in training and was previously
  thrown away. DDPG stays exploration-noise sensitive across seeds (low noise suits seed 0, higher
  noise seeds 1–2), so we also implemented **TD3** (clipped double-Q + target smoothing + delayed
  updates) for robustness. Final: all three seeds reach **100% success** on held-out eval with
  returns far above go-straight (§7).
- **REINFORCE seed-variance — and the fix.** Plain REINFORCE+GAE beat greedy on only its best seed
  (per-seed eval 25.3 / 119.0 / 2.6 → **48.98 ± 50.36**): two seeds got stuck in a no-/half-delivery
  local optimum — the policy never commits (entropy stays high) because the Monte-Carlo advantages
  are too noisy to identify good actions. Sweeping lr / batch / entropy / value-fits / lr-decay
  across ~16 configs did **not** make it reliable. The fix that did: **behavior-clone
  `greedy_nearest` first** (§3.2) so every seed starts already delivering, then let REINFORCE refine
  below greedy. Result: all three seeds converge to **2.65 ± 0.09**. A2C needs no warm-start (its
  bootstrapped advantages are low-variance) — the lesson of the REINFORCE→A2C progression.
