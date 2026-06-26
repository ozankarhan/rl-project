# AI Tool Usage Declaration (Role B)

**Author:** Ozan Karhan (Role B — policy-based methods)

Per project rules (§2), AI tools may be used freely; this is an honest, complete declaration, not
a penalty. I directed the work and can explain, modify, and defend every line of the Role B code.

## Tools used
- **Claude Code (Anthropic)** — the only AI tool used for Role B. Used as a pair-programmer, under
  my direction, to scaffold, implement, debug, and document every Role B component:
  - `code/role_b/features.py` — observation → factored per-entity feature tensors
  - `code/role_b/routing.py` — self-contained no-fly-aware BFS distance field (no simulator internals)
  - `code/role_b/networks.py` — factored actor-critic + Deep-Sets critic + DDPG actor/critic
  - `code/role_b/reinforce_gae.py`, `a2c.py`, `ddpg.py` — the three training loops
  - `code/role_b/rollout.py`, `utils.py`, `adapters.py` — rollout/replay, helpers, Policy adapters
  - `code/role_b/train_*.py`, `run_training.py`, `run_ablation_gae.py` — per-method CLIs / orchestrators
  - `code/role_b/plot_curves.py`, `run_all.py` — figures and the results-table reproducer
  - `configs/*.yaml` — every hyperparameter
  - `REPORT_roleB.md`, `README.md`, and this file — write-up and documentation

  No other AI assistant (e.g. Copilot, ChatGPT, Gemini) was used for Role B.

## What I decided vs. what was AI-assisted
- **My design decisions (I directed these):** using the simulator's no-fly-aware routed distances
  as policy features (to match the information `greedy_nearest` uses); selecting checkpoints on
  validation `cost_per_order` rather than training return; making the dispatch policy
  dimension-robust (factored per-action heads + Deep-Sets critic) so it survives the held-out
  grading config; keeping Role B self-contained (a local BFS router, no reach into simulator
  internals); the reward-scaling + Huber-loss fix for the value-function divergence; the
  behavior-cloning warm-start (clone greedy, then refine) that makes REINFORCE beat greedy on every
  seed; success-based DDPG checkpoint selection, the OU-noise/decay + minimum-speed-floor fixes, and
  the TD3 stabilisation for DDPG's don't-move optimum and seed-variance; the GAE-λ ablation design;
  and the seed-hygiene split (train on a disjoint high seed pool, validate on 200–202, report
  mean ± std on held-out seeds 0–4).
- **AI-assisted (I reviewed all of it):** PyTorch boilerplate, the GAE computation, the replay
  buffer, and the table/plot code.

## Reference implementations consulted
The algorithms follow standard public references (allowed by §2): CleanRL and OpenAI Spinning Up
for A2C/DDPG structure, and the original papers cited in `REPORT_roleB.md` (Williams 1992;
Mnih et al. 2016; Schulman et al. 2016; Lillicrap et al. 2016). No code was copied verbatim; the
implementations are written against this project's specific masked, factored action space.
