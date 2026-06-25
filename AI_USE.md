# AI Tool Usage Declaration (Role B)

Per project rules (§2), AI tools may be used freely; this is an honest declaration, not a
penalty. I can explain, modify, and defend every line of the Role B code.

## Tools used
- **Claude Code (Anthropic)** — used as a pair-programmer to scaffold and implement the Role B
  components: the observation feature extractor, the factored actor-critic and DDPG networks,
  the REINFORCE+GAE / A2C / DDPG training loops, the YAML configs, `run_all.py`, the plotting
  and ablation runners, and this report.

## What was AI-assisted vs. my own design decisions
- **My decisions (directed the work):** using the simulator's own no-fly-aware routed distances
  as policy features (to match the information `greedy_nearest` uses); selecting checkpoints on
  validation `cost_per_order` rather than training return; making the dispatch policy
  dimension-robust (factored per-action heads + deep-sets critic) so it survives the held-out
  grading config; the GAE-λ ablation; the seed-hygiene split (train on a disjoint high seed pool,
  validate on 200–202, report on held-out seeds).
- **AI-assisted:** PyTorch boilerplate, GAE implementation, replay buffer, and table/plot code.

## Reference implementations consulted
The algorithms follow standard public references (allowed by §2): CleanRL and OpenAI Spinning Up
for A2C/DDPG structure, and the original papers cited in `REPORT_roleB.md` (Williams 1992;
Mnih et al. 2016; Schulman et al. 2016; Lillicrap et al. 2016). No code was copied verbatim; the
implementations are written against this project's specific masked, factored action space.
