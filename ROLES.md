# Team Roles

| Role | Method family | Chapters | Owner |
|------|---------------|----------|-------|
| A — Value-based | DQN → Double DQN → Dueling DQN (discrete dispatcher) | 15 | _<teammate>_ |
| B — Policy-based | REINFORCE + GAE → A2C, plus DDPG (continuous sub-env) | 16, 17 | **Ozan Karhan** |
| C — Planning | Dyna-Q / MCTS repositioning planner | 18 | _<teammate>_ |

Joint components (built and graded as a team): Offline RL (Ch. 20) and Multi-agent (Ch. 21).

This repository currently contains the **Role B** deliverables (under `code/role_b/`,
`configs/`, `weights/`, `logs/`, `run_all.py`). Roles A and C plug in alongside without
touching the Role B modules.
