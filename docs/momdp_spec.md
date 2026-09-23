# MOMDP Specification — Vision-Guided MORL for UR5e Manipulation

Month 1, Week 2 deliverable (per `4-month-plan.md`). This is a design document —
values are either (a) grounded in a specific cited paper, or (b) our own design
choice, clearly labeled as such. Nothing here is fabricated as "from a paper."

## 1. Objectives (k=3)

| Objective | Symbol | Direction |
|---|---|---|
| Task success | `r_success` | maximize |
| Energy | `r_energy` | minimize (penalize) |
| Smoothness / jerk | `r_smooth` | minimize (penalize) |

All three are expressed as *maximize* signals (standard MORL-Baselines / vector-reward
convention — `env.step()` returns a vector to be maximized component-wise), so the
penalty terms are written already-negated below.

## 2. Reward formulas — our design, standard robotics-RL conventions

```
r_success = 1.0                            if ||p_ee - p_target|| < eps_success
          = exp(-||p_ee - p_target|| / sigma)   otherwise

r_energy  = -sum_i |tau_i * qdot_i| * dt        (i = 1..6 joints; power = torque * velocity)

r_smooth  = -||qddot_t||^2                       (joint-space jerk/acceleration penalty)
```

`eps_success`, `sigma` are task-specific tuning constants, set during Month 2 Week 2
reward verification (per `4-month-plan.md`'s "validate each learnable by single-objective
PPO" step) — not fixed here.

**Status: proposed, not yet paper-verified.** Sanity-check the exact functional form
against de Heuvel et al. (IROS 2025) and He et al. (PRC-CMORL, Neurocomputing 2023)
before finalizing — this doc captures the structure, not a confirmed-correct formula.

## 3. Preference space Ω

Standard MORL convention (matches de Heuvel et al.'s setup): probability simplex over
the 3 objectives —

```
Ω = { λ ∈ R^3 : λ_i ≥ 0, Σ λ_i = 1 }
```

Sampled during training via HER-over-preferences (per plan's Part A row for de Heuvel
et al.). Power-transform interpolator `I(λ) = λ^p` applied before conditioning the
policy — `p` is a hyperparameter, tuned empirically, not derived from the paper.

## 4. CMDP safety constraints

| Bound | Value | Source |
|---|---|---|
| EE velocity | `< 0.43 m/s` | **Confirmed**: Chi et al., RSS 2023 ("Diffusion Policy"), arXiv 2303.04137, Appendix D.0.1 |
| Table clearance | `≥ 1cm above table` | **Confirmed**: same source |
| Control rate | `125Hz EE-space, interpolated from 10Hz command` | **Confirmed**: same source |
| Collision margin `D_safe` | `0.05m (5cm)`, static, v1 | **Our own design choice** — see rationale below. Not from a cited paper. |

### D_safe rationale (not a citation)

- We looked for a numeric collision-margin value in "Pushp et al." (originally cited
  in `12-month-plan.md:59` as the source for this row) and confirmed it does not exist
  as a fixed number — that paper (POVNav) defines the constraint only symbolically
  (`d(s_t, O_t) ≥ D_safe`, sec. 3.2) and computes the margin at runtime from robot
  geometry + camera parameters, not as a citable constant.
- Our D456 camera's stated depth accuracy is `<2%` at 4m range; within the UR5e's
  actual workspace (well under 1m), sensor noise is on the order of a few millimeters
  — not the binding constraint on `D_safe`.
- `0.05m` is a reasonable starting default consistent with common manipulator-safety
  practice, providing headroom above sensor noise without overly constraining the
  workspace.
- **Planned refinement (Month 3, real safety validation):** a velocity-scaled margin
  `D_safe(v) = D_min + k*v`, once real stopping-distance data from the physical UR5e
  is available. Flag this explicitly in the paper as our contribution, not inherited.

## 5. Implementation notes

- Reward vector shape: `[batch, 3]`, matching `rollout_storage.py`'s
  `compute_returns_multi` (see `algorithms/rsl_rl_gcr_ppo/`) and MORL-Baselines'
  vector-reward `env.step()` convention (Felten et al., NeurIPS 2023).
- To be implemented as an mjlab reward-manager term set, keyed to the UR5e entity
  config in `sim/ur5e_mjlab/ur5e_constants.py`.
- Next step: wire `r_success`/`r_energy`/`r_smooth` into an mjlab `EnvCfg`'s reward
  manager, verify each is independently learnable via single-objective PPO before
  moving to the multi-head critic (per `4-month-plan.md` Month 2, Week 2).