# Ecognomy

A sandbox where an economy emerges from a population of learning agents. The
player sits in the designer's chair, turning world parameters and watching what
falls out.

## Read these first

- `handoff.md` — the vision. What the thing is for and what counts as it working.
- `docs/environment.md` — the environment specification. The formal object, the
  tick phase order, every knob. Code follows this; if they disagree, the spec is
  wrong or the code is, and one of them needs fixing.
- `docs/decisions.md` — **the ledger.** Every choice that shapes the world and
  where it came from: the brief, an explicit decision, or an assumption made to
  keep moving. Read it before changing anything about the world.

## Do not let the model and the collaborator drift apart

**Never add a mechanism to the world without saying so explicitly and adding a
row to `docs/decisions.md`.** Not a passing mention inside a longer message — a
statement that a choice is being made, and what its consequences are.

This rule exists because a regional resource pool was introduced with one
parenthetical aside, and then silently capped production at ~46% of what agents
attempted and drove the correlation between production skill and welfare to
+0.003 for an entire working session. It was found only because two agents that
should have been identical visibly diverged in the UI. Anything discovered that
way was undiscovered for a long time first.

A row marked `assumed` in the ledger is an open question, not a settled decision.
Surface them; do not let them ratify themselves by sitting in the code.

Both are design documents, not build instructions, and they age differently from
the code. Keep them updated rather than duplicating them inline.

## Setup

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

## Layout

| module | owns | swappable |
|---|---|---|
| `ecognomy/config.py` | the dataclass tree the control panel binds to | no |
| `ecognomy/utility.py` | the reward function | no |
| `ecognomy/topology.py` | region graph (weighted, directed, cyclic) | no |
| `ecognomy/world.py` | state arrays, tick phase order | no |
| `ecognomy/actions.py` | the action vectors and their clipping | **yes** |
| `ecognomy/mechanism.py` | meeting rule, execution rule | **yes** |
| `ecognomy/metrics.py` | observables | no |
| `ecognomy/policy/` | decision-making (random control, myopic rung) | **yes** |
| `ecognomy/scenarios.py` | hand-built diagnostic worlds | no |
| `ecognomy/baseline.py` | the autarky counterfactual | no |
| `ecognomy/recorder.py` | run recording and replay | no |
| `ecognomy/viewer/` | the dashboard | **yes, per panel** |

`world` never imports `policy`. Policies propose actions; the world disposes,
silently degrading illegal proposals to IDLE and counting them in metrics.

## Running it

```bash
.venv/bin/python -m ecognomy.simulate --agents 20 --ticks 1000 --out runs/my-run
.venv/bin/python -m ecognomy.viewer.app runs/my-run     # omit the path for the newest run
```

The simulator is headless and writes to disk; the viewer reads from disk and the
two never share a process. Rendering therefore cannot throttle simulation, and
replay is not a separate feature -- live viewing is tailing a run in progress.

## Adding or removing a dashboard panel

Panels are auto-discovered. **A panel is one file in `ecognomy/viewer/panels/`
that defines a module-level `PANEL`.** Deleting the file removes it from the
dashboard; there is no registry to edit, no import to remove, nothing else to
touch. A panel that fails to import is warned about and skipped rather than
taking the dashboard down.

`requires=(...)` names the arrays a panel needs, so a panel skips itself on runs
that predate a field instead of erroring. `tests/test_viewer.py` asserts that
every discovered panel builds against a real run.

Charts go through `viewer/theme.py`: a fixed-order categorical palette (never
cycled), a one-hue sequential ramp for magnitude, 2px lines, and direct labels.
Both light and dark values are validated for colorblind separation and contrast;
`MODE` selects between them. Two measures of different scale get two charts --
never a second y-axis.

## Invariants worth not breaking

- **Nothing is a module constant.** Every quantity in the environment is a config
  field, because the panel is the product.
- **The sandbox must be able to fail.** `tests/test_environment.py` asserts that
  five degenerate configurations produce dead worlds. If one of those ever passes
  with a functioning economy, the design has assumed its answer.
- **Comparative advantage is a sampler property.** `shape_spread` varies the
  *ranking* of goods within an agent's efficiency vector; `scale_spread` only
  makes agents uniformly better or worse. A world with `shape_spread == 0` has no
  gains from trade on the production side, whatever else is set.
- **Reward is linear**: consumption times preference, nothing else. Do not add
  curvature to manufacture an effect — the point is to see effects emerge. The
  consequence to keep in mind is that gains from trade come *only* from agents
  valuing goods differently, and that willingness to pay never responds to
  scarcity.
- **Surplus is scored with true posted prices, never floored ones.** Prices are
  floored only for the division that forms an exchange rate. Scoring with the
  floored price lets `eps * rate` masquerade as gain — with a rate of ~2.6e4 that
  was large enough to make dead scenarios look alive.
- **Scenario endowments must not smuggle in goods nobody produces.** Endowing
  every agent with a little of everything hands each pair something the other
  wants and destroys `triangular` and `autarky` as controls.
- **Nothing in the tick may depend on agent index.** Rationing is pro rata and
  candidate trades are shuffled before sorting, both to stop low ids acquiring a
  structural advantage. `test_identical_agents_end_up_identical` is the guard.
- **No published price.** Agents observe only their own trades and the offers
  shown to them. Price formation is the convergence of subjective estimates; a
  global price signal anywhere in the code would destroy the emergence claim.

## State of play

The environment is built and tested. **No real policy exists yet** — only
`RandomPolicy`, which is a control, not a contender. Under random play the world
trades roughly nine times in 300 ticks, which is the double coincidence of wants
biting as intended.

The dashboard is built. The step-by-step world panel is the main instrument:
locations, ground truth per agent, per-region price boards, recorded who-saw-whom,
and executed trades. Summary panels sit around it.

Open, in rough order: the `assumed` rows in `docs/decisions.md` (unratified
choices sitting in the code), the planner, and the control panel that writes a
`WorldConfig` and launches a run.

Two things `handoff.md` asks for that do not exist yet: **recipes** (goods
composing into other goods) and **taxes and redistribution**.

Not yet verified: the chokepoint price wedge. `prices.py` is written to show it,
but random play trades too thinly to estimate regional prices, so it needs a real
policy before it can be confirmed.
