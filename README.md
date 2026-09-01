# Ecognomy

A sandbox where an economy emerges from a population of agents rather than being
stipulated. You sit in the designer's chair: turn knobs on the world, run it, and
watch what falls out — division of labour, trade, price formation, and ideally a
medium of exchange nobody hardcoded.

The design documents are [`handoff.md`](handoff.md) (what it's for),
[`docs/environment.md`](docs/environment.md) (the environment, formally), and
[`docs/decisions.md`](docs/decisions.md) (every choice that shapes the world and
where it came from).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -e ".[dev,viz]"
```

Then either activate the venv once:

```bash
source .venv/bin/activate      # afterwards, just `python -m ...`
```

or prefix each command with `.venv/bin/python`. The examples below use the
explicit path so they work regardless.

## Running it

Two steps. The simulator is headless and writes a run to disk; the viewer reads
it back. They never share a process, so rendering can't slow the simulation down,
and replaying an old run is the same code path as watching a new one.

**1. Simulate.**

```bash
.venv/bin/python -m ecognomy.simulate --ticks 1000 --out runs/my-run
```

Prints a summary and writes `runs/my-run/` (a `run.npz` and a `config.json`).

**2. View.**

```bash
.venv/bin/python -m ecognomy.viewer.app
```

Serves on <http://127.0.0.1:8050> — open the URL it prints. With no argument it
picks the newest run under `runs/`. Ctrl-C to stop.

```bash
.venv/bin/python -m ecognomy.viewer.app runs/demo        # a specific run
.venv/bin/python -m ecognomy.viewer.app --port 8051      # if 8050 is busy
```

### A run worth looking at first

The 20-agent default trades about 13 times in 300 ticks, which is correct — the
double coincidence of wants is severe and agents are currently random — but it
leaves the charts sparse. This one has enough activity to fill them:

```bash
.venv/bin/python -m ecognomy.simulate \
    --agents 60 --k 12 --ticks 1000 --anneal-ticks 500 --seed 3 --out runs/demo
.venv/bin/python -m ecognomy.viewer.app runs/demo
```

### Simulate flags

| flag | default | what it does |
|---|---|---|
| `--ticks` | 1000 | length of the run |
| `--agents` | 20 | population size |
| `--regions` | 3 | regions, connected in a line |
| `--seed` | 0 | RNG seed |
| `--sight-mean` | 3 | average number of a region's posted prices an agent can see — the **search-friction knob**. Low values give persistent price dispersion and durable arbitrage; high values approach a frictionless market. `0` disables the market. |
| `--sight-spread` | 0.6 | dispersion in market access. **Each agent has a different K.** `0` makes everyone equal. |
| `--shape-spread` | 2.0 | how differently agents are good at things. Since every agent can make every good, this carries most of the reason to trade — at 1.0 the world goes near-autarkic. |
| `--concentration` | 0.6 | Dirichlet concentration on tastes; lower means sharper preferences. |
| `--anneal-ticks` | 0 | ticks over which the token's intrinsic value decays to zero. `0` disables it. |
| `--policy` | myopic | `random` is the control; `myopic` is rational with no learned parameters. |

**Every agent can make every good**, at its own efficiency. Effort is a vector,
so an agent spreads a fixed budget over whichever goods it likes — 0.5 apple and
0.5 banana is a valid action — and specialisation is a choice it makes rather
than a restriction imposed on it.

Because nobody is locked out of anything, **`--shape-spread` carries most of the
reason to trade**: agents only gain from exchange when they differ sharply in
what they are good at. The brief asks for this distribution to be spread wide,
and the default of 2.0 reflects that. At 1.0 the sampled world goes near-autarkic
— gain over autarky +29 (x1.02), 64% of agents helped, against +234 (x1.11) and
86% at 2.0.

The myopic policy still goes all-in on one good, and that is correct: production
is linear in effort within a tick, so splitting never beats the best single good.
Making a split rational needs diminishing returns inside the tick (`effort^β`,
β < 1), which is a one-line change to the produce phase and deliberately not
made yet.
| `--kappa` | 3.0 | myopic only: value of stock in hand against utility now. Above 1 patient, below 1 impatient. |
| `--out` | timestamped | where to write the run |

Everything else — the goods, spoilage rates, ρ, production spread, topology
weights, chokepoint capacities — lives in `ecognomy/config.py` rather than on the
command line, because those are the knobs the control panel will eventually bind
to. Edit the dataclass defaults to reach them for now.

## What you'll see

| panel | question it answers |
|---|---|
| Welfare vs autarky | **The score.** How much pleasurable consumption did this world deliver, and how much of it did the market contribute? |
| Participation and trade | Is anything happening at all? A flat line is a dead world. |
| World, step by step | **The main instrument.** Agent locations and ground truth, each region's price board, who can see which prices, and every trade that executed — one tick at a time, with step and play controls. |
| Prices by region | Do regions price goods differently, and does a throttled chokepoint open a wedge? Built on **posted** prices, which every agent sets every tick, rather than the sparse record of what executed. |
| Token | Does acceptance survive the token losing its intrinsic value? |
| Volume by good pair | Is the token becoming the hub that other goods route through? |
| Concentration | Is anyone cornering a market? |
| Agent over time | One agent's trajectory — above all the prices it posts, since price formation is the convergence of those lines across agents. |

## Reward

An agent's reward is **the amount of each good it consumed, multiplied by its
preference for that good**, and nothing else:

```
reward = Σ (theta[good] × consumed[good])   −  effort cost  −  travel cost
```

Deliberately linear. Three consequences worth knowing before reading any result:

**A posted price is exact at any quantity**, because a good's value never changes
with how much you hold. The mechanism requires both sides of a trade to gain, and
under a linear reward that is a real guarantee — measured across 651 trades, zero
sides were made worse off. Under a concave reward it was not a guarantee: a price
computed at the margin, applied to half a holding, approved trades that hurt a
participant **14.8% of the time**, destroying about a fifth of the gross gains.

**Gains from trade come only from differing preferences**, never from differing
holdings. Two agents who value goods identically gain nothing by exchanging,
however lopsided their inventories. A population with uniform tastes is a dead
world whatever its production arrangement.

**Willingness to pay does not respond to scarcity.** Holding almost none of a
good does not make an agent want it more, so regional price differences reflect
*which agents are standing where* rather than local shortage. A chokepoint sorts
agents between regions; it does not make a good locally dear. This is a real
change to one of the observables in `handoff.md`, taken deliberately — the
alternative was building a taste for variety into the reward to manufacture the
effect, rather than letting effects emerge.

## The score

Every run reports **total welfare**: realised consumption utility summed over all
agents and all ticks, net of the effort and travel spent getting it. It is not
GDP and not market cap — it is a measure of how much pleasurable consumption the
world actually delivered.

The number alone means little, because a world can post a high total just by
making production cheap. So every run is also measured against **the same world,
same seed, same policy, with the market switched off**. The difference is what
trade contributed, as distinct from what production contributed.

```
  welfare vs autarky — the designer's score
    with market           242.789
    autarky baseline        8.066
    gain from trade      +234.722   x30.10
    agents better off than autarky: 100%
    -> trade is creating value
```

**Tuning parameters to raise the gain is the game.** Pass `--no-baseline` to skip
the counterfactual and halve runtime.

Two readings worth keeping apart. `gain` is always safe. The ratio is reported
only when the baseline is positive, because with both welfares negative the
quotient inverts — `-0.96 / -1.04` reads as 0.92, suggesting a loss for a run
that is in fact ahead.

Also watch **agents better off**: a total can rise while most agents lose. A
market that lifts the total by enriching two agents and impoverishing eighteen is
a different world from one that lifts everybody, and only the per-agent chart
distinguishes them.

A run can come out *below* its own autarky baseline, and that is a real outcome
rather than a bug — the mechanism checks that both sides posted compatible
ratios, not that either posted a sensible one, so a policy that trades badly
loses value by trading.

## Diagnostic scenarios

Sampled worlds tell you whether an economy formed. They cannot tell you *which
capability* a policy is missing, because everything varies at once. A scenario is
a tiny hand-built world — 2 to 5 agents — with every preference, efficiency and
holding fixed, so exactly one capability is under test and the answer is known
in advance.

```bash
.venv/bin/python -m ecognomy.simulate --scenario triangular --policy myopic --sight-mean 20 --out runs/tri
.venv/bin/python -m ecognomy.viewer.app runs/tri
```

| scenario | agents | tests | solvable by direct exchange |
|---|---|---|---|
| `mutual_gains` | 4 | direct exchange — each makes what it does not want, and wants what its neighbour makes | yes |
| `comparative_advantage` | 2 | specialisation without dependence — each can make both goods, but is better at the one the *other* wants | yes |
| `triangular` | 3 | **indirect exchange** — wants form a cycle, no pair can trade directly | no |
| `triangular_with_token` | 5 | **medium of exchange** — the cycle plus a good nobody can eat | no |
| `autarky` | 2 | negative control — everyone already makes what they want | no |

`triangular` is the discriminator. Every unwanted good has weight exactly zero,
so whoever receives is always handed something worthless. The only route is to
accept a good you cannot consume and spend it onward, which is the same
capability money needs. Current results:

| scenario | welfare | autarky | gain |
|---|---|---|---|
| `mutual_gains` | 251.5 | 16.0 | **+235.5** |
| `comparative_advantage` | 481.6 | 250.9 | **+230.7** |
| `triangular` | 0.0 | 0.0 | 0.00 |
| `triangular_with_token` | 0.0 | 0.0 | 0.00 |
| `autarky` | 505.2 | 505.2 | 0.00 |

`triangular` has a **positive control**: `RandomPolicy` posts prices unrelated to
its preferences, so it accepts goods it does not consume, stumbles right around
the cycle, and earns real welfare there. That proves the cycle is traversable —
so the myopic agent's exact zero is a missing capability, not a world built to be
dead. The myopic agent refuses every available exchange because each leaves it
strictly indifferent, and will not even produce, since its only makeable good is
worthless to it and producing on spec pays only if someone trades for it later.

## Reading the world panel

The step-by-step panel is where you watch the mechanism actually work:

- **Region cards** carry that region's **rate board** — the posting of every
  agent present: the **implied value** of each good, the **margin** it quotes
  around that value, and **how much of each it will part with**. A rate with no
  quantity behind it cannot be traded against. Value *levels* are per-agent and
  meaningless in isolation — only the ratios within a row carry information — so
  they are shown unshaded while quantities are shaded like every other amount.
  The board summarises a G×G matrix into one row per agent; the matrix itself is
  legible one agent at a time, so the full round trips live in the agent
  inspector. A margin of 1.00 is honest reciprocal posting, above 1 is a spread
  demanded in both directions, and below 1 is a posting that can be cycled
  against its own author.
- **Pick an agent** in the "highlight what agent sees" dropdown and the rows it
  can see this tick are shaded. Sight is resampled every tick, so this is
  recorded rather than derived — an agent with `sight` 8 standing in a region
  with 3 others sees 3.
- **Agents — ground truth** lists what no agent can observe about another: true
  θ, true production efficiency, mobility, and sight. Comparing an agent's posted
  price against its true θ is how you tell honest pricing from shading.
- **Trades are grouped by region**, since a trade only ever happens between agents
  standing in the same place. Every region gets a card whether or not it traded,
  so the row keeps its shape tick to tick and a market going quiet reads as a
  change rather than a layout shift. Rows are shaded by total units moved, so big
  trades stand out. Participants are marked with a dot on the board above.
- **Population — ground truth** is separate from the per-tick table because none of
  it ever changes: `wants` (how much each agent values each good), `can make`
  (units per full tick of effort, a dot meaning it cannot make that good at all),
  `sight` and `speed`. Every good gets its own column so agents line up for
  comparison regardless of the font.
- **Darker means larger, everywhere.** Green shades what an agent wants and can
  make, blue what it is capable of (sight, speed), orange quantities of goods —
  both held and moving in a trade. Shading is normalised against a **run-wide**
  ceiling, not a per-tick one, so a cell only changes colour when its value
  actually changes and a number can be tracked across ticks. Every ramp step is
  checked to keep text at 4.5:1 or better in both light and dark mode, and
  `tests/test_theme.py` fails if an edit ever breaks that.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Two kinds. Mechanics tests check the transition does what the spec says. **Failure-mode
tests check that the sandbox can fail** — each asserts a specific degenerate
configuration produces a dead world. That matters: if every configuration
produced a working economy, the design would have assumed its own answer and no
positive result would mean anything.

## Layout

```
ecognomy/
  config.py       every knob, as a dataclass tree
  utility.py      the reward function
  topology.py     the region graph
  world.py        state arrays and the tick
  actions.py      what agents may do          (swappable)
  mechanism.py    how trades match and clear  (swappable)
  policy.py       how agents decide           (swappable)
  metrics.py      observables
  recorder.py     saving and loading runs
  viewer/         the dashboard
    panels/       one file per panel          (swappable)
docs/environment.md
```

Adding a dashboard panel means adding one file to `viewer/panels/` that defines a
module-level `PANEL`. Deleting the file removes it — there's no registry to edit.

## State of play

The environment and the dashboard are built and tested. **No agent learns
anything yet.** There are two policies: `RandomPolicy` (the control) and
`MyopicPolicy` (rational, no learned parameters — it prices trades against its
own preferences and picks the best action each tick).

Agents emit **simultaneous continuous vectors** each tick — what to eat, how to
split effort, what rate to post for every ordered pair of goods, and how much of
each they will part with — rather than picking one discrete action. That change removed a
pathology: when offering competed with eating and producing for a single action
slot, a policy either never traded or traded constantly and starved, with nothing
in between. Welfare gain over autarky on the sampled 20-agent world went from
+11 to **+1461**, with 95% of agents better off, and the myopic policy runs about
4.5x faster.

Agents post a **rate matrix**, not a price per good: `ask[a][b]` is the least
amount of `b` an agent will take for a unit of `a`, and the reverse direction is
a separate number. That is what lets an agent quote a **spread** — sell an apple
for 2 bananas but pay only 1.5 — which a single price per good cannot express,
since it pins each rate to the reciprocal of its opposite. It also makes shading
point the right way: under a price vector, marking an apple up made you a tougher
apple seller *and* a keener apple buyer off the same number.

A trade exists where two postings **cross**, `ask_i[a][b] × ask_j[b][a] < 1`, and
executes at the geometric mean of the two — which is forced rather than chosen,
because any other split needs you to nominate a money good, and this is a barter
economy with no such good. There is **no markup parameter**: the posted rate is
simultaneously the agent's reservation and its ask, so shading the rate *is* the
markup — and how far to shade requires knowing what rivals post, which is the
next rung, not this one.

Nothing stops an agent posting rates that do not hang together. If its own bid
crosses its own ask, a counterparty can cycle goods through it and hand back less
than it took. That is legal and measured, not prevented — the mechanism promises
both sides gain in *posted* terms, never in true utility.

Market access is a drawn capability: **each agent has a different `sight`**, the
number of posted prices it can see in its region. Seeing more helps, but only
weakly: rank correlation between `sight` and lifetime welfare is about **+0.12**,
positive in 12 of 12 seeds, with top-quartile agents earning roughly 1.15x
bottom-quartile. Use the rank correlation rather than Pearson — `sight` is
log-normal, and its long tail makes Pearson read near zero (+0.03) while the
effect is consistently there.

The effect is modest partly because *being seen is as good as seeing*: a pair is
evaluated if either side drew the other, and about 36% of evaluated pairs exist
because only one side looked. A low-sight agent that popular agents keep drawing
still trades plenty.

What the myopic rung still cannot do is hold a good it does not consume. It
prices such a good at zero, and the mechanism requires strictly positive surplus
on both sides, so it refuses. That rules out indirect exchange, money and
arbitrage — the same behaviour under three names — and it is exactly what the
`triangular` scenario measures.

Use the step-by-step panel to see it directly: agents stockpile goods they cannot
consume while wanting goods their neighbours are sitting on.

One thing to know before reading any token result: elderberry has lower spoilage
than the other goods, so agents retain and therefore offer more of it even under
random play. The token gets a volume head start from durability alone,
independent of anything monetary. Run a control with uniform spoilage to separate
the two effects.
