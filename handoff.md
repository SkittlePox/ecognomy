# Economy Sandbox — Project Brief

> Vision document. Deliberately light on implementation; those decisions get hashed out
> separately. Read this for *what the thing is for* and *what would count as it working*.

## What this is

A multi-agent sandbox where a population of reinforcement learning agents inhabits a
spatial world, and the goal is for an **economy to emerge rather than be stipulated** —
division of labor, trade, price formation, and ideally a medium of exchange that nobody
hardcoded.

I sit in the designer's chair, not the trader's. I turn knobs on the world and watch what
falls out of it.

## What I'm actually chasing

Two things, and the project is only good if it delivers both.

**The research question.** Can a population of learning agents be induced to *do work* and
*participate in an economy*? This is harder than it sounds, because idleness and autarky
are both stable attractors — production costs effort, so doing nothing is locally optimal
until trade exists, and trade doesn't exist until someone produces a surplus. Nobody moves
first. Escaping that trap by shaping the environment is the actual research.

**The feeling.** Leverage. The specific pleasure of accumulating a good and watching it
appreciate, of seeing a supply curve bend because of something I did. Offworld Trading
Company gets at this from the inside; I want it from the outside — the same tension, but
where my move is changing the world's parameters and the market's response is the payoff.

That second point is not decoration. It is a *requirement on the interface*, and it drives
the transparency constraint below. A sandbox whose dynamics I can't see is a sandbox I
can't play.

## The knobs I want

The panel is the product. Everything here should be a first-class, tweakable parameter,
not a constant buried in a config:

- **Agent preferences** — heterogeneous across the population, and dialable in how much
  agents can substitute one good for another. This parameter alone decides whether a good
  can develop a price floor (and become money) or collapse toward worthless.
- **Production costs** — who is good at what. Heterogeneity here is load-bearing: if
  everyone has the same cost vector there is no comparative advantage, no gains from
  trade, and no economy. Spread this distribution wide.
- **Sinks and faucets** — what brings value into the world and what destroys it. Most
  homebrew economies fail here, because a faucet with no drain only ever accumulates.
- **What goods exist** — and how they compose into recipes.
- **Spatial structure** — distances, travel costs, and especially **chokepoints**. Spatial
  friction is what produces price differences between regions, and price differences are
  what create merchants. This is the feature most likely to generate emergent behavior I
  didn't design.
- **Taxes and redistribution.**
- **Token properties** — whether the currency-candidate has intrinsic use, and how that
  intrinsic use decays over time.

## What success looks like

Not a score. Observables, tracked live:

- fraction of agents producing above subsistence
- trade volume
- price dispersion across regions (should reflect transport cost; a throttled chokepoint
  should open a visible wedge)
- **acceptance rate of the token over time** — does it converge to 1, to 0, or oscillate

Critically: **the sandbox must be able to fail.** If every configuration produces a
functioning economy, the design has assumed its own answer and the results mean nothing.
A dead world is a valid outcome and should be legible as one.

## The headline experiment

Barter has been shown to emerge in settings like this. A purely conventional currency has
not, and there's a clean reason: a token with no intrinsic reward gives a learner no
gradient to follow until enough others already accept it.

So: **annealed commodity money.** Give the token real consumption value, establish
acceptance, then decay that intrinsic value toward zero and measure whether acceptance
survives the decay. If it does, the population is sustaining a medium of exchange held up
by nothing but mutual expectation. That's the result worth chasing.

## Interface principles

- **Transparency is a hard requirement.** Behavior and dynamics must be legible from the
  interface. If I have to read logs to understand what happened, the interface failed.
- **The dashboard matters more than the map.** Live plots of prices by region, trade
  volume, token acceptance, and participation are the instrument panel. Watching little
  agents walk around is secondary.
- **Graphics can be minimal.** Whatever is cheapest and clearest.
- Pause, inspect an individual agent's state and recent decisions, and replay a run.

## Hard constraints

- **Python.** I want to be able to read and modify all of it.
- **Performance is a first-class design constraint, not an optimization pass.** This has to
  run on a laptop without cooking it. Emergence needs a large sample budget, so throughput
  decisions should drive library choice from the start rather than being retrofitted.
- **Scalable.** Population size and world size should be able to grow substantially without
  a rewrite.
- **Standard, maintained libraries only.** No bespoke frameworks, no abandoned repos, no
  fragile installs. If a well-supported library does the job, use it.

## Suggestions (directional, not decided)

- Think in arrays over the whole population rather than per-agent Python loops. This single
  choice probably determines whether the thing runs on a laptop.
- Keep the simulator headless and stream state to a separate viewer, so rendering never
  throttles simulation and runs can be replayed after the fact.
- Expect two phases: a flexible prototype while the rules are still churning, then an
  accelerated version once they stop. Paying the acceleration tax early makes rule changes
  expensive at exactly the moment they should be cheap.
- Worth reading as *specification* rather than as engine or dependency: DeepMind's Melting
  Pot (for its substrate/scenario split and its insistence on evaluating agents against
  co-players they never trained with), Johanson et al.'s emergent bartering work, and
  Kiyotaki-Wright for the theory of why money acceptance is a coordination equilibrium with
  multiple outcomes.

## Open questions to hash out

- Library and acceleration choices, given the performance constraint.
- How much of the exchange protocol to hand the agents versus let them discover. Full
  emergence from primitives is a reward-sparsity problem; some scaffolding is likely
  necessary, and where exactly to put it is a real design decision.
- Target population scale.
- Whether agents share a policy conditioned on their preferences, or train independently.
- How to escape the lazy equilibrium in early runs without permanently propping the
  economy up.

---

*If dropping this into a repo for Claude Code, point `CLAUDE.md` at this file rather than
pasting it inline — it's a design brief, not build instructions, and the two age
differently.*
