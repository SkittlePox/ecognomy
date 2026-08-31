# Decision ledger

Every choice that shapes the world, and where it came from. Three sources:

- **brief** — specified in `handoff.md`
- **agreed** — decided explicitly in conversation
- **assumed** — introduced by Claude without explicit agreement

**Anything marked `assumed` is unratified.** It is in the code because a decision
had to be made to keep moving, not because it was chosen. This file exists because
one such assumption — a regional resource pool that production drew from — silently
halved output and suppressed comparative advantage for an entire working session,
and was only caught when two identical agents visibly diverged in the UI. That is a
bad way to find things.

## Rules

1. Nothing new goes in the code without a row here.
2. An `assumed` row is a question, not a decision. Raise it; don't let it settle by
   default.
3. When something is ratified, move it to `agreed` and say what was decided.

---

## Environment

| what | value | source | note |
|---|---|---|---|
| Goods, alphanumeric names | apple…elderberry | agreed | asked for by name |
| Number of goods | 5 | assumed | 4 ordinary + 1 token candidate |
| **Recipes / goods composing into other goods** | **not built** | brief | `handoff.md` asks for "what goods exist — and how they compose into recipes". There is no production chain: every good is made from effort alone. |
| **Taxes and redistribution** | **not built** | brief | `handoff.md` lists it as a first-class knob. Nothing implements it. |
| Reward = consumption × preference, linear | — | agreed | chosen over concave after measuring 14.8% of trade sides losing utility |
| Preference vector θ per agent | Dirichlet | brief | "heterogeneous across the population" |
| Dirichlet concentration | 0.6 | assumed | sets how specialised tastes are; under linear reward this is the *only* source of gains from trade |
| Production efficiency per agent | lognormal | brief | "who is good at what… spread this distribution wide" |
| `n_producible` — each agent can make only its best k goods | **2** | **assumed** | Not in the brief. Forces dependence rather than letting specialisation emerge. With all 5 producible, autarky is optimal and nobody trades — so this is doing a lot of work. |
| `effort_cost` — a cost per tick of production | 0.02 | assumed | brief says "production costs", which efficiency already expresses; this is a second, separate cost |
| Spoilage per good | 0.02, token 0.005 | **assumed** | brief asks for sinks. Making the **token more durable than everything else** is my choice, and it gives the token a volume head start unrelated to anything monetary. |
| Initial inventory | 1.0 of each good | assumed | |
| **Regional resource stock** | **removed** | assumed | Added by me, capped production at ~46% of attempted and drove comparative-advantage correlation to +0.003. Removed on request. |

## Space

| what | value | source | note |
|---|---|---|---|
| Weighted directed graph, cycles allowed | — | agreed | asked for "arbitrary weighted DAG"; built as a directed graph since a DAG drains agents into sinks with no return path |
| Per-agent mobility | lognormal | agreed | "some agents are born with more efficient travel than others" |
| Travel takes multiple ticks (transit state) | — | assumed | brief says "travel costs"; multi-tick transit with edge occupancy is my elaboration |
| Edge capacity as the chokepoint | — | brief | "distances, travel costs, and especially chokepoints" |
| Default topology | 3 regions in a line | assumed | |

## Market

| what | value | source | note |
|---|---|---|---|
| Bilateral exchange, no central clearing | — | agreed | chosen over an order book |
| Agents post a price **vector**, not a rate matrix | — | agreed | discussed and measured |
| Execution at the geometric mean of the two rates | — | assumed | neutral between the pair, and invariant to which good is labelled first |
| Both sides must gain **strictly** | — | assumed | stops an indifferent agent being traded into; load-bearing for `triangular` |
| Postings normalised before ranking | max = 1 | assumed | closes an exploit where posting bigger numbers bought queue priority |
| Candidates filled greedily by joint surplus | — | assumed | not welfare-optimal; a solver would be an auctioneer, which was rejected |
| Per-agent `sight`, each agent a different K | lognormal | agreed | "Each agent has a different K!" |
| Seeing is symmetric for matching | — | **assumed** | a pair is evaluated if **either** side drew the other, so being seen is as good as seeing. ~36% of evaluated pairs exist because only one side looked. |
| An agent may trade several times per tick | — | assumed | against a price it posted once, before those trades happened |

## Agents

| what | value | source | note |
|---|---|---|---|
| Simultaneous vector actions, not one action per tick | — | agreed | proposed and adopted after the single-slot encoding forced a knife edge |
| Effort splittable across goods | — | agreed | noted that a split only becomes *rational* with diminishing returns inside the tick, which is not implemented |
| Myopic policy posts honest preferences | — | assumed | it has no basis for shading; that needs a model of others |
| Myopic offers its entire holding | — | assumed | safe under linear reward, since any executed trade raises its posted value |

## Measurement

| what | value | source | note |
|---|---|---|---|
| Welfare = summed realised reward | — | agreed | asked for |
| Autarky counterfactual (`sight = 0`) | — | agreed | asked for a comparison against autarky |
| Diagnostic scenarios | 5 | agreed | asked for |
| Token anneal | off by default | brief | the headline experiment |
