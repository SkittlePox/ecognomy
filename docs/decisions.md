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
| Recipes / goods composing into other goods | **deferred** | agreed | in the brief; ruled "not yet, maybe in the future". Every good is made from effort alone. |
| Taxes and redistribution | **deferred** | agreed | in the brief; ruled "may come later". |
| Reward = consumption × preference, linear | — | agreed | chosen over concave after measuring 14.8% of trade sides losing utility |
| Preference vector θ per agent | Dirichlet | brief | "heterogeneous across the population" |
| Dirichlet concentration | 0.6 | assumed | sets how specialised tastes are; under linear reward this is one of only two sources of gains from trade |
| `shape_spread` — how differently agents are good at things | **2.0** | **agreed** | ruled: default 2.0. The brief asks to "spread this distribution wide", and since every agent can make every good this carries most of the reason to trade. At 1.0 the sampled world is near-autarkic (+29, ×1.02, 64% helped); at 2.0 it is +234, ×1.11, 86% helped, with higher total welfare. Exposed as `--shape-spread`. |
| Production efficiency per agent | lognormal | brief | "who is good at what… spread this distribution wide" |
| Every agent can make every good | — | **agreed** | ruled: `n_producible` removed. An agent spreads a fixed effort budget over whichever goods it likes; specialisation is its choice, not a restriction. The measurement that motivated the cap ("autarky is optimal if you can make everything") was taken under the one-action-per-tick encoding, where offering competed with producing. It does not survive: with no cap, welfare is the *highest* of any setting. |
| `effort_cost` — utility charged per unit of effort | 0.02 | **agreed** | ruled: keep. It is the threshold below which producing is not worth the bother — a myopic agent makes good *g* only when `theta_g × e_ig` clears it. Without it, an agent always produces something however worthless. A *utility* sink, distinct from efficiency. |
| Spoilage per good | 0.02, token 0.005 | **agreed** | ruled: keep. The brief asks for sinks; making the **token more durable than everything else** gives it a head start in trade volume that is not itself monetary behaviour, so read the token results with that in mind — a uniform-spoilage control run separates the two effects. |
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
| Agents post a rate **matrix**, not a price vector | `ask[a, b]` | **agreed** | ruled: **reverses the earlier vector decision.** A vector pinned an agent's rate one way to the reciprocal of its rate the other way, so a spread — "I sell an apple for 2 bananas but only pay 1.5" — was not expressible, and shading was inverted: raising a posted price made you a tougher seller *and* a keener buyer of the same good off one number. This is what rungs 2–3 in `policy/` were waiting on. `ask[a,b]` is the minimum units of `b` demanded per unit of `a` given up. |
| No numeraire, so no buyer and no seller | — | **agreed** | ruled: every agent is a barterer posting rates of exchange. Which side of a trade is "buying" depends on which good you price things in, and the world picks none. Bid/ask is a useful reading of the matrix, never a structure in the code. |
| Crossing test `ask_i[a,b] · ask_j[b,a] < 1` | — | **agreed** | the numeraire-free form of "the bid crosses the ask". Reduces to the old vector condition when both postings are reciprocal, so the matrix is a strict superset. |
| Execution at the geometric mean | k = 1/2 | **agreed** | ruled: **forced, not chosen.** `r = lo^(1-k)·hi^k` must give the reciprocal rate when the same trade is written in the other good's units, which requires `k = 1/2`. Every other split — pay-as-bid included — needs a nominated money good, and a barter economy has none. Was `assumed`; now proved and tested. |
| Both sides must gain **strictly** | `min_depth = 1.0` | assumed | a depth of exactly 1.0 means the postings touch without crossing. Stops an indifferent agent being traded into; load-bearing for `triangular`. Now a config knob: above 1.0 it is a minimum margin. |
| Candidates ranked by **cross depth** | `1/sqrt(ask_i·ask_j)` | **agreed** | ruled: replaces ranking by `joint surplus × quantity`. Depth is dimensionless, identical for both sides, and comparable across good pairs. Because `r = ask_i · w`, one global sort by depth simultaneously serves every agent's own ranking over the competing uses of its goods. It is also what `mechanism.py` always claimed to do ("rationed by rate priority") while the code did something else. Cost: welfare 3079 → 3058 on the reference run, trades 1842 → 2070. Ranking is not meant to maximise welfare — a solver would be an auctioneer. |
| Postings normalised before ranking | **removed** | assumed | the exploit it guarded (post 1000× bigger numbers, buy the front of the queue) cannot be expressed against a matrix, which has no free scale. |
| **A meeting yields at most one trade** | `argmax` | **assumed** | **was in the code all along with no row here.** Two agents who could beneficially swap apples-for-bananas *and* cherries-for-durians do only the deeper one. Letting every crossing into the queue is a few lines and is the better rule — a good currently only moves if it is the star of some meeting, which is arbitrary — but it raises volume and would confound the measurement of the matrix change, so it is deliberately a separate step. **Open.** |
| Buying queue priority costs terms of trade | — | **agreed** | ruled: not a design choice but an identity. `r = sqrt(ask_i/ask_j)`, so softening your ask to deepen the cross worsens your received rate by the same square root. Escalation is self-limiting; tested. |
| Simultaneous posting, no intra-tick auction | — | **agreed** | ruled: was `assumed` ("an agent may trade several times per tick against a price it posted once"). Kept, because Myerson–Satterthwaite makes some manipulability mandatory in any budget-balanced bilateral mechanism, so an auction relocates the distortion rather than removing it; because one posted matrix is a commitment across every counterparty in the tick, which is what stops aggression being aimed at a contested rival; and because price discovery already has a time axis — 300+ ticks of re-posting. An intra-tick auction adds a second one. |
| Incoherent postings are legal | measured, not prevented | **agreed** | ruled: any non-negative matrix is legal, including one whose round trip `ask[a,b]·ask[b,a]` falls below 1 and can be money-pumped. The mechanism guarantees both sides gain *in posted terms*, never in true utility — the same stance that lets `RandomPolicy` trade itself poorer. `metrics.arbitrage_depth` measures the exposure over cycles of any length instead. Myopic: depth 1.000, 0 of 20 pumpable. Random: 0.516, 15 of 20. |
| An ask of exactly zero | floored for the rate only | assumed | "any positive amount will do" is a real statement, but the geometric split of `[0, hi]` is 0, which would price the trade at nothing and then drop it on the quantity check — so a good an agent does not want would stop being dumpable. The floor keeps the rate tiny and positive. The crossing test and the depth are scored as written. |
| Quantity: one cap per good, no schedule | `max_trade` (N, G) | **agreed** | ruled: a supply schedule (quantity at each rate) and a buy-side cap are both **redundant under a linear reward** — a posted rate is exact at any quantity, and with no satiation more of a good you value is always better. They become *necessary* if the reward ever goes concave, so that decision and this one are joined. |
| Per-agent `sight`, each agent a different K | lognormal | agreed | "Each agent has a different K!" |
| Seeing is symmetric for matching | — | **agreed** | ruled: good as is. A pair is evaluated if **either** side drew the other, so being seen is as good as seeing; ~36% of evaluated pairs exist because only one side looked. This is why `sight` is a weak predictor of welfare (rank correlation +0.12). |

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
