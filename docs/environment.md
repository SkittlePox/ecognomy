# Environment Specification

> The world, formally. No policy or learning decisions here — those live separately.
> Companion to `handoff.md`, which is the vision doc.

## Formal object

A **partially observable Markov game** (stochastic game): `(N, S, {A_i}, P, {R_i}, {O_i}, γ)`.

Not a Dec-POMDP: rewards are individual, not shared. Agents are self-interested and
there is no team objective. This is what makes trade a real negotiation rather than a
coordination problem with a known solution.

- `N` — agents. Default 20.
- `S` — world state, including in-transit agents.
- `A_i` — per-agent action set, varies by state (masked).
- `P` — transition, defined by the tick phase order below.
- `R_i` — individual reward: consumption times preference, net of effort and travel.
- `O_i` — partial observation. Agents never see others' inventories, preferences,
  efficiencies, or mobility. Those must be modeled from observed behavior.
- `γ` — discount, config.

All quantities are `float32` and continuous. Producing 0.4 apples in a tick is the
normal case, not a special one.

## Goods

Five goods, alphanumeric so ordering is unambiguous:

| index | name | role |
|---|---|---|
| 0 | apple | ordinary |
| 1 | banana | ordinary |
| 2 | cherry | ordinary |
| 3 | durian | ordinary |
| 4 | elderberry | **token candidate** |

`elderberry` is an ordinary good in every mechanical respect. It is the token only in
that its consumption weight is the one the anneal schedule drives to zero. The
annealed-commodity-money experiment requires the token to start as a genuine commodity.
`token_good` is a config index; `n_goods` is a config value. Adding `fig` should
require no code change.

## Reward

Per-tick reward for agent `i`:

```
R_i(t) = u_i(q_i(t))  −  effort_cost · effort_i(t)  −  travel_cost_i(t)

u_i(q) = Σ_g  θ_ig · q_g
```

where `q_i(t)` is the vector *consumed this tick*. **Reward is the amount of each
good consumed multiplied by the preference for it, and nothing else.** There is no
substitutability parameter and no curvature: an agent's value for a good does not
change with how much it holds.

Consumption is entirely voluntary — an agent may hold inventory indefinitely and
consume nothing, which is what leaves hoarding and market-cornering available.

**On flat per-tick penalties.** A constant "tick of pain" is policy-invariant: it
shifts every trajectory's return equally and changes no decision. It is deliberately
not used. Consumption pressure comes from **spoilage** `δ_g` instead — held goods
decay, so inventory has a carrying cost, and unlike a flat penalty that changes
decisions.

### What linearity buys, and what it costs

**A posted price is exact at any quantity.** Since a good's value never changes, the
rate an agent should demand for one unit is the rate it should demand for a thousand.
The mechanism's requirement that both sides gain is therefore a real guarantee rather
than a marginal approximation. Under a concave reward it was not: a price computed at
the margin, applied to half a holding, approved trades that left a participant worse
off — measured at **14.8% of trade sides, destroying about a fifth of the gross
gains.** That failure mode is now structurally impossible, and a test asserts it.

**Gains from trade come only from differing preferences.** Two agents who value goods
identically have nothing to gain by exchanging, however lopsided their holdings. With
a concave reward, inventory differences were a second, independent source of gains;
that channel is closed. A population with uniform tastes is now a dead world whatever
its production arrangement, and that is one of the failure-mode tests.

**Willingness to pay does not respond to scarcity.** Holding almost none of a good
does not make an agent want it more. Regional price differences are therefore
*compositional* — they reflect which agents are standing where — rather than driven
by local shortage. A throttled chokepoint can still sort agents between regions, but
it cannot open a price wedge by making a good locally dear. This is a real change to
one of the observables `handoff.md` asks for, made deliberately: the alternative was
baking a taste for variety into the reward function to manufacture the effect.

### δ, the hoardability knob

`δ_g` is per-good spoilage per tick. It doubles as the durability parameter:

- **Low `δ_g`** — durable, hoardable, corner-able, viable store of value, plausible
  money candidate.
- **High `δ_g`** — perishable, cannot be monopolized, cannot be money.

Durability is the property that historically distinguishes commodity monies, so the
token's `δ` is a design choice as load-bearing as its consumption weight. Setting all
`δ` high forbids monopoly; setting one low invites it.

## Agent attributes

Fixed at spawn, drawn per agent:

- **Preference vector** `θ_i ∈ R^G`, `θ_i ≥ 0`, from a Dirichlet with a concentration
  knob. Low concentration gives sharply specialized tastes; high gives near-uniform ones.
  Under a linear reward this heterogeneity is the *only* demand-side reason to trade,
  so a high concentration is itself a way to kill the economy.
- **Production efficiency** `e_i ∈ R^G`. `e_ig` is units of good `g` produced per tick
  of effort. **Every agent can make every good**, so specialisation is a choice an
  agent makes by where it puts its effort, never a restriction imposed on it.
  Since nobody is locked out of any good, `shape_spread` carries most of the reason to
  trade — agents gain from exchange only when they differ sharply in what they are good
  at, which is why the brief asks for this distribution to be spread wide.
  **Comparative advantage requires `e_i` to vary in relative, not absolute terms** — an agent uniformly worse at everything still has a comparative advantage in
  something. The sampler must vary the *shape* of `e_i`, not just its scale, or there
  are no gains from trade and no other setting can rescue the economy. This is a sampler
  correctness property, not a config value, and it is easy to get wrong invisibly.
- **Mobility** `m_i > 0`. Traversal speed; see transit below.
- **Sight** `sight_i >= 1`. How many postings in its region the agent can see.

## Spatial structure

An **arbitrary weighted directed graph**. Regions are nodes; passages are directed
edges. Cycles are allowed and expected — a DAG would make travel one-way with no return
path, draining agents irreversibly into sink regions. A DAG remains expressible as a
topology if one is ever wanted.

- `weight[e]` — **distance**, not a toll.
- `capacity[e]` — max agents *occupying* the edge at once. This is the chokepoint knob.
  A traversal that would exceed capacity is refused and the move becomes a no-op.
- Per-direction weights, so upstream and downstream can differ.
- Default topology: 3 regions in a line, so the middle is structurally a broker. Any
  region count and topology is config.

### Transit

Movement is not instantaneous. An agent entering edge `e` accumulates progress at rate
`m_i` per tick and arrives when progress reaches `weight[e]`.

While in transit an agent: holds its inventory, **cannot trade, produce, or consume**,
and occupies one unit of `capacity[e]`. Travel cost is charged per tick in transit.

Consequences worth being explicit about, since they are the point of the mechanism:
high-mobility agents cross expensive edges in fewer ticks and forgo less trading time,
which makes them naturally suited to spatial arbitrage; and chokepoints bite on
occupancy over time rather than on instantaneous flow, so a long throttled edge is a far
harder barrier than a short one of the same capacity.

## Tick phase order

Fully ordered, so the transition is deterministic given actions and RNG draws.

1. **Transit.** Advance in-transit agents by `m_i`; arrivals enter their destination.
   New traversals admitted if edge capacity allows, else refused. Travel cost charged.
2. **Produce.** Effort is a vector: an agent allocates up to one unit of effort across
   goods and receives `effort_g · e_ig` of each.

   Production is limited only by effort and efficiency. There is no shared resource
   pool: the drains that stop goods accumulating are consumption and spoilage, both
   proportional to what is held, so they self-limit without a cap on the faucet.

   **The environment already allows splitting** — 0.5 apple and 0.5 banana in one tick is
   legal and the transition handles it. `MyopicPolicy` nonetheless puts all its effort on
   one good, and that is correct rather than a limitation: production is *linear* in
   effort within a tick, so the return to splitting is never higher than going all-in on
   the best good, and the optimum is always a corner.

   Making a split genuinely optimal needs **diminishing returns within the tick** — for
   example `effort_g^β` with `β < 1`, so the second half of a tick spent on the same good
   yields less than the first. That is a one-line change to this phase and a config knob;
   it is deliberately not made yet, because it would change every welfare number measured
   so far and there is no evidence yet that corner production is distorting anything.
3. **Post.** Each settled agent posts a rate matrix and a quantity cap per good.
4. **Meet.** Each agent is exposed to `sight_i` counterparties sampled within its
   region. `sight` is the search-friction knob: small values give persistent
   dispersion and durable arbitrage, `sight → all` approaches Walrasian conditions.
   Seeing is symmetric for matching — a pair is evaluated if *either* side drew the
   other.
5. **Match and execute.** For each meeting, the G×G sweep of directed swaps; a swap
   is available when the two postings **cross**, `ask_i[a,b] · ask_j[b,a] < 1`. The
   deepest crossing swap per meeting becomes a candidate, all candidates across the
   region are sorted by **cross depth**, and they fill greedily against both sides'
   decrementing budgets. Execution splits the bargaining interval geometrically:

   ```
   rate   r = sqrt( ask_i[a,b] / ask_j[b,a] )
   depth  w = 1 / sqrt( ask_i[a,b] · ask_j[b,a] )   > 1 whenever crossed
   ```

   The geometric mean is **forced, not chosen.** The general split
   `r = lo^(1−k)·hi^k` must give the reciprocal rate when the same trade is written
   in the other good's units, which requires `k = 1/2`. Pay-as-bid and every other
   split need a nominated money good, and a barter economy has none.

   Since `r = ask_i[a,b] · w`, each side receives its own reservation demand times
   the depth — which is why one global sort by depth simultaneously serves every
   agent's private ranking over the competing uses of its goods. It also means
   softening your ask to deepen the cross worsens your received rate by exactly the
   same square root, so buying queue priority is never free.

   A meeting yields **at most one trade**, which is a real restriction on volume and
   an open question in the ledger.
6. **Consume.** A full bundle, clipped to inventory; `+inf` means "whatever is left".
   Reward is linear in each good, so only how much of each is eaten matters, never
   the mix.
7. **Sinks.** Spoilage `δ_g` applied to all inventories, including in transit.
9. **Anneal.** Token consumption weight stepped per schedule.
10. **Record.** Metrics appended.

## Sinks and faucets

Enumerated so the balance is auditable — the brief flags this as where homebrew
economies usually fail.

**Faucets:** production.

**Sinks:** consumption, spoilage `δ_g`, travel cost (utility), production effort (utility).

Two sinks destroy utility rather than goods. **Goods balance and utility balance are
separate accounts** and the dashboard shows both; a world can be goods-stable and
utility-bankrupt.

## Observation

Agent `i` observes: own region or transit state, own inventory, own `θ_i`, `e_i`, `m_i`,
the postings exposed to it this tick, accept/reject outcomes of its own postings, and
rates of trades it participated in. It does **not** observe others' inventories,
preferences, efficiencies, mobility, or postings it was not exposed to.

There is no published price. Each agent must estimate what rate will be accepted, and
that estimate is a subjective price. Price formation is the convergence of those
estimates across the population, which is the emergence claim the project makes.

The dashboard's "implied value" readout — the single valuation a rate matrix is
nearest to — exists so a chart can draw one line per good instead of G per agent. It
is computed in `metrics`, no agent observes it, and nothing in the tick reads it. A
global valuation visible to agents would be exactly the published price this refuses.

## Action space

Simultaneous continuous vectors, one set per agent per tick. In-transit agents emit
nothing: their heads are zeroed and their postings set to refusal.

```
consume    (N, G)     quantities to eat, clipped to inventory; +inf means all
effort     (N, G)     effort allocation across goods, rows sum to <= 1
ask        (N, G, G)  reservation rates
max_trade  (N, G)     how much of each good the agent will part with
move       (N,)       edge index to enter, or -1
```

### The rate matrix

`ask[i, a, b]` is the **minimum units of `b` that agent `i` demands per unit of `a`
it gives up**. `+inf` is a refusal to trade that pair at any rate; the diagonal is
meaningless and never read.

A matrix rather than a vector of valuations, because a vector pins the rate one way
to the exact reciprocal of the rate the other way. Three things follow from that
which the vector could not express:

- **A spread.** "I will sell an apple for 2 bananas but only pay 1.5" needs
  `ask[apple,banana] · ask[banana,apple] = 1.33 > 1`. A vector pins that product at
  exactly 1.
- **Shading in the right direction.** Under a vector, raising your posted apple
  price made you a tougher apple *seller* and simultaneously a keener apple *buyer*,
  because both directions read off one number. Shading requires the two sides to
  move independently.
- **Incoherence.** `ask[a,b] · ask[b,a] < 1` is an agent whose own bid crosses its
  own ask, and longer cycles work the same way — the matrix is a currency exchange
  table, and being pumpable is the classic negative-cycle problem. This is **legal**.
  The mechanism guarantees both sides gain in *posted* terms, never in true utility,
  and `metrics.arbitrage_depth` measures the exposure rather than preventing it.

For G goods a vector has G−1 free parameters and a matrix has G(G−1). At G=5 that is
4 against 20; the extra 16 are 10 spreads and 6 cross-rate inconsistencies.

**There is no numeraire, and therefore no buyer and no seller.** Every agent is a
barterer posting rates of exchange. Which side of a trade is "buying" depends
entirely on which good you choose to price things in, and the world chooses none.
Bid and ask are a useful *reading* of the matrix under a chosen numeraire, never a
structure in the code.

### What quantity does and does not say

`max_trade` is one cap per good — how much of it the agent will part with — and not
a schedule of quantity at each rate, nor a cap on what it will acquire. Both of
those are **redundant under a linear reward**: a posted rate is exact at any
quantity, so a schedule is a flat line, and with no satiation more of a good you
value is always better. They become *necessary* if the reward ever gains curvature,
so that decision and this one are joined.

### Legality is clipping

Malformed proposals are made harmless rather than rejected. `nan` in `ask` becomes a
refusal, not a zero — flattening a refusal to zero would turn "never" into "take it
for free", which is the most expensive possible misreading. Disabling a head is how
ablations run: a world without `trade` is the autarky control.

## Configuration

One nested dataclass tree, which the control panel binds to directly. Every quantity
above is a field; none are module constants.

```
WorldConfig
  goods, n_agents, seed, gamma
  TopologyConfig     regions, directed_edges, weight, capacity
  PreferenceConfig   dirichlet_concentration
  ProductionConfig   efficiency_mean, efficiency_scale_spread,
                     efficiency_shape_spread, effort_cost
  MobilityConfig     mobility_mean, mobility_spread, travel_cost_per_tick
  MarketConfig       min_depth
  TokenConfig        token_good, anneal_start, anneal_end, anneal_schedule
  SinkConfig         spoilage[g]
```

## Modules

| module | owns | swappable |
|---|---|---|
| `world` | state arrays, tick phase order | no |
| `actions` | the action vectors and their clipping | **yes** |
| `mechanism` | meeting rule, execution rule | **yes** |
| `policy` | decision-making | **yes** |
| `metrics` | observables, recording | no |
| `config` | dataclass tree | no |

`world` never imports `policy`. `policy` never mutates state; it returns actions that
`world` applies.

## Observables

- fraction of agents producing above subsistence
- **quoted spread** — the median round trip `ask[a,b]·ask[b,a]` a posting demands.
  1.0 is honest reciprocal posting; above 1 is a margin demanded in both directions
- **arbitrage depth** — the deepest money pump available against each agent, per leg
  of the cycle, by Karp's minimum mean cycle over `log(ask)`. Detects postings that
  do not hang together
- trade volume, total and per good pair
- price dispersion across regions, against edge distance
- token acceptance rate through the anneal
- **volume matrix over good pairs** — token-pairs staying thick while ordinary pairs go
  dead means the token has become the routing hub for indirect exchange
- **exchange-value holdings** — inventory of goods with `θ_ig ≈ 0`, held anyway.
  Shared precursor to arbitrage and money acceptance; they are the same behavior.
- **concentration per good** (Herfindahl over holdings) — detects cornering, which low
  `δ_g` permits

## Failure modes the environment must permit

The brief requires the sandbox to be able to fail. Each of these should produce a dead
world, and each is a test:

- identical `e_i` shape across agents → no comparative advantage → no trade
- identical `θ` across agents → nothing to gain from exchange → no trade
- all `δ_g` = 0 with unbounded faucets → accumulation without drain
- all `capacity` = 0 → regions autarkic → no spatial arbitrage
- `sight_mean` = 0 → no agent sees any other → no trade regardless of every other setting

If any of these still produces a functioning economy, the implementation has assumed
its answer.
