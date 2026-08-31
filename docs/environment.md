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
- `R_i` — individual reward: CES consumption utility net of effort and travel.
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

u_i(q) = [ ( Σ_g  θ_ig · q_g^ρ )^(1/ρ) ]^α
```

where `q_i(t)` is the vector *consumed this tick*. Consumption is entirely voluntary —
there is no forced draw. An agent may hold inventory indefinitely and consume nothing,
which is what makes hoarding and market-cornering available strategies.

**On flat per-tick penalties.** A constant "tick of pain" is policy-invariant: it shifts
every trajectory's return equally and changes no decision. It is deliberately not used.
Consumption pressure comes from two state-dependent sources instead:

1. **Concavity of CES** (`ρ < 1`). Marginal utility falls in quantity, so consuming
   steadily beats consuming in bulk, and a varied bundle beats a concentrated one.
2. **Spoilage** `δ_g`. Held goods decay, so inventory has a carrying cost.

Together these remove idleness as a stable attractor without any shaped reward.

### α, the scale knob

The CES aggregate alone is **homogeneous of degree 1**: `u(f·q) = f·u(q)`. That
gives constant returns to scale and leaves no interior optimum in *how much* to
consume — the consume-versus-hold tradeoff becomes linear, so the answer is
always a corner. `α < 1` is a concave transform supplying diminishing returns to
scale.

It does not disturb anything ρ does, because **the marginal rate of substitution
is invariant under a monotone transform**. α governs how much to consume; ρ
governs what to consume; the two do not interact.

### ρ, the substitutability knob

`σ = 1/(1−ρ)` is the elasticity of substitution, constant at all quantity levels.
`ρ = 1` → perfect substitutes (`σ = ∞`); `ρ → 0` → Cobb-Douglas (`σ = 1`);
`ρ → −∞` → Leontief complements (`σ = 0`).

With `θ = (0.5, 0.5)` over apple and banana:

| consumed | `ρ = 0.5` | `ρ = 1` |
|---|---|---|
| (4, 0) | 1.0 | 2.0 |
| (2, 2) | **2.0** | 2.0 |

At `ρ = 0.5` variety is worth double the same total quantity, so trade has a motive.
At `ρ = 1` the bundles are indistinguishable and no trade is worth making. `ρ` therefore
decides both whether gains from trade exist and whether any good can hold a price floor,
which is a precondition for a token becoming money.

### sight, the market-access knob

`sight_i` is how many of its region's posted prices an agent can see, and it is the
informational counterpart to `mobility_i`. **Each agent has a different K**, drawn
log-normally, so market access is a capability that varies across the population rather
than a constant. Agents who see more of the board find better rates: measured
correlation between `sight` and lifetime welfare is about **+0.42**, so broker and
arbitrageur roles can emerge from a drawn attribute rather than being assigned.

It is also the search-friction knob. Full visibility for everyone would largely dissolve
the double coincidence of wants, which is the friction a medium of exchange exists to
solve. `sight_mean = 0` disables the market entirely and is how the autarky
counterfactual is run.

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
  This heterogeneity is the demand-side reason to trade.
- **Production efficiency** `e_i ∈ R^G`. `e_ig` is units of good `g` produced per tick
  of effort. `n_producible` caps how many goods an agent can make at all, zeroing the
  rest. This is load-bearing in a way that is easy to miss: **when every agent can
  produce every good, autarky is optimal** and trade never becomes necessary, only
  advantageous. **Comparative advantage additionally requires `e_i` to vary in relative,
  not absolute terms** — an agent uniformly worse at everything still has a comparative advantage in
  something. The sampler must vary the *shape* of `e_i`, not just its scale, or there
  are no gains from trade and no other setting can rescue the economy. This is a sampler
  correctness property, not a config value, and it is easy to get wrong invisibly.
- **Mobility** `m_i > 0`. Traversal speed; see transit below.
- **Sight** `sight_i >= 1`. How many posted prices in its region the agent can see.
- **Substitution** `ρ`, population-level by default, per-agent optional.

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
   goods and receives `effort_g · e_ig` of each, gated by regional stock.

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
3. **Submit offers.** Each settled agent submits at most one `OFFER(give, want, ratio)`.
4. **Meet.** Each offer is exposed to `K` counterparties sampled within its region.
   `K` is the search-friction knob: small `K` gives persistent dispersion and durable
   arbitrage, `K → all` approaches Walrasian conditions.
5. **Match and execute.** Compatible offers execute at the midpoint of the two stated
   ratios — neutral, so neither side's stated ratio sets the price. Quantity is the
   smaller of the two sides.
6. **Consume.** A full bundle, clipped to inventory. CES utility credited. A bundle
   rather than one good because CES is defined over a bundle: under complements
   (`ρ < 0`) consuming a single good is worth exactly zero and that entire regime
   would be unreachable.
7. **Sinks.** Spoilage `δ_g` applied to all inventories, including in transit.
8. **Faucets.** Regional resource stocks regenerate toward capacity.
9. **Anneal.** Token consumption weight stepped per schedule.
10. **Record.** Metrics appended.

## Sinks and faucets

Enumerated so the balance is auditable — the brief flags this as where homebrew
economies usually fail.

**Faucets:** production (gated by regional stock), regeneration of that stock.

**Sinks:** consumption, spoilage `δ_g`, travel cost (utility), production effort (utility).

Two sinks destroy utility rather than goods. **Goods balance and utility balance are
separate accounts** and the dashboard shows both; a world can be goods-stable and
utility-bankrupt.

## Observation

Agent `i` observes: own region or transit state, own inventory, own `θ_i`, `e_i`, `m_i`,
the offers exposed to it this tick, accept/reject outcomes of its own offers, and ratios
of trades it participated in. It does **not** observe others' inventories, preferences,
efficiencies, mobility, or offers it was not exposed to.

There is no published price. Each agent must estimate what ratio will be accepted, and
that estimate is a subjective price. Price formation is the convergence of those
estimates across the population, which is the emergence claim the project makes.

## Action space

One action per agent per tick. Enumerated per-state and masked, never a fixed index set —
this is the seam that keeps action spaces swappable. In-transit agents have only `IDLE`.

```
MOVE(region)              out-neighbors of current region, capacity permitting
PRODUCE(good)             goods with e_ig > 0 and regional stock available
CONSUME(good, qty)        voluntary, continuous quantity
OFFER(give, want, ratio)  give ∈ goods held, want ∈ goods, ratio solved not enumerated
ACCEPT(offer)             offers exposed this tick
IDLE                      always available
```

`ratio` is not enumerated. It is solved against the agent's acceptance model — expected
value is `P_accept(r) · gain(r)`, a 1-D problem — so ratio stays a fully emergent
continuous quantity at zero branching cost.

## Configuration

One nested dataclass tree, which the control panel binds to directly. Every quantity
above is a field; none are module constants.

```
WorldConfig
  goods, n_agents, seed, gamma
  TopologyConfig     regions, directed_edges, weight, capacity
  PreferenceConfig   dirichlet_concentration, rho
  ProductionConfig   efficiency_mean, efficiency_scale_spread,
                     efficiency_shape_spread, effort_cost
  MobilityConfig     mobility_mean, mobility_spread, travel_cost_per_tick
  ResourceConfig     stock_capacity, regen_rate
  MarketConfig       K, execution_rule
  TokenConfig        token_good, anneal_start, anneal_end, anneal_schedule
  SinkConfig         spoilage[g]
```

## Modules

| module | owns | swappable |
|---|---|---|
| `world` | state arrays, tick phase order | no |
| `actions` | enumeration, masking, application | **yes** |
| `mechanism` | meeting rule, execution rule | **yes** |
| `policy` | decision-making | **yes** |
| `metrics` | observables, recording | no |
| `config` | dataclass tree | no |

`world` never imports `policy`. `policy` never mutates state; it returns actions that
`world` applies.

## Observables

- fraction of agents producing above subsistence
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
- `ρ → 1` → perfect substitutes → no gains from variety → no trade, no price floor
- `n_producible = n_goods` → every agent self-sufficient → autarky optimal → no trade
- all `δ_g` = 0 with unbounded faucets → accumulation without drain
- all `capacity` = 0 → regions autarkic → no spatial arbitrage
- `sight_mean` = 0 → no agent sees any other → no trade regardless of every other setting

If any of these still produces a functioning economy, the implementation has assumed
its answer.
