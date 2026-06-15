# Part 1: Environment-Dependent Network State

Working math notes for dissertation Part 1. Sources:
- 1.1: `AL4SRTB9`, Environment-framed networks.
- 1.2: `XXDCGSDI`, but the math below is grounded mainly in the current `equatorial` experiment artifacts.

---

## Section 1.1

---

## 1.1 Seasonal Arctic Service Accessibility

The transport system is represented as a dynamic graph:

$$
G(t) = (V, E(t)),
$$

where:
- `V` is the set of settlements;
- `E(t)` is the set of transport links available at time `t`;
- link availability depends on external environmental conditions, primarily temperature.

For a connection of type `k` between settlements `i` and `j`, the paper defines a temperature-dependent feasibility weight:

$$
P_{ij}^{k}(t) =
P(E_{ij}^{k}(t)=1 \mid \tilde{T}_{ij}(t))
=
\frac{1}{1 + \exp\left(-\frac{\tilde{T}_{ij}(t)-\theta_k}{\beta_k}\right)}.
$$

Here:
- `k` is the transport mode or link type: regular road, aviation, winter road, water transport;
- `\tilde{T}_{ij}(t)` is the effective temperature proxy for the connection between `i` and `j`;
- `\theta_k` is the critical temperature threshold for mode `k`;
- `\beta_k` controls how sharply the link switches between infeasible and feasible states.

This probability is not service accessibility. It only defines whether a transport edge can be used at time `t`.

After feasible edges are selected, service accessibility is evaluated over the realized network. For region `r`, with admissible travel-time threshold `D_r`, settlement `i` is connected to service provider `j` at time `t` if:

$$
\tau_{ij}(t) \le D_r,
$$

where `\tau_{ij}(t)` is travel time over the feasible network `G(t)`.

For each service type `m`, service presence is encoded as:

$$
\delta_{im} =
\begin{cases}
1, & \text{if service } m \text{ is present at node } i, \\
0, & \text{otherwise}.
\end{cases}
$$

Thus, the model separates two levels:
- environmental conditions determine which network links exist;
- planning/service thresholds determine whether demand can access a provider through the realized network.

---

## Section 1.2

---

## 1.2 Equatorial Road-Network Stability Under Climate Stressors

This formulation should be read from the implemented experiment, not mainly from the draft text.

Main artifacts:
- `equatorial/outputs/astar_accessibility_weekly/paper_experiment_rainfall_mechanism_v1/data/source_weekly_rain_speed_penalty_rules.csv`
- `equatorial/outputs/astar_accessibility_weekly/paper_experiment_rainfall_mechanism_v1/data/weekly_country_mechanism.csv`
- `equatorial/outputs/astar_accessibility_weekly/paper_experiment_rainfall_mechanism_v1/data/country_mechanism_summary.csv`
- `equatorial/outputs/astar_accessibility_weekly/paper_experiment_regimes_v1/data/top_regimes.csv`

The experiment uses a baseline road network and recalculates accessibility under weekly rainfall-driven travel-time penalties. The network can be written as:

$$
G_s(t) = (V, E_s(t)),
$$

where:
- `V` is the set of road-network nodes plus crop-origin and destination connectors;
- `s` is the rainfall/surface treatment scenario;
- `t` is a week;
- `E_s(t)` is the same road topology with scenario-adjusted edge weights, except for near-closure cases that behave like practical disconnection.

Each road edge `e` has surface class:

$$
r_e \in \{\text{paved}, \text{unpaved}, \text{synthetic connector}\}.
$$

Weekly rainfall exposure is assigned by country/week:

$$
q_c(t) = \text{weekly rainfall statistic for country } c.
$$

The artifact `source_weekly_rain_speed_penalty_rules.csv` defines a speed multiplier:

$$
\mu_e(t) = M(r_e, q_c(t)),
$$

with the current rule table:

| Road type | Weekly rainfall, mm | Speed multiplier |
| --- | ---: | ---: |
| paved | 50-100 | 0.90 |
| paved | 100-200 | 0.75 |
| paved | 200-300 | 0.40 |
| paved | 300+ | 0.05 |
| unpaved | 50-100 | 0.70 |
| unpaved | 100-150 | 0.45 |
| unpaved | 150-250 | 0.20 |
| unpaved | 250+ | 0.05 |

In the current experiment manifest, the scenario is `unknown_as_unpaved`; therefore unknown road surfaces are treated as unpaved for rainfall penalties.

If baseline edge travel time is `\tau_e^0`, weekly scenario travel time is:

$$
\tau_e^s(t) = \frac{\tau_e^0}{\mu_e(t)}.
$$

For each crop-origin / destination pair `(o,d)`, the experiment recomputes shortest-path travel time:

$$
\tau_{od}^s(t) =
\min_{p \in \mathcal{P}_{od}}
\sum_{e \in p} \tau_e^s(t).
$$

The travel-time degradation is:

$$
\Delta\tau_{od}^s(t) =
\tau_{od}^s(t) - \tau_{od}^{0}.
$$

In the artifact tables this appears as delay variables such as `median_delta_minutes`, `mean_delay_affected_h`, and `peak_delay_h`.

The weekly country-level severe burden is built from affected cells/routes:

$$
B_c(t) =
\sum_{(o,d)\in c}
\max(0, \Delta\tau_{od}^s(t) - H),
$$

where `H` is the severe-delay threshold. In the current artifact naming this is the 3-hour threshold:
- `weekly_burden_h`;
- `affected_cells_ge_3h`;
- `mean_delay_affected_h`.

The annual / experiment-period country burden is:

$$
B_c =
\sum_t B_c(t).
$$

This is stored as `total_burden_h` in `country_mechanism_summary.csv`.

For the crop-regime analysis, each regime is:

$$
\rho = (c, \text{crop}, \text{destination type}),
$$

and the artifact `top_regimes.csv` stores:
- `affected_weeks`;
- `mean_affected_delay_h`;
- `peak_delay_h`;
- `annual_severe_burden_h`;
- `affected_cluster_weight`;
- `burden_share`.

The key experimental claim is therefore not "more rainfall directly means more burden". The rainfall mechanism figure explicitly compares rainfall severity with accessibility burden and shows rank mismatches. The operational claim is narrower:

$$
\text{rainfall} + \text{road surface}
\rightarrow
\text{speed penalty}
\rightarrow
\Delta \text{travel time}
\rightarrow
\text{crop/destination accessibility burden}.
$$

---

## Generalization

---

## Generalized Part 1 Formulation

Both 1.1 and 1.2 can be described as environment-dependent network realization followed by accessibility recalculation.

Let the baseline network be:

$$
G_0 = (V, E_0).
$$

Let external environmental conditions at time `t` be:

$$
z(t) = (z_1(t), z_2(t), \ldots, z_q(t)).
$$

These can be temperature in 1.1, or rainfall, flood depth, heat, and impermeability in 1.2.

A realization function maps baseline network, edge attributes, and environmental conditions into a state-specific network:

$$
G_{\omega}(t) =
\Phi(G_0, X_E, z(t), \omega),
$$

where:
- `X_E` are edge attributes;
- `\omega` denotes the rule set or scenario;
- `G_{\omega}(t)` is the realized network at time `t`.

Accessibility is then not computed on the static network, but on the realized network:

$$
A_{od}^{\omega}(t) =
f(o, d, G_{\omega}(t)).
$$

For service systems, this becomes a provider-access condition:

$$
A_{ij}^{\omega}(t)=1
\quad \Longleftrightarrow \quad
\tau_{ij}^{\omega}(t) \le D.
$$

For supply chains, this becomes a travel-time or cost deterioration condition:

$$
\Delta \tau_{od}^{\omega}(t)
=
\tau_{od}^{\omega}(t) - \tau_{od}^{0}.
$$

The shared idea for Part 1:

$$
\text{environment}
\rightarrow
\text{network state}
\rightarrow
\text{accessibility / service or supply-chain outcome}.
$$

The difference between 1.1 and 1.2 is the realization function:
- 1.1 uses a temperature-dependent edge-feasibility model;
- 1.2 uses climate-stressor road-status rules and speed penalties.
