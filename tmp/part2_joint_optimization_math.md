# Part 2: Joint Service Placement And Transport-Network Optimization

Working math notes for dissertation Part 2. Sources:
- 2.1: `N39YX5BF`, connectivity-aware optimal service placement.
- 2.2: `AZRJ34Q6`, TNDP route generation for equitable access.
- Local implementation: `aggregated_spatial_pipeline`, `solver_flp`, `connectpt`, and `segregation-by-design-experiments/polyclinic_access_components`.

Important scope note: the current implementation is a functional joint pipeline, not a single monolithic mathematical optimizer. It links placement and transport design through service-target OD and repeated accessibility/provision recalculation.

---

## Section 2.1

---

## Connectivity-Aware Service Placement

The service-placement layer starts from residential demand blocks and service/facility candidate blocks.

Let:
- `I` be demand blocks;
- `J` be candidate facility blocks;
- `d_i` be demand at block `i`;
- `q_j` be available or new capacity at candidate `j`;
- `x_j \in \{0,1\}` indicate whether a new service is opened or capacity is expanded at `j`;
- `y_{ij} \in \{0,1\}` indicate whether demand block `i` is assigned to facility `j`;
- `T_{ij}` be travel time between `i` and `j` over the current `PT + walk` accessibility graph;
- `D` be the accessibility threshold.

The accessibility relation is:

$$
a_{ij} =
\mathbf{1}\{T_{ij} \le D\}.
$$

For polyclinic experiments, `D` corresponds to the accepted accessibility threshold for `PT + walk` access.

The simplified capacitated covering formulation is:

$$
\min \sum_{j \in J} x_j
$$

subject to:

$$
\sum_{j \in J} y_{ij} \ge 1
\quad \forall i \in I \text{ with unmet demand},
$$

$$
y_{ij} \le a_{ij} x_j
\quad \forall i,j,
$$

$$
\sum_{i \in I} d_i y_{ij} \le q_j x_j
\quad \forall j \in J,
$$

$$
x_j, y_{ij} \in \{0,1\}.
$$

In the local target-coverage experiments, this is not solved for full coverage by default. For target coverage `\gamma`, e.g. `0.9`, unmet demand is scaled so that the solver answers:

$$
\text{How many additional polyclinics are needed to reach or exceed } \gamma?
$$

The current output metric is:

$$
N_{\text{new}}(\gamma)
=
\sum_{j \in J} x_j.
$$

Local artifacts:
- `segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/`
- `aggregated_spatial_pipeline/outputs/*/joint_inputs/*/pipeline_2/placement_exact_target90*/polyclinic/summary_after.json`
- `aggregated_spatial_pipeline/outputs/*/joint_inputs/*/pipeline_2/solver_inputs/polyclinic/`

---

## Section 2.2

---

## Transport Route Generation As Network Design

The transport-design layer adds or changes public-transport routes to improve accessibility.

Let:
- `S` be PT stops;
- `R` be the generated route set;
- `g_R` be the public-transport graph after adding generated routes;
- `OD_{uv}` be demand or service-target weight between stop `u` and stop `v`;
- `C_p(R)` be passenger cost;
- `C_o(R)` be operator cost;
- `C_w(R)` be demand-weighted connectivity/accessibility cost;
- `P(R)` be route-validity penalties.

A generic TNDP objective is:

$$
\min_R
\lambda_p C_p(R)
+
\lambda_o C_o(R)
+
\lambda_w C_w(R)
+
\lambda_{\text{pen}} P(R).
$$

In the local pipeline, `ConnectPT` receives a service-aware OD matrix rather than a generic citywide OD:

$$
OD^{svc}_{uv}
=
\sum_{i,j}
w_i
\mathbf{1}\{u = stop(i)\}
\mathbf{1}\{v = stop(j)\},
$$

where:
- `i` is a demand block with unmet accessibility demand;
- `j` is its assigned or candidate service block;
- `stop(i)` is the nearest stop to the demand block;
- `stop(j)` is the nearest stop to the facility/candidate block;
- `w_i` is the residual unmet demand weight.

This is materialized as:

$$
\text{placement result}
\rightarrow
\text{assignment links}
\rightarrow
\text{service-target OD}
\rightarrow
\text{route generation}.
$$

Local artifacts:
- `aggregated_spatial_pipeline/outputs/*/joint_inputs/*/pipeline_2/accessibility_first/service_target_od/`
- `segregation-by-design-experiments/polyclinic_access_components/outputs/route_strategy_service_reduction_20260612/`
- `segregation-by-design-experiments/polyclinic_access_components/outputs/overnight_route_strategy_batch_20260613_routes3_finalcanvas_probe/`

---

## Joint Pipeline

---

## Unified Planning Problem

The dissertation-level formulation can be written as a joint planning problem with two decision layers:

$$
\min_{x, R}
\left[
\alpha \sum_{j \in J} x_j
+
\beta C_o(R)
+
\eta C_p(R)
+
\zeta P(R)
\right]
$$

subject to:

$$
\text{Coverage}(x, R) \ge \gamma,
$$

$$
\sum_{i \in I} d_i y_{ij}(x,R) \le q_j(x)
\quad \forall j,
$$

$$
R \in \mathcal{R},
$$

where:
- `x` controls service placement or expansion;
- `R` controls transport-route interventions;
- `\gamma` is target coverage;
- `\mathcal{R}` encodes route feasibility constraints such as route length and valid stop sequence;
- `Coverage(x,R)` is recomputed on the graph after both placement and transport intervention.

Coverage can be written as:

$$
\text{Coverage}(x,R)
=
\frac{
\sum_{i \in I} d_i \cdot
\mathbf{1}
\left[
\min_{j:x_j=1}
T_{ij}(R)
\le D
\right]
}{
\sum_{i \in I} d_i
}.
$$

The local implementation approximates this joint problem sequentially:

$$
\text{baseline accessibility}
\rightarrow
\text{placement}
\rightarrow
\text{service-target OD}
\rightarrow
\text{route generation}
\rightarrow
\text{accessibility recompute}
\rightarrow
\text{provision recompute}.
$$

The key comparison is:

$$
N_{\text{new}}^{\text{placement only}}(\gamma)
\quad \text{vs.} \quad
N_{\text{new}}^{\text{after route intervention}}(\gamma).
$$

If:

$$
N_{\text{new}}^{\text{after route intervention}}(\gamma)
<
N_{\text{new}}^{\text{placement only}}(\gamma),
$$

then the transport intervention substitutes for part of service placement.

This is the operational meaning of the combined FLP/TNDP extension in the dissertation:

$$
\text{service placement}
\leftrightarrow
\text{transport-network design}
\rightarrow
\text{target accessibility / provision}.
$$

---

## Experiment Readout

---

## What The Artifacts Measure

The current artifacts support three levels of evidence:

1. Placement-only baseline:
   - `additional_polyclinics_needed_to_0_9`;
   - `summary_after.json`;
   - `city_target90_pattern_lift_*`.

2. Route strategy intervention:
   - generated route summaries;
   - `bus_service_target_od.csv`;
   - `connectpt_bus_summary.json`;
   - `polyclinic_summary_after_routes.json`;
   - post-route placement outputs.

3. Planning interpretation:
   - where adding services is still required;
   - where route intervention reduces unmet accessibility;
   - where route intervention can reduce the number of additional services.

The strongest Part 2 claim should stay narrow:

$$
\text{transport-network intervention can be evaluated as a substitute or complement to new service placement under a fixed accessibility target}.
$$

