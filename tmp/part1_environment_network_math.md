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

$$
\begin{array}{lcl}
X_E & = & \text{edge attributes},\\
\omega & = & \text{rule set or scenario},\\
G_{\omega}(t) & = & \text{realized network at time } t.
\end{array}
$$

Accessibility is then not computed on the static network, but on the realized network:

$$
A_{od}^{\omega}(t) =
f(o, d, G_{\omega}(t)).
$$

For both service accessibility and supply-chain accessibility, the KPI can be written as reaching at least one admissible destination within a time threshold:

$$
A_i^{h,\omega}(t)=
\begin{cases}
1, & \text{if } \min_{j\in \mathcal{D}_i^h} \tau_{ij}^{\omega}(t) \le S_i^h,\\
0, & \text{otherwise}.
\end{cases}
$$

where:

$$
\begin{array}{lcl}
h & = & \text{accessibility task type},\\
i & = & \text{origin / demand node},\\
\mathcal{D}_i^h & = & \text{admissible destination set for } i,\\
S_i^h & = & \text{maximum admissible travel time / cost}.
\end{array}
$$

For service accessibility, origins are all demand nodes and destinations are fixed service providers. For supply chains, origins are spatially distributed production / demand nodes and destinations are any admissible destination of the required type.

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

---

## Обобщенная постановка для части 1

$$
G_0 = (V, E_0).
$$

$$
z(t) = (z_1(t), z_2(t), \ldots, z_q(t)).
$$

$$
G_{\omega}(t) =
\Phi(G_0, X_E, z(t), \omega),
$$

$$
\begin{array}{lcl}
X_E & = & \text{атрибуты ребер},\\
\omega & = & \text{набор правил или сценарий воздействия},\\
G_{\omega}(t) & = & \text{реализованное состояние сети в момент } t.
\end{array}
$$

$$
A_{od}^{\omega}(t) =
f(o, d, G_{\omega}(t)).
$$

$$
A_i^{h,\omega}(t)=
\begin{cases}
1, & \text{если } \min_{j\in \mathcal{D}_i^h} \tau_{ij}^{\omega}(t) \le S_i^h,\\
0, & \text{иначе}.
\end{cases}
$$

$$
\begin{array}{lcl}
h & = & \text{тип задачи доступности},\\
i & = & \text{узел-источник / узел спроса},\\
\mathcal{D}_i^h & = & \text{множество допустимых destination для } i,\\
S_i^h & = & \text{максимально допустимое время / стоимость пути}.
\end{array}
$$

$$
\text{внешняя среда}
\rightarrow
\text{состояние сети}
\rightarrow
\text{доступность / сервисный или supply-chain результат}.
$$

Пояснения:

$$
\begin{array}{lcl}
G_0 & = & \text{исходная сеть},\\
z(t) & = & \text{внешние условия среды в момент } t,\\
G_{\omega}(t) & = & \text{состояние сети после применения сценария } \omega,\\
A_{od}^{\omega}(t) & = & \text{доступность, рассчитанная по } G_{\omega}(t),\\
A_i^{h,\omega}(t)=1 & \Longleftrightarrow & \text{узел } i \text{ достигает допустимый destination за время } S_i^h,\\
O^h & = & \text{множество узлов-источников для задачи } h.
\end{array}
$$

---

## Обобщение по линкам

$$
e=(i,j)\in E_0
$$

$$
x_e=(\kappa_e,\rho_e,l_e)
$$

$$
z(t)=(z_1(t),z_2(t),\ldots,z_q(t))
$$

$$
a_e^{\omega}(t)=A_e(x_e,z(t),\omega)
$$

$$
c_e^{\omega}(t)=C_e(x_e,z(t),\omega)
$$

$$
E_{\omega}(t)=\{e\in E_0\mid a_e^{\omega}(t)=1\}
$$

$$
\tau_{ij}^{\omega}(t)=
\min_{p:i\to j}
\sum_{e\in p} c_e^{\omega}(t),
\qquad
e\in E_{\omega}(t)
$$

$$
\begin{array}{lcl}
e & = & \text{линк между узлами } i \text{ и } j,\\
x_e & = & \text{атрибуты линка},\\
\kappa_e & = & \text{класс / тип линка},\\
\rho_e & = & \text{качество / поверхность / режим линка},\\
l_e & = & \text{длина линка},\\
z(t) & = & \text{внешние факторы в момент } t,\\
a_e^{\omega}(t) & = & \text{доступность линка},\\
c_e^{\omega}(t) & = & \text{время / стоимость прохождения линка}.
\end{array}
$$

Общий смысл: в части 1 сеть не проектируется заново. Меняется только состояние уже существующих линков: они могут стать недоступными или получить другую стоимость прохождения под внешними факторами.

---

## Допустимые вмешательства для устойчивости

$$
\text{логистическая задача:}\qquad
\rho_e \rightarrow \rho'_e
$$

$$
c_e^{\omega}(t)=C_e(\kappa_e,\rho'_e,l_e,z(t),\omega),
\qquad
a_e^{\omega}(t)=A_e(\kappa_e,\rho'_e,l_e,z(t),\omega)
$$

$$
\text{сервисно-транспортная задача:}\qquad
E_{\omega}^{\tau}(t)=E_{\omega}^{\tau,0}(t)\cup Y
$$

$$
Y\subseteq \mathcal{Y},
\qquad
y=(i,j,\kappa)\in\mathcal{Y}
$$

$$
\begin{array}{lcl}
\rho'_e & = & \text{улучшенный тип / качество существующего дорожного ребра},\\
\mathcal{Y} & = & \text{заранее допустимое множество новых транспортных связей},\\
y=(i,j,\kappa) & = & \text{новая транспортная связь типа } \kappa \text{ между } i \text{ и } j,\\
Y & = & \text{выбранные транспортные связи}.
\end{array}
$$

---

## Различия постановок

### Сервисно-транспортная задача

$$
\begin{array}{ll}
\text{origin} &
O^h=\text{все узлы спроса},\\
\text{destination} &
\mathcal{D}_i^h=\text{сервисы нужного типа},\\[4pt]
\text{сеть} &
E_{\omega}^{\tau}(t)=E_{\omega}^{\tau,0}(t)\cup Y,\\
\text{ребра} &
\kappa_e=\text{класс транспортной связи},\\
&
A_e,C_e=A_e,C_e(\kappa_e,z(t),\omega),\\[4pt]
\text{вмешательство} &
Y\subseteq\mathcal{Y},\\
&
y=(i,j,\kappa)\in\mathcal{Y}.
\end{array}
$$

### Логистическая задача

$$
\begin{array}{ll}
\text{origin} &
O^h=\text{destination нужного типа},\\
\text{destination} &
\mathcal{D}_i^h=\text{demand-узлы},\\[4pt]
\text{сеть} &
E_{\omega}(t)=\{e\in E_0\mid a_e^{\omega}(t)=1\},\\
\text{ребра} &
\rho_e=\text{класс дороги},\\
&
A_e,C_e=A_e,C_e(\rho_e,z(t),\omega),\\[4pt]
\text{вмешательство} &
\rho_e\rightarrow\rho'_e.
\end{array}
$$

---

## Словарь элементов сети

$$
\begin{array}{lcl}
V & = & \text{множество узлов рассматриваемой сети},\\
V^D \subseteq V & = & \text{узлы спроса},\\
V^S \subseteq V & = & \text{узлы предложения / сервисов},\\
i \in V & = & \text{origin-узел, из которого считается достижимость},\\
j \in V & = & \text{destination-узел, до которого нужно добраться},\\
d_i^k(t) & = & \text{спрос типа } k \text{ в узле } i,\\
q_i^k(t) & = & \text{мощность / предложение сервиса типа } k \text{ в узле } i.
\end{array}
$$

$$
\begin{array}{lcl}
E_t^r & = & \text{дорожные ребра в момент } t,\\
E_t^\tau & = & \text{транспортные связи поверх дорожной сети в момент } t,\\
e=(i,j) & = & \text{link между узлами } i \text{ и } j,\\
\rho_e & = & \text{класс / качество дорожного ребра},\\
\kappa_e & = & \text{класс транспортной связи},\\
l_e & = & \text{длина link},\\
a_e(t) & = & \text{доступность link в текущем состоянии сети},\\
c_e(t) & = & \text{время / стоимость прохождения link},\\
E_t^{new} & = & \text{новые дорожные ребра},\\
E_t^{route} & = & \text{новые транспортные связи / маршруты},\\
E_t^{lost} & = & \text{недоступные ребра после воздействия внешних факторов}.
\end{array}
$$

---

## Слайдовая версия

![Общая математическая постановка задачи 1](./part1_environment_network_slide.png)

---

## Мультиуровневая интерпретация сети

![Мультиуровневая сеть: дороги, транспорт, состояние среды](./part1_multilayer_stack_schema.png)

![Мультиуровневость в двух постановках](./part1_multilayer_task_comparison.png)
