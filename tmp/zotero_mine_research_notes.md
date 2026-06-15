# Zotero MINE research notes

Collection: `MINE` (`26LW7QPR`)

Read from local Zotero PDFs on 2026-06-15. These are working notes, not dissertation text.

## Inventory

| Priority | Part | Title | Year/date | PDF text |
| --- | --- | --- | --- | --- |
| important | 1.1 | Environment-framed networks: seasonal reconfiguration of service accessibility in Arctic transport systems | 2026-03-28 | `tmp/pdfs/mine/4PTTGNGK_environment_framed_networks_2026.txt` |
| important | 1.2 | Assessment of Transport Networks Stability in the context of Climate Stressors on Urban Agriculture Supply Chain in Equatorial regions | no date | `tmp/pdfs/mine/5WD98ACA_PDF.txt` |
| important | 2.1 | Enhancing Urban Planning Through Improved Connectivity: A Genetic Algorithm Approach for Optimal Service Placement | 2024 | `tmp/pdfs/mine/QFRQYCVM_PDF.txt` |
| medium | 2.2 | One Rule to Bring Them All: Investigating Transport Connectivity in Public Transport Route Generation for Equitable Access | no Zotero date | `tmp/pdfs/mine/AZRJ34Q6_AAAI_one_rule_to_bring_them_all.txt` |
| low support | support for 2.1 | Spatial-Morphological Modeling for Multi-Attribute Imputation of Urban Blocks | no Zotero date | `tmp/pdfs/mine/F6Y4XNCZ_AAAI_spatial_morphological_modelling.txt` |
| minimal | background only | Assessing the transport connectivity of urban territories, based on intermodal transport accessibility | 2023-06-15 | `tmp/pdfs/mine/HET2Z28V_PDF.txt` |
| minimal | background only | Assessment of Spatial Inequality in Agglomeration Planning | 2023 | `tmp/pdfs/mine/YH5F68C6_PDF.txt` |

## Part 1.1: Environment-Framed Networks

Source: Kontsevik et al. 2026, `AL4SRTB9`.

This is one of the main sources for Part 1.1. It gives the strongest published formulation of environment-dependent network state and seasonal service accessibility.

Core contribution:
- service accessibility is not static in climate-dependent regions;
- transport links can appear/disappear because environmental conditions cross mode-specific feasibility thresholds;
- resilience is measured through stability of service catchments and provider assignments, not only static graph connectivity.

How the model is framed:
- EFN = Environment-Framed Network;
- dynamic graph `G(t) = (V, E(t))`;
- nodes are settlements with population, services, and transport facilities;
- edges are multimodal transport links: regular road, aviation, winter-only road, water transport;
- edge feasibility is modeled as a sigmoid of effective OD temperature and mode-specific threshold;
- temperature controls transport feasibility;
- service accessibility is evaluated only after feasible links are selected;
- a settlement is connected to a provider if the feasible path travel time is within regional planning constraints.

Important distinction from classical multilayer networks:
- classical multilayer transport networks usually treat layers as transport subsystems;
- EFN treats transport as a shared environment-constrained backbone;
- layers are service-specific flows over that backbone;
- the central effect is not node destruction, but edge feasibility and service-flow reassignment.

Empirical results to reuse:
- up to 40% of settlements switch service providers;
- 15-20% of settlements can become temporarily isolated for 2-4 months;
- isolation and provider-switching peaks occur during seasonal transition months;
- regional severity differs, but the mechanism is shared: threshold crossings trigger seasonal reconfiguration.

Useful numeric anchors:
- Yakutia/Chukotka: mean isolated months 9.095, peak isolation share 0.762;
- Yamalo-Nenets AD: mean isolated months 3.263, peak isolation share 0.289;
- Mezen: mean isolated months 9.525, peak isolation share 0.800;
- Northern Administrative District: mean isolated months 1.773, peak isolation share 0.182, peak switch share 0.432.

Use in dissertation:
- as the formal basis for "external factors change network state";
- as a published precedent for recalculating service accessibility on climate-feasible networks;
- as the bridge from static facility accessibility to environment-constrained network design;
- as support for adding `environment-framed / climate-constrained networks` to the classification scheme.

Limitations to mention briefly:
- local route data are partly manual;
- worldwide datasets miss winter roads, local aviation, water routes, and informal links;
- extreme weather is simplified into transport-mode temperature restrictions;
- service-flow layers are independent and do not model cross-layer cascading effects.

## Part 1.2: Climate Stressors And Urban Agriculture Supply Chains

Source: draft `XXDCGSDI`.

This is the main source for Part 1.2. It transfers the environment-framed/network-state idea from Arctic service accessibility to equatorial urban agriculture supply chains.

Core contribution:
- urban agriculture supply chains depend on road network stability under climate stressors;
- road links should be treated as heterogeneous by class and surface;
- external climate factors change link performance, closure probability, or link availability;
- vulnerability is supply-chain specific: production zones to markets, not generic city mobility.

Problem framing:
- equatorial cities face intense rainfall, flood risk, heat, impermeability, and year-round agricultural flows;
- perishable goods make delays costly;
- informal markets and fragmented food distribution systems increase vulnerability;
- road quality matters: unpaved/minor roads are especially sensitive to shallow flooding and rainfall.

Data layer:
- OSM road network, road class, surface;
- CHIRPS and ERA5 rainfall;
- flood hazard maps such as GLOFAS/global flood products;
- Landsat/MODIS land surface temperature and land-cover/imperviousness;
- market and urban agriculture zones from publications, OSM POIs, satellite imagery, and proxies.

Methodological core:
- extract drive network with OSMnx;
- classify roads by type and surface;
- compute centrality and connectivity metrics with NetworkX;
- spatially join climate indicators to road segments;
- identify production zones and markets;
- compute baseline OD routes, travel times, and distances;
- apply climate-stress scenarios by removing or downgrading vulnerable segments;
- recalculate connectivity, travel time, route redundancy, and access;
- compute vulnerability indices combining structural criticality, climate exposure, redundancy, and supply-chain relevance.

Road-status rules to reuse:
- unpaved residential/tertiary/track links with flood depth >= 0.30 m are treated as impassable;
- paved residential/tertiary links under the same depth remain but receive speed reductions;
- primary/secondary segments are removed only above 0.50 m depth;
- high-intensity rainfall, such as local 95th percentile, increases temporary closure probability for unpaved low-lying links;
- heat plus impermeability increases vulnerability weights/disruption probabilities.

Use in dissertation:
- as the main source for road quality, link degradation, and climate stressors;
- as the applied equatorial-region counterpart to EFN;
- as support for a branch like `Road networks -> link quality / climate stressors / supply-chain vulnerability`;
- as the place where `highway`, `surface`, flood depth, rainfall intensity, heat, and imperviousness become network parameters.

Current caution:
- this is a draft, not a published source;
- first version has empty abstract/results/conclusion placeholders;
- second version has stronger abstract/objectives but still lacks full empirical results;
- references include placeholders and should be cleaned before formal citation.

## Part 2.1: Connectivity-Aware Optimal Service Placement

Source: Kontsevik et al. 2024, `N39YX5BF`.

This is the key source for Part 2.1. It is the clearest bridge from facility location to network design.

Core contribution:
- facility/service placement should not assume a fixed cost matrix;
- improving accessibility between selected neighborhoods can reduce the number of facilities required;
- the transport network becomes a decision space, not only an input.

Problem setting:
- service: polyclinic;
- accessibility norm: 15 minutes;
- model: CLSCP-SO, a capacitated location set covering problem variant;
- new polyclinics have fixed average capacity of 400 people;
- cost matrix is travel-time accessibility between neighborhoods on an intermodal graph.

Hypothesis:
- improving connectivity between neighborhoods by no more than 40% can reduce the minimum number of optimally placed new polyclinics.

Optimization logic:
- select cost-matrix cells where current travel time is above 15 minutes but can fall below 15 minutes after up to 40% improvement;
- generate candidate matrices by multiplying selected travel times by random factors from 0.6 to 1;
- run CLSCP-SO on the modified matrix;
- genetic algorithm fitness is the number of facilities selected by CLSCP-SO;
- objective is to minimize the number of new services.

Case and data:
- port-industrial district of Saint Petersburg;
- selected residential blocks tied to industrial worker housing/job mismatch logic;
- OSM service locations, public transport network, residential building parameters;
- BlocksNet for availability and intermodal graph.

Result:
- required new polyclinics reduced from 13 to 9;
- average required transport-connectivity improvement is about 10 minutes;
- in some cases, the model places new facilities where services already exist, indicating possible need for expansion rather than new construction;
- distant neighborhoods sometimes need only slight improvement, while spatially close neighborhoods can require larger travel-time reductions because of route/network structure.

Use in dissertation:
- central support for `facility location + network design`;
- proves why `cost/accessibility matrix` can be mutable;
- supports combined interventions: add facilities, expand existing facilities, or improve transport links;
- directly connects FLP, accessibility, and network design in the classification scheme.

Limitations to mention:
- selected subterritory, not the full city;
- redistribution of service loads at city scale is not modeled;
- future direction is multi-criteria optimization of transport and land-use/block types.

## Part 2.2: TNDP Route Generation For Equitable Access

Source: top-level PDF `AZRJ34Q6`, anonymous submission.

This is important for Part 2.2, but it is less grounded than `N39YX5BF` because the evidence is mainly synthetic and the work is anonymous/unpublished.

Core contribution:
- public transport route generation should include a connectivity-aware accessibility objective;
- TNDP optimization should not rely only on passenger cost and operator cost;
- accessibility/equity can be introduced as a target in generative network design.

Problem:
- Transit Network Design Problem, focused on line generation/selection;
- city graph nodes are stops;
- edges are travel-time links;
- demand matrix is static;
- route set must satisfy route length and connectivity constraints;
- TNDP is NP-hard.

Objective components:
- `Cp`: passenger cost, average in-vehicle travel time with transfer penalties;
- `Co`: operator cost, total route traversal time;
- constraint penalty;
- `Cw`: demand-weighted transport connectivity based on all-pairs transit times and demand weights.

Algorithms benchmarked:
- LC-100 learned constructive baseline;
- BCO metaheuristic baseline;
- NeuroBCO / NEA hybrid neuro-evolutionary algorithm.

Results to reuse:
- passenger-cost optimization lowers travel time but can inflate operator cost by 2-3x;
- operator-cost optimization creates compact networks but worsens direct trips and equity;
- adding `Cw` improves direct-trip shares and reduces long transfer chains;
- balanced objectives sit closer to Pareto tradeoffs;
- NeuroBCO performs best when `Cw` is included;
- overemphasizing `Cw` can raise operator cost 20-40% compared with balanced settings.

Use in dissertation:
- supports the move from measuring accessibility to generating/designing transport networks;
- fits after Part 2.1 as a transport-network-design extension;
- useful for the scheme branch around `Transport networks -> link parameters / public transit routes / fares or costs`.

Limitations:
- mostly synthetic benchmarks: Mandl and Mumford instances;
- paper itself notes ambiguity of applying network generation to real data;
- use as an extension/supporting direction, not the main empirical anchor.

## Support For Part 2.1: Spatial-Morphological Imputation

Source: top-level PDF `F6Y4XNCZ`, anonymous submission.

This is significantly less important. Use it as one applied take for demand-side scenario modeling.

Single useful take:
- using the model developed in this paper, it is possible to estimate how an urban block may change under a given land-use type in a given city;
- for this dissertation, the important consequence is not morphology itself, but demand: how much population/service demand may be added when land-use assumptions change.

Method in one line, only if needed:
- land-use composition and site area are used to infer likely block morphology (`FSI`, `GSI`), which can then be translated into expected capacity/population change.

Use in experiments:
- can define a demand-growth scenario for selected blocks;
- this scenario can be compared with transport-side interventions, such as adding roads or improving public transport;
- useful question: if land-use change increases demand in a block, is it better to add/expand facilities, add roads, or improve transport accessibility?

Do not expand:
- do not present this as a main facility-location or network-design source;
- do not spend space on imputation benchmark details unless the experiment explicitly uses them.

## Minimal Background Sources

These two works are useful mainly as provenance for the accessibility/connectivity vocabulary. They should not take much space in the dissertation.

### Morozov et al. 2023, `VDI4QPKV`

Use only as the early methodological source for intermodal accessibility as mutual connectivity between urban blocks.

Keep from it:
- intermodal graph built from walking links and public transport routes/stops;
- all-to-all travel-time matrix between blocks;
- comparison against Euclidean proximity;
- idea that public transport accessibility reveals enclaves hidden by straight-line distance.

Avoid expanding:
- city-by-city clustering results;
- t-SNE/GMM details unless needed for historical continuity.

### Kontsevik et al. 2023, `R89G28XS`

Use only as early support for connecting service provision with intercity transport accessibility.

Keep from it:
- spatial inequality can be modeled through availability of services plus accessibility to neighboring service territories;
- agglomeration-scale intermodal graph;
- service placement and transport connectivity are alternative/complementary ways to reduce inequality.

Avoid expanding:
- mall-specific case details;
- full indicator scale;
- agglomeration boundary discussion.

## Cross-Paper Priority Line

1. Part 1.1: `AL4SRTB9` gives the published formal model for environment-framed networks and seasonal service accessibility.
2. Part 1.2: `XXDCGSDI` applies the same logic to equatorial road networks, climate stressors, and urban agriculture supply chains.
3. Part 2.1: `N39YX5BF` turns accessibility/cost matrix into a decision variable in facility placement.
4. Part 2.2: `AZRJ34Q6` extends the network-design side toward accessibility-aware TNDP.
5. `F6Y4XNCZ` gives one demand-growth scenario mechanism for Part 2.1 experiments.
6. `VDI4QPKV` and `R89G28XS` remain minimal historical/methodological background.

## Concepts To Reuse

High priority:
- Environment-framed network.
- Temperature-dependent edge feasibility.
- Service catchment stability.
- Seasonal provider switching.
- Climate-adjusted network realization.
- Road surface/class-sensitive disruption rules.
- Climate-weighted criticality.
- Mutable accessibility/cost matrix.
- Facility location + network design.
- Transport intervention as substitute/complement for opening facilities.

Medium priority:
- Demand-weighted transport connectivity.
- Accessibility-aware TNDP.
- Public transport route generation.
- Passenger/operator/connectivity objective tradeoff.

Low priority:
- Intermodal accessibility graph.
- Mutual transport accessibility.
- Spatial inequality through service provision and accessibility.
- Land-use-driven block demand scenario.
