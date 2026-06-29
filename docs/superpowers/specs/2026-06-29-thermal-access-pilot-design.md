# Thermal Access Pilot Design

**Date:** 2026-06-29
**Status:** approved design, pending implementation plan

## Objective

Build a standalone research subproject, `thermal_access_pilot`, that demonstrates on real data how pedestrian heat exposure changes route choice and stop accessibility in an N-minute-city analysis.

The pilot must:

- use every building as an origin, without block aggregation;
- calculate baseline and heat-aware walking routes to public-transport stops;
- measure the length and share of each route above a thermal-stress threshold;
- distinguish physical walking time from scenario-based generalized time;
- produce inspectable geospatial artifacts and publication-ready maps;
- fail explicitly if the real thermal model or required data cannot be produced, rather than substituting a proxy.

## Scope

### Study area

- Central Kaliningrad, Russia.
- Circular analysis area with a 1.25 km radius.
- Working CRS: EPSG:32634.
- Target thermal grid resolution: 2 m.

The existing Kaliningrad joint-input bundle is the primary local source:

`aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/kaliningrad_russia`

The exact center will be derived reproducibly from the existing local analysis geometry and recorded in the run manifest.

### Interpretation

The current pilot models **outdoor thermal stress**, not classical canopy-air urban heat-island intensity. Its primary spatial threshold is:

`UTCI > 32 degrees C`

Outputs must therefore be labelled as thermal-stress or heat-exposure areas. They must not be presented as measured urban heat-island intensity.

Spatial cold-wind modelling is out of scope for this pilot. URock and PALM-4U extensions are recorded in the repository backlog.

## Data Sources

### Local vector data

Reuse existing project artifacts rather than downloading equivalent OSM layers:

- Buildings: `derived_layers/buildings_floor_enriched.parquet`.
- Walking network: walk edges and nodes from `intermodal_graph_iduedu`.
- PT destinations: stop/platform nodes from the same `iduedu` intermodal graph, including available bus, trolleybus, and tram modes.

Observed bundle counts before study-area clipping are approximately:

- 7,851 buildings;
- 8,895 graph nodes;
- 24,075 graph edges, including 22,920 walk edges.

These counts are orientation values, not acceptance criteria. Exact clipped counts must be written to the run summary.

### Building height

For each building:

1. use valid OSM `height` when present;
2. otherwise use the existing `storey_restored * 3 m` estimate;
3. enforce a 3 m minimum only for geometries classified as buildings;
4. preserve the chosen source in a provenance column.

No new opaque height-imputation model is introduced in this pilot.

### Terrain, land cover, and vegetation

- Terrain: real SRTM elevation data, cropped to the study area.
- Land cover: ESA WorldCover 10 m, cropped and reprojected.
- Canopy height: ETH Global Canopy Height 2020, 10 m, derived from Sentinel-2 and GEDI.

The ETH product is open and tile-downloadable. It is used as measured/modelled canopy height; constant assumed tree heights are not permitted.

Source: <https://langnico.github.io/globalcanopyheight/>

### Weather

Use hourly ERA5 reanalysis for calendar year 2025:

- 2 m air temperature;
- 2 m dew point or derived relative humidity;
- 10 m wind components;
- surface solar radiation downwards.

Reuse or adapt the existing public ARCO ERA5 access path in `equatorial/src/data/fetchers/era5.py` when compatible.

Select the hottest daylight hour in 2025 for the headline map. Run SOLWEIG for the complete selected day in chronological order so thermal-state variables are not initialized only at the headline hour. Record the selected date, hour, weather values, timezone treatment, and selection rule in metadata.

ERA5 must be described as reanalysis, not station observation.

## Thermal Model

Use the official Python `solweig` package to calculate:

- mean radiant temperature (`Tmrt`);
- Universal Thermal Climate Index (`UTCI`);
- shadow state;
- supporting model metadata and radiation terms where available.

Required surface inputs:

- DSM assembled from terrain and building height;
- DEM;
- canopy DSM from the real canopy-height product;
- land-cover data;
- location and hourly weather.

The implementation must pin package versions and store the SOLWEIG run metadata. If the package cannot run correctly on the local macOS environment, the run stops with a clear error and retained diagnostics. A proxy thermal raster is not an acceptable fallback.

## Network and Route Model

### Origins and destinations

- Origin: centroid or representative interior point of every clipped building.
- Snap each origin to the nearest reachable walk-network node, retaining snap distance.
- Destination set: reachable PT stop/platform nodes in the clipped graph or a small routing buffer around it.
- No block, grid-cell, or neighbourhood aggregation is allowed.

Buildings with no valid network connection must remain in the building results with an explicit failure reason.

### Edge exposure

Intersect or densely sample each walk edge against the UTCI raster at a spacing no coarser than the thermal-cell size. For every edge, store:

- edge length;
- hot length above the threshold;
- hot fraction in `[0, 1]`;
- mean and maximum UTCI where raster coverage is valid;
- raster coverage fraction.

Edges with incomplete raster coverage must be flagged; missing pixels must not silently count as non-hot.

### Baseline routing

For each building, find the minimum physical walking-time path to any PT stop. Physical edge time is derived from edge length and one explicit, configurable pedestrian speed used consistently throughout the pilot.

### Heat-aware routing

For hot fraction `h` and scenario penalty `p`, define edge generalized time as:

`generalized_time = physical_time * (1 + p * h)`

Run sensitivity scenarios:

- `p = 0.25`;
- `p = 0.50`;
- `p = 1.00`.

For each scenario, reroute to any eligible PT stop. The heat-aware route may change both path and destination stop.

The penalty is an explicit behavioural scenario, not a calibrated physiological slowing model. Report separately:

1. baseline physical time;
2. heat-aware route physical time;
3. heat-aware generalized time.

This separation allows the analysis to show both the actual detour and the assumed disutility of heat.

## Building-Level Outputs

One record per building and penalty scenario must include at least:

- building ID and geometry;
- height value and provenance;
- origin node and snap distance;
- baseline and heat-aware destination stop IDs;
- baseline and heat-aware route IDs;
- physical route lengths;
- physical walking times;
- generalized times;
- hot lengths and hot fractions;
- mean and maximum route UTCI;
- physical-time delta;
- generalized-time delta;
- whether the selected stop changed;
- routing or coverage failure status.

Routes and exposed network edges are stored as GeoParquet. Summary tables must report counts, coverage, distributions, and scenario comparisons without replacing building-level data.

## Required Maps

The pilot is incomplete unless all required final maps are generated and visually inspected:

1. `01_inputs.png` — buildings, walking network, PT stops, DSM, and canopy height.
2. `02_thermal_fields.png` — Tmrt, UTCI, and the `UTCI > 32 degrees C` mask.
3. `03_routes_examples.png` — selected baseline and heat-aware route pairs over the hot mask.
4. `04_building_exposure.png` — baseline hot-route fraction assigned to individual buildings.
5. `05_time_change.png` — physical-time and generalized-time change by building.
6. `06_sensitivity.png` — comparable results for penalties 25%, 50%, and 100%.

The route-examples map must use deterministic selection rules, such as buildings with the largest exposure reduction and largest physical detour, rather than hand-picked cases.

GeoTIFF and GeoParquet sources for the maps must be retained so each map can be reproduced in GIS.

## Project Layout

Keep code and outputs inside a standalone directory in the super-dissertation repository:

```text
thermal_access_pilot/
  README.md
  pyproject.toml
  src/thermal_access_pilot/
  tests/
  configs/
  outputs/
    kaliningrad/
      inputs/
      thermal/
      routes/
      tables/
      maps/
      manifest.json
      summary.json
```

This is a normal in-repository subproject, not a Git submodule.

The implementation should remain minimal: reuse existing fetchers and graph artifacts, avoid a general multi-city framework, and add abstractions only where needed for the Kaliningrad pilot or its tests.

## Validation and Failure Behaviour

Automated checks must cover:

- each output row corresponds to one building and one declared scenario;
- no hidden block aggregation occurs;
- all hot fractions are in `[0, 1]`;
- route segment lengths reconcile with complete route lengths within tolerance;
- every successful route terminates at an eligible PT stop;
- heat-aware generalized cost is no greater than the baseline route evaluated with the same penalty;
- missing raster coverage is measured and surfaced;
- deterministic reruns produce the same selected weather hour and example routes.

After an important run, inspect directly:

- `manifest.json` and `summary.json`;
- row counts and key columns in building and route GeoParquet files;
- raster min/max, nodata share, and spatial bounds;
- all six PNG files, confirming that expected layers are actually visible.

Successful process exit alone is not evidence of a correct result.

## Non-Goals

- Empirical calibration of the heat penalty from observed pedestrian trajectories.
- Spatial cold-wind simulation.
- Classical urban heat-island intensity based on canopy-air temperature difference.
- Citywide or cross-city production deployment.
- Aggregation to blocks as the primary analytical unit.

## Acceptance Criteria

The pilot is accepted when a reproducible command produces, for the real Kaliningrad study area:

- a successful real SOLWEIG day run and valid headline-hour UTCI raster;
- baseline and three heat-aware route scenarios from individual buildings to PT stops;
- building-level and route-level GeoParquet outputs;
- manifest and summary artifacts with explicit assumptions and failure counts;
- all six required maps;
- passing automated tests and completed direct artifact inspection.
