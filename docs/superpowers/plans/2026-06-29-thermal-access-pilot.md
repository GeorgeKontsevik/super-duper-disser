# Thermal Access Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a real-data Kaliningrad pilot that calculates SOLWEIG UTCI, routes from individual buildings to PT stops, quantifies route heat exposure, and produces six verified maps.

**Architecture:** A small standalone Python package owns configuration, raster preparation, SOLWEIG execution, network exposure, routing, and rendering. It reuses the existing Kaliningrad `iduedu` graph/building bundle and invokes the existing `equatorial` fetch workflow for ERA5 and WorldCover; only SRTM and ETH canopy tiles need a minimal cached downloader. One reverse multi-source Dijkstra per scenario computes the best stop for all snapped building origins.

**Tech Stack:** Python 3.12, uv, GeoPandas, Rasterio, Shapely, NetworkX, Xarray, Matplotlib, Pillow, official `solweig==0.1.0b88`, pytest.

---

## File map

- Create `thermal_access_pilot/pyproject.toml` — isolated runtime and CLI entry point.
- Create `thermal_access_pilot/configs/kaliningrad.toml` — pilot parameters and source paths.
- Create `thermal_access_pilot/configs/equatorial_kaliningrad.yaml` — input to the existing ERA5/WorldCover fetch workflow.
- Create `thermal_access_pilot/README.md` — exact run command, outputs, and analytical caveats.
- Create `thermal_access_pilot/src/thermal_access_pilot/config.py` — typed TOML configuration.
- Create `thermal_access_pilot/src/thermal_access_pilot/local_inputs.py` — study geometry, buildings, walk network, and stops.
- Create `thermal_access_pilot/src/thermal_access_pilot/external.py` — cached SRTM/canopy downloads and reuse of the equatorial fetch CLI.
- Create `thermal_access_pilot/src/thermal_access_pilot/surfaces.py` — aligned 2 m DEM/DSM/CDSM/UMEP land-cover rasters.
- Create `thermal_access_pilot/src/thermal_access_pilot/weather.py` — ERA5 conversion and extreme-day selection.
- Create `thermal_access_pilot/src/thermal_access_pilot/thermal.py` — thin SOLWEIG adapter and headline rasters.
- Create `thermal_access_pilot/src/thermal_access_pilot/routing.py` — edge exposure and multi-source route choice.
- Create `thermal_access_pilot/src/thermal_access_pilot/maps.py` — six required deterministic figures.
- Create `thermal_access_pilot/src/thermal_access_pilot/pipeline.py` — stage orchestration, manifest, summary, and validation.
- Create `thermal_access_pilot/src/thermal_access_pilot/__main__.py` — `--config` and `--force` CLI.
- Create `thermal_access_pilot/tests/conftest.py` — repository and real-artifact fixtures.
- Create focused tests under `thermal_access_pilot/tests/` matching the modules above.

Generated data remain under `thermal_access_pilot/outputs/` and are not committed.

### Task 1: Package, configuration, and CLI contract

**Files:**
- Create: `thermal_access_pilot/pyproject.toml`
- Create: `thermal_access_pilot/configs/kaliningrad.toml`
- Create: `thermal_access_pilot/src/thermal_access_pilot/__init__.py`
- Create: `thermal_access_pilot/src/thermal_access_pilot/config.py`
- Create: `thermal_access_pilot/src/thermal_access_pilot/__main__.py`
- Create: `thermal_access_pilot/tests/conftest.py`
- Test: `thermal_access_pilot/tests/test_config.py`

- [ ] **Step 1: Write the failing configuration test**

```python
from pathlib import Path

from thermal_access_pilot.config import load_config


def test_load_config_resolves_repo_paths_and_scenarios(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    cfg_dir = repo / "thermal_access_pilot" / "configs"
    cfg_dir.mkdir(parents=True)
    path = cfg_dir / "pilot.toml"
    path.write_text(
        """
[study]
center_lon = 20.4531
center_lat = 54.7003
core_radius_m = 1250
model_halo_m = 250
crs = "EPSG:32634"
pixel_size_m = 2
[routing]
walk_speed_m_s = 1.4
hot_threshold_c = 32
penalties = [0.25, 0.5, 1.0]
[paths]
city_bundle = "aggregated_spatial_pipeline/data/city"
output_dir = "thermal_access_pilot/outputs/kaliningrad"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(path, repo_root=repo)

    assert cfg.model_radius_m == 1500
    assert cfg.city_bundle == repo / "aggregated_spatial_pipeline/data/city"
    assert cfg.output_dir == repo / "thermal_access_pilot/outputs/kaliningrad"
    assert cfg.penalties == (0.25, 0.5, 1.0)
```

- [ ] **Step 2: Run the test and verify the expected red state**

Run: `cd thermal_access_pilot && uv run pytest tests/test_config.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'thermal_access_pilot'`.

- [ ] **Step 3: Add the minimal package and typed configuration**

Pin `solweig==0.1.0b88` because its API is explicitly alpha; let `uv.lock` pin transitive versions. Use this complete package definition:

```toml
[project]
name = "thermal-access-pilot"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = [
  "geopandas>=1.0,<2",
  "gcsfs>=2025.1",
  "h5netcdf>=1.6",
  "matplotlib>=3.10,<4",
  "networkx>=3.4,<4",
  "numpy>=2,<3",
  "pandas>=2.2,<3",
  "pillow>=11,<13",
  "pyarrow>=20",
  "rasterio>=1.4,<2",
  "shapely>=2,<3",
  "solweig==0.1.0b88",
  "xarray>=2025.1",
]

[project.optional-dependencies]
test = ["pytest>=8,<10"]

[project.scripts]
thermal-access-pilot = "thermal_access_pilot.__main__:main"

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Use this real configuration:

```toml
[study]
center_lon = 20.4531
center_lat = 54.7003
core_radius_m = 1250
model_halo_m = 250
crs = "EPSG:32634"
pixel_size_m = 2

[routing]
walk_speed_m_s = 1.4
hot_threshold_c = 32
penalties = [0.25, 0.5, 1.0]
max_snap_distance_m = 100

[paths]
city_bundle = "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/kaliningrad_russia"
output_dir = "thermal_access_pilot/outputs/kaliningrad"
```

Define one immutable `PilotConfig` and load TOML with stdlib `tomllib`:

```python
@dataclass(frozen=True)
class PilotConfig:
    repo_root: Path
    city_bundle: Path
    output_dir: Path
    center_lon: float
    center_lat: float
    core_radius_m: float
    model_halo_m: float
    crs: str
    pixel_size_m: float
    walk_speed_m_s: float
    hot_threshold_c: float
    max_snap_distance_m: float
    penalties: tuple[float, ...]

    @property
    def model_radius_m(self) -> float:
        return self.core_radius_m + self.model_halo_m


def load_config(path: Path, repo_root: Path | None = None) -> PilotConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    root = (repo_root or path.resolve().parents[2]).resolve()
    study, routing, paths = raw["study"], raw["routing"], raw["paths"]
    return PilotConfig(
        repo_root=root,
        city_bundle=root / paths["city_bundle"],
        output_dir=root / paths["output_dir"],
        center_lon=float(study["center_lon"]),
        center_lat=float(study["center_lat"]),
        core_radius_m=float(study["core_radius_m"]),
        model_halo_m=float(study["model_halo_m"]),
        crs=str(study["crs"]),
        pixel_size_m=float(study["pixel_size_m"]),
        walk_speed_m_s=float(routing["walk_speed_m_s"]),
        hot_threshold_c=float(routing["hot_threshold_c"]),
        max_snap_distance_m=float(routing.get("max_snap_distance_m", 100)),
        penalties=tuple(float(value) for value in routing["penalties"]),
    )
```

The real TOML uses the approved center, 1.25 km core, 250 m model halo, 2 m grid, 1.4 m/s walking speed, 32 °C threshold, and penalties 0.25/0.50/1.00.

- [ ] **Step 4: Add the CLI entry point**

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/kaliningrad.toml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    from .pipeline import run
    run(load_config(args.config), force=args.force)
```

- [ ] **Step 5: Add shared real-path fixtures**

```python
from pathlib import Path

import pytest

from thermal_access_pilot.config import load_config


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def real_config(repo_root: Path):
    return load_config(repo_root / "thermal_access_pilot/configs/kaliningrad.toml", repo_root=repo_root)


@pytest.fixture(scope="session")
def real_output_dir(real_config) -> Path:
    return real_config.output_dir
```

- [ ] **Step 6: Lock dependencies and verify green**

Run: `cd thermal_access_pilot && uv lock && uv run pytest tests/test_config.py -v`

Expected: `1 passed` and `uv.lock` contains `solweig 0.1.0b88`.

- [ ] **Step 7: Commit the scaffold**

```bash
git add thermal_access_pilot/pyproject.toml thermal_access_pilot/uv.lock thermal_access_pilot/configs/kaliningrad.toml thermal_access_pilot/src thermal_access_pilot/tests/conftest.py thermal_access_pilot/tests/test_config.py
git commit -m "Scaffold thermal access pilot"
```

### Task 2: Real local buildings, walk network, and PT stops

**Files:**
- Create: `thermal_access_pilot/src/thermal_access_pilot/local_inputs.py`
- Test: `thermal_access_pilot/tests/test_local_inputs.py`

- [ ] **Step 1: Write failing tests for the height rule and unaggregated origins**

```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from thermal_access_pilot.local_inputs import resolve_heights, select_building_origins


def test_resolve_heights_uses_height_then_storey_then_three_metres() -> None:
    frame = pd.DataFrame(
        {"height": ["12 m", None, None], "storey": [2.0, 4.0, None], "storey_source": ["osm", "model_predicted", "missing"]}
    )
    result = resolve_heights(frame)
    assert result["building_height_m"].tolist() == [12.0, 12.0, 3.0]
    assert result["height_rule"].tolist() == ["osm_height", "storey_x_3m", "minimum_3m"]


def test_select_building_origins_keeps_one_row_per_building() -> None:
    buildings = gpd.GeoDataFrame(
        {"building": ["yes", "yes"]},
        geometry=[box(0, 0, 2, 2), box(3, 0, 5, 2)],
        crs=32634,
    )
    result = select_building_origins(buildings, Point(2.5, 1).buffer(10))
    assert len(result) == 2
    assert result["building_id"].is_unique
    assert result.geometry.geom_type.eq("Point").all()
```

- [ ] **Step 2: Verify both tests fail because the functions are absent**

Run: `cd thermal_access_pilot && uv run pytest tests/test_local_inputs.py -v`

Expected: import failure naming `resolve_heights` or `select_building_origins`.

- [ ] **Step 3: Implement local input selection**

Implement the approved rules exactly:

```python
def resolve_heights(buildings: pd.DataFrame) -> pd.DataFrame:
    result = buildings.copy()
    osm_height = result["height"].astype("string").str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    osm_height = pd.to_numeric(osm_height, errors="coerce").where(lambda values: values > 0)
    storeys = pd.to_numeric(result["storey"], errors="coerce").where(lambda values: values > 0)
    result["building_height_m"] = osm_height.fillna(storeys.mul(3.0)).fillna(3.0)
    result["height_rule"] = np.select(
        [osm_height.notna(), storeys.notna()],
        ["osm_height", "storey_x_3m"],
        default="minimum_3m",
    )
    return result
```

`load_local_inputs()` must reproject buildings to EPSG:32634, keep representative points inside the 1.25 km core circle, and create stable `building_id` values from the source row index. It loads `graph.pkl`, retains only `type == "walk"` edges whose endpoints lie in the 3 km square model domain, and retains reachable node types `platform`, `bus`, `tram`, and `trolleybus` as destinations. Store the original polygons and representative-point origins separately.

- [ ] **Step 4: Add the real-bundle contract test**

```python
def test_kaliningrad_bundle_has_expected_contract(real_config) -> None:
    inputs = load_local_inputs(real_config)
    assert 250 <= len(inputs.buildings) <= 350
    assert inputs.buildings["building_id"].is_unique
    assert len(inputs.walk_edges) > 100
    assert len(inputs.stops) > 5
    assert set(inputs.walk_edges["type"]) == {"walk"}
    assert inputs.buildings.crs.to_epsg() == 32634
```

- [ ] **Step 5: Run the tests and inspect real counts**

Run: `cd thermal_access_pilot && uv run pytest tests/test_local_inputs.py -v -s`

Expected: all tests pass; the real core contains roughly 286 building origins, with exact counts printed for the run summary rather than hard-coded as results.

- [ ] **Step 6: Commit local inputs**

```bash
git add thermal_access_pilot/src/thermal_access_pilot/local_inputs.py thermal_access_pilot/tests/test_local_inputs.py
git commit -m "Load building-level Kaliningrad routing inputs"
```

### Task 3: Cached real raster acquisition

**Files:**
- Create: `thermal_access_pilot/configs/equatorial_kaliningrad.yaml`
- Create: `thermal_access_pilot/src/thermal_access_pilot/external.py`
- Test: `thermal_access_pilot/tests/test_external.py`

- [ ] **Step 1: Write failing tile and cache tests**

```python
from pathlib import Path

from thermal_access_pilot.external import canopy_tile_url, srtm_tile_url


def test_kaliningrad_static_tile_urls() -> None:
    assert srtm_tile_url(54.7003, 20.4531).endswith("/N54/N54E020.hgt.gz")
    assert "ETH_GlobalCanopyHeight_10m_2020_N54E018_Map.tif" in canopy_tile_url(54.7003, 20.4531)
```

- [ ] **Step 2: Verify the missing URL functions fail**

Run: `cd thermal_access_pilot && uv run pytest tests/test_external.py -v`

Expected: import failure naming `canopy_tile_url`.

- [ ] **Step 3: Implement only the two missing source adapters**

Use deterministic floor-based tile names. SRTM is a 1° tile; ETH canopy is a 3° tile. `download_cached()` writes `destination.part`, checks a non-zero content length, atomically renames, and returns SHA-256. It never replaces an existing non-empty file unless `force=True`.

The verified Kaliningrad URLs are:

```python
SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/N54/N54E020.hgt.gz"
CANOPY_URL = (
    "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download"
    "?path=%2F3deg_cogs&files=ETH_GlobalCanopyHeight_10m_2020_N54E018_Map.tif"
)
```

Decompress SRTM with stdlib `gzip` to `srtm_N54E020.hgt`; retain the compressed source and both checksums in the manifest.

- [ ] **Step 4: Reuse the existing equatorial fetch workflow**

The YAML requests WorldCover v200 and ARCO ERA5 variables `t2m`, `d2m`, `u10`, `v10`, `ssrd`, and `sp` for 2025 over `[20.25, 54.5, 20.75, 55.0]`:

```yaml
global:
  data_root: outputs/kaliningrad/inputs/external
  timeout_seconds: 600
  max_retries: 3
  user_agent: thermal-access-pilot/0.1.0

study_area:
  country_code: RUS
  country_name: Russia
  slug: kaliningrad
  bbox: [20.25, 54.5, 20.75, 55.0]

datasets:
  era5:
    enabled: true
    backend: arco_zarr
    dataset_id: arco-era5
    start_date: "2025-01-01"
    end_date: "2025-12-31"
    target_filename: era5_kaliningrad_2025.nc
    temporal_resolution: hourly
    request:
      variable: [t2m, d2m, u10, v10, ssrd, sp]
  worldcover:
    enabled: true
    year: 2021
    version: v200
    layer: Map
    temporal_resolution: annual snapshot
```

Implement:

```python
def run_equatorial_fetch(cfg: PilotConfig, force: bool = False) -> None:
    expected = cfg.output_dir / "inputs/external/raw/era5/era5_kaliningrad_2025.nc"
    if expected.exists() and not force:
        return
    python = cfg.repo_root / "equatorial/.venv/bin/python"
    command = [str(python), "-m", "src.data.fetch", "--config", str(cfg.repo_root / "thermal_access_pilot/configs/equatorial_kaliningrad.yaml"), "--datasets", "era5,worldcover"]
    env = os.environ | {"PYTHONPATH": str(cfg.repo_root / "equatorial")}
    subprocess.run(command, cwd=cfg.repo_root / "equatorial", env=env, check=True)
    if not expected.exists():
        raise RuntimeError(f"equatorial fetch completed without {expected}")
```

Do not add a second ERA5 or WorldCover client.

- [ ] **Step 5: Run unit tests and source availability checks**

Run: `cd thermal_access_pilot && uv run pytest tests/test_external.py -v`

Run: `curl -I -L --fail https://s3.amazonaws.com/elevation-tiles-prod/skadi/N54/N54E020.hgt.gz`

Run: `curl -I -L --fail 'https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=%2F3deg_cogs&files=ETH_GlobalCanopyHeight_10m_2020_N54E018_Map.tif'`

Expected: tests pass and both HTTP checks return 200 or 206.

- [ ] **Step 6: Commit acquisition code**

```bash
git add thermal_access_pilot/configs/equatorial_kaliningrad.yaml thermal_access_pilot/src/thermal_access_pilot/external.py thermal_access_pilot/tests/test_external.py
git commit -m "Add real thermal pilot data acquisition"
```

### Task 4: Build aligned 2 m SOLWEIG surface rasters

**Files:**
- Create: `thermal_access_pilot/src/thermal_access_pilot/surfaces.py`
- Test: `thermal_access_pilot/tests/test_surfaces.py`

- [ ] **Step 1: Write a failing land-cover mapping test**

```python
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from thermal_access_pilot.surfaces import GridSpec, assemble_surface_arrays, worldcover_to_umep


def test_worldcover_to_umep_material_codes() -> None:
    source = np.array([[10, 50, 60, 80, 0]], dtype=np.uint8)
    mapped = worldcover_to_umep(source)
    assert mapped.tolist() == [[5, 1, 6, 7, 0]]
```

- [ ] **Step 2: Verify the mapping test fails**

Run: `cd thermal_access_pilot && uv run pytest tests/test_surfaces.py -v`

Expected: import failure naming `worldcover_to_umep`.

- [ ] **Step 3: Implement the explicit material mapping**

```python
WORLDCOVER_TO_UMEP = {
    10: 5, 20: 5, 30: 5, 40: 5,
    50: 1, 60: 6, 70: 6, 80: 7,
    90: 5, 95: 5, 100: 5,
}


def worldcover_to_umep(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.uint8)
    for source, target in WORLDCOVER_TO_UMEP.items():
        result[values == source] = target
    return result
```

Buildings override WorldCover with UMEP code 2 after rasterization. Unknown WorldCover values remain code 0 and are counted in the summary.

- [ ] **Step 4: Write and verify a failing synthetic surface test**

```python
def test_build_surface_adds_relative_building_height_to_dem(tmp_path: Path) -> None:
    grid = GridSpec(bounds=(0, 0, 10, 10), pixel_size=2, crs="EPSG:32634")
    buildings = gpd.GeoDataFrame(
        {"building_height_m": [12.0]}, geometry=[box(2, 2, 6, 6)], crs=32634
    )
    result = assemble_surface_arrays(
        grid=grid,
        dem=np.full(grid.shape, 5.0, dtype=np.float32),
        canopy=np.zeros(grid.shape, dtype=np.float32),
        worldcover=np.full(grid.shape, 50, dtype=np.uint8),
        buildings=buildings,
    )
    assert result.dsm.max() == 17.0
    assert np.all(result.land_cover[result.building_mask] == 2)
```

Run: `cd thermal_access_pilot && uv run pytest tests/test_surfaces.py -v`

Expected: failure naming `assemble_surface_arrays`.

- [ ] **Step 5: Implement alignment and GeoTIFF outputs**

Create a square model grid centered on the approved point with 1.5 km half-width. Reproject SRTM and canopy with bilinear resampling, WorldCover with nearest-neighbour resampling, and rasterize building heights with `rasterio.features.rasterize`. Save:

- `inputs/dem_2m.tif` — ground elevation;
- `inputs/dsm_2m.tif` — DEM plus building height;
- `inputs/cdsm_2m.tif` — relative canopy height;
- `inputs/land_cover_umep_2m.tif` — UMEP codes;
- `inputs/core_mask_2m.tif` — 1.25 km analysis circle.

Assert identical CRS, transform, width, height, and bounds for all five rasters. Reject negative canopy values and nodata inside the model domain rather than filling them silently.

- [ ] **Step 6: Run tests and inspect raster metadata**

Run: `cd thermal_access_pilot && uv run pytest tests/test_surfaces.py -v`

Expected: all tests pass.

After the first real build run: `cd thermal_access_pilot && uv run python -m thermal_access_pilot --config configs/kaliningrad.toml`

Expected at this stage: the command may stop at the not-yet-implemented thermal stage, but the five rasters exist and `gdalinfo` or Rasterio inspection shows EPSG:32634, 2 m pixels, matching bounds, plausible elevation, positive building heights, and non-zero canopy coverage.

- [ ] **Step 7: Commit surface preparation**

```bash
git add thermal_access_pilot/src/thermal_access_pilot/surfaces.py thermal_access_pilot/tests/test_surfaces.py
git commit -m "Prepare SOLWEIG surface rasters"
```

### Task 5: Convert ERA5 and select the real extreme day

**Files:**
- Create: `thermal_access_pilot/src/thermal_access_pilot/weather.py`
- Test: `thermal_access_pilot/tests/test_weather.py`

- [ ] **Step 1: Write failing tests for humidity, solar units, and selection**

```python
import pandas as pd

from thermal_access_pilot.weather import relative_humidity, select_extreme_day, solar_w_m2


def test_weather_conversions() -> None:
    assert relative_humidity(20.0, 20.0) == 100.0
    assert solar_w_m2(3_600_000.0, "J m**-2") == 1000.0
    assert solar_w_m2(500.0, "W m**-2") == 500.0


def test_select_extreme_day_uses_hottest_daylight_hour() -> None:
    frame = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(["2025-07-01T10:00Z", "2025-07-01T11:00Z", "2025-08-02T11:00Z"]),
            "ta_c": [29.0, 31.0, 30.0],
            "global_rad_w_m2": [600.0, 0.0, 700.0],
        }
    )
    selected = select_extreme_day(frame, daylight_min_w_m2=20.0)
    assert str(selected["time_utc"]) == "2025-08-02 11:00:00+00:00"
```

- [ ] **Step 2: Verify the tests fail for missing functions**

Run: `cd thermal_access_pilot && uv run pytest tests/test_weather.py -v`

Expected: import failure naming `relative_humidity`.

- [ ] **Step 3: Implement weather conversion without hidden substitutions**

Use Xarray interpolation at the study center. Convert Kelvin to Celsius, dew point to RH with the Magnus formula, `u10/v10` to wind magnitude, Pa to hPa, and hourly `ssrd` J/m² to W/m² by dividing by 3600. Reject unknown units. Convert ERA5 UTC timestamps to `Europe/Kaliningrad`, then remove timezone information only for SOLWEIG's local-clock `Weather.datetime`; retain `time_utc` and `time_local_iso` in the weather table.

```python
def relative_humidity(ta_c: float, dewpoint_c: float) -> float:
    numerator = math.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    denominator = math.exp((17.625 * ta_c) / (243.04 + ta_c))
    return float(np.clip(100.0 * numerator / denominator, 0.0, 100.0))


def solar_w_m2(value: float, units: str) -> float:
    normalized = units.lower().replace(" ", "")
    if normalized.startswith("j"):
        return value / 3600.0
    if normalized.startswith("w"):
        return value
    raise ValueError(f"unsupported solar-radiation units: {units}")


def select_extreme_day(frame: pd.DataFrame, daylight_min_w_m2: float = 20.0) -> pd.Series:
    daylight = frame.loc[frame["global_rad_w_m2"] > daylight_min_w_m2]
    if daylight.empty:
        raise ValueError("ERA5 series has no daylight rows")
    return daylight.sort_values(["ta_c", "time_utc"], ascending=[False, True]).iloc[0]
```

Select the maximum `ta_c` among rows with global radiation above 20 W/m², with earliest UTC time as a deterministic tie-breaker. Keep all 24 local hours belonging to the selected local date and write:

- `thermal/weather_2025_point.parquet`;
- `thermal/selected_day_weather.parquet`;
- `thermal/selected_hour.json`.

- [ ] **Step 4: Add strict real-file checks**

```python
def validate_weather_day(frame: pd.DataFrame) -> None:
    required = {"time_utc", "time_local", "ta_c", "rh_pct", "global_rad_w_m2", "wind_m_s", "pressure_hpa"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"weather columns missing: {sorted(missing)}")
    if len(frame) != 24:
        raise ValueError(f"selected local day has {len(frame)} rows, expected 24")
    if not frame["rh_pct"].between(0, 100).all():
        raise ValueError("relative humidity outside [0, 100]")
    if (frame[["global_rad_w_m2", "wind_m_s"]] < 0).any().any():
        raise ValueError("negative radiation or wind")
```

- [ ] **Step 5: Run tests and inspect the selected real day**

Run: `cd thermal_access_pilot && uv run pytest tests/test_weather.py -v`

Expected: all tests pass.

Inspect the real Parquet and JSON after fetching: confirm 8,760 hourly source rows for 2025 when the ERA5 point has complete coverage, exactly 24 selected-day rows, positive daylight radiation, and a plausible Kaliningrad summer maximum.

- [ ] **Step 6: Commit weather preparation**

```bash
git add thermal_access_pilot/src/thermal_access_pilot/weather.py thermal_access_pilot/tests/test_weather.py
git commit -m "Prepare ERA5 weather for SOLWEIG"
```

### Task 6: Run official SOLWEIG and produce headline thermal rasters

**Files:**
- Create: `thermal_access_pilot/src/thermal_access_pilot/thermal.py`
- Test: `thermal_access_pilot/tests/test_thermal.py`

- [ ] **Step 1: Write a failing real-library smoke test**

```python
from pathlib import Path
from datetime import datetime

import numpy as np
import solweig

from thermal_access_pilot.thermal import run_solweig_arrays


def test_solweig_adapter_writes_real_utci(tmp_path: Path) -> None:
    shape = (24, 24)
    weather = [solweig.Weather(datetime=datetime(2025, 7, 1, 14), ta=30, rh=50, global_rad=700, ws=2)]
    result = run_solweig_arrays(
        dsm=np.zeros(shape, dtype=np.float32),
        dem=np.zeros(shape, dtype=np.float32),
        cdsm=np.zeros(shape, dtype=np.float32),
        land_cover=np.full(shape, 5, dtype=np.uint8),
        weather=weather,
        location=solweig.Location(latitude=54.7003, longitude=20.4531, utc_offset=2),
        output_dir=tmp_path,
        pixel_size=2,
    )
    assert np.isfinite(result.utci_max).any()
    assert (tmp_path / "run_metadata.json").exists()
```

- [ ] **Step 2: Verify the smoke test fails only because the adapter is missing**

Run: `cd thermal_access_pilot && uv run pytest tests/test_thermal.py -v`

Expected: import failure naming `run_solweig_arrays`; importing `solweig` itself succeeds.

- [ ] **Step 3: Implement the thin SOLWEIG adapter**

For the smoke test call `SurfaceData.prepare(dsm=dsm, dem=dem, cdsm=cdsm, land_cover=land_cover, pixel_size=pixel_size, cdsm_relative=True)`. For the real run use file mode:

```python
surface = solweig.SurfaceData.prepare(
    dsm=inputs / "dsm_2m.tif",
    dem=inputs / "dem_2m.tif",
    cdsm=inputs / "cdsm_2m.tif",
    land_cover=inputs / "land_cover_umep_2m.tif",
    cdsm_relative=True,
    working_dir=thermal_dir / "surface_cache",
)
location = solweig.Location(latitude=cfg.center_lat, longitude=cfg.center_lon, utc_offset=2)
warnings = solweig.validate_inputs(surface, location, weather_records)
summary = solweig.calculate(
    surface=surface,
    weather=weather_records,
    location=location,
    output_dir=thermal_dir / "solweig",
    outputs=["tmrt", "utci", "shadow"],
)
```

Persist validation warnings. Treat missing output files, all-nodata outputs, or non-finite UTCI as errors.

- [ ] **Step 4: Extract headline rasters and threshold mask**

Resolve the selected local timestamp to SOLWEIG filenames `utci_YYYYMMDD_HHMM.tif`, `tmrt_YYYYMMDD_HHMM.tif`, and `shadow_YYYYMMDD_HHMM.tif`. Copy or link them to stable names:

- `thermal/headline_utci.tif`;
- `thermal/headline_tmrt.tif`;
- `thermal/headline_shadow.tif`.

Create `thermal/hot_mask_utci_gt_32.tif` with values 0/1 and nodata preserved. Record min, max, mean, nodata fraction, and hot fraction inside the core mask.

- [ ] **Step 5: Run the smoke test, then the real 24-hour model**

Run: `cd thermal_access_pilot && uv run pytest tests/test_thermal.py -v`

Expected: smoke test passes and writes real SOLWEIG metadata.

Run the real stage through the CLI. During the long surface/SVF and 24-hour calculation, emit short stage and heartbeat logs. Expected: 24 Tmrt/UTCI/shadow files plus stable headline rasters and `run_metadata.json`.

- [ ] **Step 6: Inspect actual thermal artifacts**

Open the headline rasters and verify bounds, transform, nodata share, and plausible ranges. Render a temporary preview and visually confirm building and canopy shadow structure rather than accepting process exit alone.

- [ ] **Step 7: Commit the thermal adapter**

```bash
git add thermal_access_pilot/src/thermal_access_pilot/thermal.py thermal_access_pilot/tests/test_thermal.py
git commit -m "Run SOLWEIG thermal stress model"
```

### Task 7: Sample heat exposure onto walk edges

**Files:**
- Create: `thermal_access_pilot/src/thermal_access_pilot/routing.py`
- Test: `thermal_access_pilot/tests/test_edge_exposure.py`

- [ ] **Step 1: Write a failing synthetic exposure test**

```python
import numpy as np
from affine import Affine
from shapely.geometry import LineString

from thermal_access_pilot.routing import sample_edge_exposure


def test_sample_edge_exposure_measures_hot_fraction() -> None:
    utci = np.array([[30, 30, 34, 34]], dtype=np.float32)
    transform = Affine.translation(0, 2) * Affine.scale(2, -2)
    result = sample_edge_exposure(LineString([(0, 1), (8, 1)]), utci, transform, threshold_c=32, spacing_m=2)
    assert result.coverage_fraction == 1.0
    assert result.hot_fraction == 0.5
    assert result.hot_length_m == 4.0
```

- [ ] **Step 2: Verify the test fails for the absent sampler**

Run: `cd thermal_access_pilot && uv run pytest tests/test_edge_exposure.py -v`

Expected: import failure naming `sample_edge_exposure`.

- [ ] **Step 3: Implement midpoint sampling at thermal-cell resolution**

Split every line into `ceil(length / spacing)` equal-length intervals, sample the UTCI at interval midpoints, and weight each sample by its interval length. Return total length, valid coverage length/fraction, hot length/fraction, mean UTCI, and max UTCI. If coverage is incomplete, divide hot fraction by valid-covered length but retain coverage fraction; if coverage is zero, return NaN exposure metrics and an explicit invalid flag.

- [ ] **Step 4: Enrich the real walk-edge table**

Apply the sampler to each walk edge in the model domain. Save `routes/exposed_walk_edges.parquet` with stable `edge_id`, `u`, `v`, original `time_min`, physical time recomputed from the configured 1.4 m/s speed, exposure fields, geometry, and scenario weights:

```python
edges["physical_time_min"] = edges.geometry.length / cfg.walk_speed_m_s / 60.0
for penalty in cfg.penalties:
    key = f"generalized_time_p{int(penalty * 100):03d}"
    edges[key] = edges["physical_time_min"] * (1.0 + penalty * edges["hot_fraction"])
```

Reject successful routing across edges with incomplete raster coverage; keep those edges in the diagnostic file with `routing_eligible=False`.

- [ ] **Step 5: Run and inspect edge outputs**

Run: `cd thermal_access_pilot && uv run pytest tests/test_edge_exposure.py -v`

Expected: tests pass.

Inspect real GeoParquet row count, bounds, fraction ranges, coverage distribution, and several geometries over the hot-mask raster.

- [ ] **Step 6: Commit edge exposure**

```bash
git add thermal_access_pilot/src/thermal_access_pilot/routing.py thermal_access_pilot/tests/test_edge_exposure.py
git commit -m "Measure heat exposure on walking edges"
```

### Task 8: Route every building to the best PT stop

**Files:**
- Modify: `thermal_access_pilot/src/thermal_access_pilot/routing.py`
- Test: `thermal_access_pilot/tests/test_routing.py`

- [ ] **Step 1: Write a failing route-choice test**

```python
import geopandas as gpd
import networkx as nx

from thermal_access_pilot.routing import route_all_origins


def test_heat_penalty_can_choose_longer_cooler_route() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, physical_time_min=1.0, hot_fraction=1.0)
    graph.add_edge(1, 3, physical_time_min=1.0, hot_fraction=1.0)
    graph.add_edge(0, 2, physical_time_min=1.5, hot_fraction=0.0)
    graph.add_edge(2, 3, physical_time_min=1.5, hot_fraction=0.0)
    baseline = route_all_origins(graph, origins=[0], stops=[3], penalty=0.0)[0]
    heat = route_all_origins(graph, origins=[0], stops=[3], penalty=1.0)[0]
    assert baseline.node_path == [0, 1, 3]
    assert heat.node_path == [0, 2, 3]
    assert heat.physical_time_min == 3.0
    assert heat.generalized_time_min == 3.0
```

- [ ] **Step 2: Verify the route-choice test fails**

Run: `cd thermal_access_pilot && uv run pytest tests/test_routing.py -v`

Expected: import failure naming `route_all_origins`.

- [ ] **Step 3: Implement one reverse multi-source Dijkstra per scenario**

For each penalty, assign `generalized_time_min` to eligible graph edges, reverse the directed walk graph, and call `nx.multi_source_dijkstra(reversed_graph, sources=sorted(stops), weight="generalized_time_min")`. Reverse returned node paths back to origin→stop order. For each node pair, choose the parallel edge with the minimum scenario weight and deterministic edge-key tie-break. Sum physical time, generalized time, length, hot length, and sampled UTCI metrics.

The baseline is penalty 0.0. For each non-zero scenario also evaluate the baseline route under that penalty so the optimality check compares like with like.

- [ ] **Step 4: Snap all building origins without aggregation**

Use `GeoDataFrame.sjoin_nearest` against eligible walk nodes, store `origin_node` and `snap_distance_m`, and reject snaps above 100 m with status `snap_too_far`. Preserve disconnected buildings as rows with status `no_route`.

Write:

- `routes/routes.parquet` — one geometry per building/scenario;
- `tables/building_results.parquet` — one building/scenario record with baseline comparison, destination stop, exposure, times, deltas, and `stop_changed`.

- [ ] **Step 5: Add invariant tests**

```python
def test_route_results_satisfy_cost_invariants(real_output_dir) -> None:
    results = gpd.read_parquet(real_output_dir / "tables/building_results.parquet")
    ok = results.query("status == 'ok'")
    assert ok["hot_fraction"].between(0, 1).all()
    assert (ok["generalized_time_min"] <= ok["baseline_route_generalized_time_min"] + 1e-9).all()
    assert (ok["physical_time_min"] >= 0).all()
    assert set(ok["destination_stop_type"]) <= {"platform", "bus", "tram", "trolleybus"}
```

- [ ] **Step 6: Run tests and inspect real route tables**

Run: `cd thermal_access_pilot && uv run pytest tests/test_routing.py -v`

Expected: all tests pass.

Inspect scenario row counts, success/failure counts, route endpoints, exposure fractions, stop changes, and the largest physical/generalized deltas. Confirm route segment lengths reconcile with route geometry within tolerance.

- [ ] **Step 7: Commit building-level routing**

```bash
git add thermal_access_pilot/src/thermal_access_pilot/routing.py thermal_access_pilot/tests/test_routing.py
git commit -m "Route buildings with thermal exposure penalties"
```

### Task 9: Required maps, manifest, and end-to-end validation

**Files:**
- Create: `thermal_access_pilot/src/thermal_access_pilot/maps.py`
- Create: `thermal_access_pilot/src/thermal_access_pilot/pipeline.py`
- Create: `thermal_access_pilot/README.md`
- Test: `thermal_access_pilot/tests/test_maps.py`
- Test: `thermal_access_pilot/tests/test_pipeline.py`

- [ ] **Step 1: Write a failing map contract test**

```python
from pathlib import Path

import numpy as np
from PIL import Image

from thermal_access_pilot.maps import REQUIRED_MAPS, validate_maps


def test_required_maps_are_nonempty_pngs(tmp_path: Path) -> None:
    x = np.tile(np.arange(1200, dtype=np.uint16) % 256, (800, 1)).astype(np.uint8)
    pixels = np.dstack([x, np.flipud(x), np.fliplr(x)])
    for name in REQUIRED_MAPS:
        Image.fromarray(pixels, mode="RGB").save(tmp_path / name)
    validate_maps(tmp_path)


def test_required_map_names_are_fixed() -> None:
    assert REQUIRED_MAPS == (
        "01_inputs.png",
        "02_thermal_fields.png",
        "03_routes_examples.png",
        "04_building_exposure.png",
        "05_time_change.png",
        "06_sensitivity.png",
    )
```

- [ ] **Step 2: Verify the map contract test fails**

Run: `cd thermal_access_pilot && uv run pytest tests/test_maps.py -v`

Expected: import failure naming `REQUIRED_MAPS`.

- [ ] **Step 3: Implement the six deterministic maps**

Use a shared extent, core boundary, scale bar, north arrow, source note, and consistent UTCI/penalty color scales:

1. inputs: DSM, canopy, buildings/network/stops;
2. thermal fields: Tmrt, UTCI, hot mask;
3. route examples: deterministically select up to five buildings with greatest hot-length reduction, then greatest physical detour, breaking ties by `building_id`;
4. individual building polygons colored by baseline hot fraction;
5. physical-time and generalized-time deltas by building for penalty 0.50;
6. three aligned building maps for penalties 0.25, 0.50, and 1.00.

`validate_maps()` opens every PNG with Pillow, requires at least 1,000 × 700 pixels for real outputs, checks non-zero file size, and rejects nearly uniform images by requiring more than 32 RGB colors.

- [ ] **Step 4: Implement pipeline artifacts and final invariants**

`pipeline.run()` creates output directories, executes stages in dependency order, logs stage starts/completions, and writes `manifest.json` plus `summary.json` only after validation. The manifest records source URLs/checksums, config, software versions, selected weather hour, raster statistics, counts, failures, and relative output paths.

The final validator requires:

```python
def validate_final_outputs(output_dir: Path, expected_buildings: int, scenarios: int) -> None:
    buildings = gpd.read_parquet(output_dir / "tables/building_results.parquet")
    routes = gpd.read_parquet(output_dir / "routes/routes.parquet")
    if len(buildings) != expected_buildings * scenarios:
        raise ValueError("building/scenario row count mismatch")
    if not buildings["hot_fraction"].dropna().between(0, 1).all():
        raise ValueError("route hot fraction outside [0, 1]")
    if routes.query("status == 'ok'").empty:
        raise ValueError("no successful routes")
    validate_maps(output_dir / "maps")
```

- [ ] **Step 5: Add an end-to-end test over prepared real artifacts**

```python
import json
from pathlib import Path

def test_real_output_contract(real_output_dir: Path) -> None:
    manifest = json.loads((real_output_dir / "manifest.json").read_text())
    summary = json.loads((real_output_dir / "summary.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["thermal_model"] == "SOLWEIG"
    assert manifest["hot_area_label"] == "thermal stress: UTCI > 32 C"
    assert summary["counts"]["buildings"] > 0
    assert summary["counts"]["successful_routes"] > 0
    assert len(list((real_output_dir / "maps").glob("*.png"))) == 6
```

- [ ] **Step 6: Document the exact command and analytical limits**

README run command:

```bash
cd /Users/gk/Code/super-duper-disser/thermal_access_pilot
uv sync
uv run python -m thermal_access_pilot --config configs/kaliningrad.toml
```

State plainly that the maps show modelled thermal stress, ERA5 is reanalysis, route penalties are sensitivity scenarios, physical and generalized time differ, and spatial cold wind awaits URock/PALM-4U.

- [ ] **Step 7: Run all automated verification**

Run: `cd thermal_access_pilot && uv run pytest -v`

Expected: all unit, integration, and real-artifact contract tests pass with no warnings hiding failed stages.

Run: `cd thermal_access_pilot && uv run python -m compileall -q src tests`

Expected: exit code 0.

- [ ] **Step 8: Run the complete real pipeline and inspect artifacts directly**

Run the README command. Then inspect:

- `outputs/kaliningrad/manifest.json` and `summary.json`;
- raster dimensions, bounds, min/max, nodata, and hot fraction;
- GeoParquet row counts and representative routes;
- all six PNG files at full size.

Open every PNG visually. Confirm that thermal fields, stops, buildings, routes, and legends are visible and spatially aligned. Correct rendering defects and rerun maps before completion.

- [ ] **Step 9: Commit the verified pilot**

```bash
git add thermal_access_pilot
git commit -m "Complete Kaliningrad thermal access pilot"
```

Do not stage generated `outputs/` or unrelated existing worktree changes.
