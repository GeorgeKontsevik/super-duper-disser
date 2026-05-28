from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "connectpt" / "datasets" / "real_morph_no_bergen_loop25_bus50"

ACTIVE_19_JOINT_INPUTS = (
    ROOT / "aggregated_spatial_pipeline" / "outputs" / "active_19_good_cities_20260412" / "joint_inputs"
)
RANDOM50_LIGHT_JOINT_INPUTS = (
    ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "batch_runs"
    / "random50_light_connectpt_20260429"
    / "joint_inputs"
)

CURRENT_CANDIDATE_CITY_DIRS = [
    ACTIVE_19_JOINT_INPUTS / "gothenburg_sweden",
    RANDOM50_LIGHT_JOINT_INPUTS / "aix_en_provence_provence_alpes_c_te_d_azur_france",
    ACTIVE_19_JOINT_INPUTS / "krakow_poland",
    ACTIVE_19_JOINT_INPUTS / "bristol_united_kingdom",
    RANDOM50_LIGHT_JOINT_INPUTS / "zagreb_zagreb_grad_croatia",
    RANDOM50_LIGHT_JOINT_INPUTS / "skopje_skopje_north_macedonia",
    ACTIVE_19_JOINT_INPUTS / "bologna_italy",
    ACTIVE_19_JOINT_INPUTS / "porto_portugal",
    RANDOM50_LIGHT_JOINT_INPUTS / "pristina_prishtin_kosovo",
    ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "old"
    / "batch_runs"
    / "random50_pop200k_10km"
    / "joint_inputs"
    / "huainan_anhui_china",
    ROOT / "aggregated_spatial_pipeline" / "outputs" / "old" / "joint_inputs" / "arequipa_peru",
    ROOT / "aggregated_spatial_pipeline" / "outputs" / "old" / "joint_inputs" / "adelaide_south_australia_australia",
]

BERGEN_EVAL_CITY_DIR = ACTIVE_19_JOINT_INPUTS / "bergen_norway"

CURRENT_GRAVITY_OK_CITIES = [
    "adelaide_south_australia_australia",
    "bologna_italy",
    "bristol_united_kingdom",
    "gothenburg_sweden",
    "huainan_anhui_china",
    "krakow_poland",
    "porto_portugal",
]

CURRENT_MISSING_BLOCKS_CITIES = [
    "aix_en_provence_provence_alpes_c_te_d_azur_france",
    "pristina_prishtin_kosovo",
    "skopje_skopje_north_macedonia",
    "zagreb_zagreb_grad_croatia",
]

CURRENT_SKIPPED_LOW_LOOP_CITIES = [
    "arequipa_peru",
]

TOPUP_CITY_PLACES = [
    "Kazan, Tatarstan, Russia",
    "Yekaterinburg, Sverdlovsk Oblast, Russia",
    "Canberra, Australian Capital Territory, Australia",
    "Hobart, Tasmania, Australia",
    "Da Nang, Vietnam",
    "Chiang Mai, Thailand",
    "Portland, Oregon, United States",
    "Pittsburgh, Pennsylvania, United States",
    "Curitiba, Parana, Brazil",
    "Valparaiso, Valparaiso, Chile",
]
