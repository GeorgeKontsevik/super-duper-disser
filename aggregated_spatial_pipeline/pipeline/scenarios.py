from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd

from aggregated_spatial_pipeline.spec import PipelineSpec

from .transfers import apply_transfer_rule


@dataclass
class ScenarioResult:
    scenario_id: str
    quarters: gpd.GeoDataFrame
    cities: gpd.GeoDataFrame
    metadata: dict


def run_scenarios(
    *,
    spec: PipelineSpec,
    layers: dict[str, gpd.GeoDataFrame],
    crosswalks: dict[str, gpd.GeoDataFrame],
) -> dict[str, ScenarioResult]:
    results: dict[str, ScenarioResult] = {}
    rule_by_id = {rule["rule_id"]: rule for rule in spec.transfer_rules["rules"]}
    scenario_by_id = {scenario["scenario_id"]: scenario for scenario in spec.scenarios["scenarios"]}
    crosswalk_by_id = {crosswalk["crosswalk_id"]: crosswalk for crosswalk in spec.crosswalks["crosswalks"]}

    city_rules = [
        rule
        for rule in spec.transfer_rules["rules"]
        if crosswalk_by_id[rule["crosswalk_id"]]["source_layer"] == "quarters"
        and crosswalk_by_id[rule["crosswalk_id"]]["target_layer"] == "cities"
    ]

    def materialize(scenario_id: str) -> ScenarioResult:
        if scenario_id in results:
            return results[scenario_id]

        scenario = scenario_by_id[scenario_id]
        quarters = layers["quarters"].copy()
        cities = layers["cities"].copy()
        metadata = {"applied_operations": [], "pending_operations": []}

        for operation in scenario["operations"]:
            if operation["kind"] == "copy_from":
                parent = materialize(operation["scenario_id"])
                quarters = parent.quarters.copy()
                cities = parent.cities.copy()
                metadata["applied_operations"].append({"kind": "copy_from", "scenario_id": parent.scenario_id})
                continue

            if operation["kind"] == "attribute_transfer":
                rule = rule_by_id[operation["rule_id"]]
                crosswalk = next(
                    item for item in spec.crosswalks["crosswalks"] if item["crosswalk_id"] == rule["crosswalk_id"]
                )
                source_layer = crosswalk["source_layer"]
                target_layer = crosswalk["target_layer"]
                target_gdf = quarters if target_layer == "quarters" else cities
                updated_target = apply_transfer_rule(
                    source_gdf=layers[source_layer] if source_layer != "quarters" else quarters,
                    target_gdf=target_gdf,
                    crosswalk_gdf=crosswalks[crosswalk["crosswalk_id"]],
                    source_layer=source_layer,
                    target_layer=target_layer,
                    attribute=rule["attribute"],
                    aggregation_method=rule["aggregation_method"],
                    weight_field=rule["weight_field"],
                )
                if target_layer == "quarters":
                    quarters = updated_target
                else:
                    cities = updated_target
                metadata["applied_operations"].append(operation)
                continue

            metadata["pending_operations"].append(operation)

        for rule in city_rules:
            crosswalk = crosswalk_by_id[rule["crosswalk_id"]]
            cities = apply_transfer_rule(
                source_gdf=quarters,
                target_gdf=cities,
                crosswalk_gdf=crosswalks[crosswalk["crosswalk_id"]],
                source_layer="quarters",
                target_layer="cities",
                attribute=rule["attribute"],
                aggregation_method=rule["aggregation_method"],
                weight_field=rule["weight_field"],
            )
            metadata["applied_operations"].append(
                {
                    "kind": "attribute_transfer",
                    "rule_id": rule["rule_id"],
                    "auto_generated": True,
                }
            )

        results[scenario_id] = ScenarioResult(
            scenario_id=scenario_id,
            quarters=quarters,
            cities=cities,
            metadata=metadata,
        )
        return results[scenario_id]

    for scenario in spec.scenarios["scenarios"]:
        materialize(scenario["scenario_id"])

    return results
