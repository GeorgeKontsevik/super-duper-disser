from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import pandas as pd


def resolve_attribute_columns(source_gdf: gpd.GeoDataFrame, attribute: str) -> list[str]:
    if attribute == "street_pattern_probs":
        probability_columns = [column for column in source_gdf.columns if column.startswith("prob_")]
        if not probability_columns:
            raise KeyError("No probability columns found for attribute 'street_pattern_probs'.")
        return probability_columns

    if attribute == "street_pattern_class":
        for candidate in ("class_name", "street_pattern_class", "predicted_class"):
            if candidate in source_gdf.columns:
                return [candidate]
        raise KeyError("No class column found for attribute 'street_pattern_class'.")

    if attribute not in source_gdf.columns:
        raise KeyError(f"Attribute {attribute!r} not present in source layer.")
    return [attribute]


def apply_transfer_rule(
    *,
    source_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    crosswalk_gdf: gpd.GeoDataFrame,
    source_layer: str,
    target_layer: str,
    attribute: str,
    aggregation_method: str,
    weight_field: str,
) -> gpd.GeoDataFrame:
    source_id = f"{source_layer}_id"
    target_id = f"{target_layer}_id"
    attribute_columns = resolve_attribute_columns(source_gdf, attribute)

    source_attributes = source_gdf[[source_id, *attribute_columns]].copy()
    joined = crosswalk_gdf[[source_id, target_id, weight_field]].merge(source_attributes, on=source_id, how="left")

    result = target_gdf.copy()

    if aggregation_method == "weighted_mean":
        for attribute_column in attribute_columns:
            result[attribute_column] = _weighted_mean(joined, target_id, attribute_column, weight_field)
        return result

    if aggregation_method == "majority_vote":
        attribute_column = attribute_columns[0]
        result[attribute] = _majority_vote(joined, target_id, attribute_column, weight_field)
        return result

    if aggregation_method == "sum":
        attribute_column = attribute_columns[0]
        result[attribute] = _weighted_sum(joined, target_id, attribute_column, weight_field)
        return result

    raise NotImplementedError(f"Unsupported aggregation method: {aggregation_method}")


def _weighted_mean(joined: pd.DataFrame, target_id: str, attribute_column: str, weight_field: str) -> pd.Series:
    non_null = joined.dropna(subset=[attribute_column, weight_field]).copy()
    weighted = non_null[attribute_column] * non_null[weight_field]
    grouped_weighted = weighted.groupby(non_null[target_id]).sum()
    grouped_weights = non_null.groupby(target_id)[weight_field].sum()
    return grouped_weighted / grouped_weights


def _weighted_sum(joined: pd.DataFrame, target_id: str, attribute_column: str, weight_field: str) -> pd.Series:
    non_null = joined.dropna(subset=[attribute_column, weight_field]).copy()
    weighted = non_null[attribute_column] * non_null[weight_field]
    return weighted.groupby(non_null[target_id]).sum()


def _majority_vote(joined: pd.DataFrame, target_id: str, attribute_column: str, weight_field: str) -> pd.Series:
    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in joined.dropna(subset=[attribute_column, weight_field]).iterrows():
        scores[row[target_id]][str(row[attribute_column])] += float(row[weight_field])

    winners = {}
    for target_value, class_scores in scores.items():
        winners[target_value] = max(class_scores.items(), key=lambda item: item[1])[0]
    return pd.Series(winners, name=attribute_column)
