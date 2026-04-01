from __future__ import annotations

from collections import defaultdict
import re

import geopandas as gpd
import pandas as pd


def resolve_attribute_columns(source_gdf: gpd.GeoDataFrame, attribute: str) -> list[str]:
    if attribute == "street_pattern_class_shares":
        for candidate in ("top1_class_name", "class_name", "street_pattern_class", "predicted_class"):
            if candidate in source_gdf.columns:
                return [candidate]
        raise KeyError("No class column found for attribute 'street_pattern_class_shares'.")

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
    joined = _prepare_joined_crosswalk(
        crosswalk_gdf=crosswalk_gdf,
        source_attributes=source_attributes,
        source_id=source_id,
        target_id=target_id,
        weight_field=weight_field,
    )

    result = target_gdf.copy()

    if aggregation_method == "weighted_mean":
        for attribute_column in attribute_columns:
            aggregated = _weighted_mean(joined, target_id, attribute_column, weight_field)
            result[attribute_column] = result[target_id].map(aggregated)
        return result

    if aggregation_method == "majority_vote":
        attribute_column = attribute_columns[0]
        aggregated = _majority_vote(joined, target_id, attribute_column, weight_field)
        result[attribute] = result[target_id].map(aggregated)
        return result

    if aggregation_method == "sum":
        attribute_column = attribute_columns[0]
        aggregated = _weighted_sum(joined, target_id, attribute_column, weight_field)
        result[attribute] = result[target_id].map(aggregated)
        return result

    if aggregation_method == "class_area_shares":
        attribute_column = attribute_columns[0]
        return _class_area_shares(
            joined=joined,
            result=result,
            target_id=target_id,
            attribute_column=attribute_column,
            weight_field=weight_field,
            prefix=attribute,
        )

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


def _class_area_shares(
    *,
    joined: pd.DataFrame,
    result: gpd.GeoDataFrame,
    target_id: str,
    attribute_column: str,
    weight_field: str,
    prefix: str,
) -> gpd.GeoDataFrame:
    non_null = joined.dropna(subset=[attribute_column, weight_field]).copy()
    if non_null.empty:
        return result

    non_null[attribute_column] = non_null[attribute_column].astype(str)
    class_share = (
        non_null.groupby([target_id, attribute_column])[weight_field]
        .sum()
        .unstack(fill_value=0.0)
    )
    if class_share.empty:
        return result

    for class_name in class_share.columns:
        safe_name = _slugify_class_name(class_name)
        column_name = f"{prefix}_{safe_name}"
        result[column_name] = result[target_id].map(class_share[class_name]).fillna(0.0)
    return result


def _slugify_class_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    normalized = normalized.strip("_")
    return normalized or "unknown"


def _prepare_joined_crosswalk(
    *,
    crosswalk_gdf: gpd.GeoDataFrame,
    source_attributes: pd.DataFrame,
    source_id: str,
    target_id: str,
    weight_field: str,
) -> pd.DataFrame:
    required_columns = [source_id, target_id]
    for candidate_weight in ("intersection_area", "source_share", "target_share", weight_field):
        if candidate_weight in crosswalk_gdf.columns and candidate_weight not in required_columns:
            required_columns.append(candidate_weight)

    joined = crosswalk_gdf[required_columns].merge(source_attributes, on=source_id, how="left")
    _ensure_weight_field(joined, weight_field)
    return joined


def _ensure_weight_field(joined: pd.DataFrame, weight_field: str) -> None:
    if weight_field in joined.columns:
        return

    if (
        weight_field == "population_weight"
        and "population_total" in joined.columns
        and "source_share" in joined.columns
    ):
        joined[weight_field] = joined["population_total"] * joined["source_share"]
        return

    for fallback in ("intersection_area", "source_share", "target_share"):
        if fallback in joined.columns:
            joined[weight_field] = joined[fallback]
            return

    raise KeyError(f"Weight field {weight_field!r} is not available and no fallback can be derived.")
