from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"


@dataclass
class PipelineSpec:
    layers: dict
    crosswalks: dict
    transfer_rules: dict
    scenarios: dict
    policy: dict

    @classmethod
    def load(cls, config_dir: Path = CONFIG_DIR) -> "PipelineSpec":
        return cls(
            layers=_load_json(config_dir / "layers.json"),
            crosswalks=_load_json(config_dir / "crosswalks.json"),
            transfer_rules=_load_json(config_dir / "transfer_rules.json"),
            scenarios=_load_json(config_dir / "scenarios.json"),
            policy=_load_json(config_dir / "policy.json"),
        )

    def validate(self) -> list[str]:
        issues: list[str] = []

        layer_ids = {layer["layer_id"] for layer in self.layers["layers"]}

        for crosswalk in self.crosswalks["crosswalks"]:
            source_layer = crosswalk["source_layer"]
            target_layer = crosswalk["target_layer"]
            if source_layer not in layer_ids:
                issues.append(
                    f"Crosswalk {crosswalk['crosswalk_id']} references missing source layer {source_layer!r}."
                )
            if target_layer not in layer_ids:
                issues.append(
                    f"Crosswalk {crosswalk['crosswalk_id']} references missing target layer {target_layer!r}."
                )

        rule_ids = {rule["rule_id"] for rule in self.transfer_rules["rules"]}
        for rule in self.transfer_rules["rules"]:
            crosswalk_id = rule["crosswalk_id"]
            if crosswalk_id not in {
                crosswalk["crosswalk_id"] for crosswalk in self.crosswalks["crosswalks"]
            }:
                issues.append(
                    f"Rule {rule['rule_id']} references missing crosswalk {crosswalk_id!r}."
                )

        for scenario in self.scenarios["scenarios"]:
            for operation in scenario["operations"]:
                if operation["kind"] == "attribute_transfer":
                    rule_id = operation["rule_id"]
                    if rule_id not in rule_ids:
                        issues.append(
                            f"Scenario {scenario['scenario_id']} references missing rule {rule_id!r}."
                        )

        return issues

    def summary(self) -> str:
        lines = [
            "Aggregated spatial pipeline specification",
            f"- layers: {len(self.layers['layers'])}",
            f"- crosswalks: {len(self.crosswalks['crosswalks'])}",
            f"- transfer rules: {len(self.transfer_rules['rules'])}",
            f"- scenarios: {len(self.scenarios['scenarios'])}",
            f"- policy: {self.policy['policy_id']}",
            "",
            "Layers:",
        ]
        for layer in self.layers["layers"]:
            lines.append(
                f"  - {layer['layer_id']}: {layer['unit_type']} ({layer['role']})"
            )
        return "\n".join(lines)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    spec = PipelineSpec.load()
    issues = spec.validate()
    print(spec.summary())
    if issues:
        print("\nValidation issues:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("\nValidation: OK")


if __name__ == "__main__":
    main()
