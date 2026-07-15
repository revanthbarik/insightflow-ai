"""Build a user-facing readiness report from existing CRM module metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from src.module_registry import MODULE_REGISTRY
from src.relationship_builder import PREFERRED_RELATIONSHIPS, RelationshipStatus
from src.zoho_adapter import DATE_COLUMNS, ModuleDiagnostics

NULL_WARNING_THRESHOLD = 0.6


@dataclass(frozen=True)
class ModuleHealthStatus:
    """Health details for a loaded CRM module."""

    module_name: str
    row_count: int
    required_columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]
    status: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class JoinHealthStatus:
    """Match coverage for one known module relationship."""

    left_module: str
    right_module: str
    join_key: str
    status: str
    rows_checked: int | None
    matched_rows: int | None
    unmatched_rows: int | None
    reason: str


@dataclass(frozen=True)
class CapabilityReadinessStatus:
    """Readiness display for an existing analytic capability."""

    name: str
    status: str
    reason: str


@dataclass(frozen=True)
class DataHealthReport:
    """Structured data readiness output for the Streamlit presentation layer."""

    module_health: tuple[ModuleHealthStatus, ...]
    join_health: tuple[JoinHealthStatus, ...]
    capability_health: tuple[CapabilityReadinessStatus, ...]
    warnings: tuple[str, ...]
    resolved_mappings: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def module_count(self) -> int:
        return len(self.module_health)

    @property
    def total_rows(self) -> int:
        return sum(module.row_count for module in self.module_health)

    @property
    def enabled_capability_count(self) -> int:
        return sum(capability.status == "Enabled" for capability in self.capability_health)

    def module_overview_frame(self) -> pd.DataFrame:
        """Return a compact, non-technical module health table."""
        return pd.DataFrame(
            [
                {
                    "Module": module.module_name,
                    "Rows": module.row_count,
                    "Required columns": "Complete"
                    if not module.missing_required_columns
                    else f"Missing: {', '.join(module.missing_required_columns)}",
                    "Status": module.status,
                }
                for module in self.module_health
            ]
        )

    def capability_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Capability": capability.name,
                    "Readiness": capability.status,
                    "Details": capability.reason,
                }
                for capability in self.capability_health
            ]
        )

    def join_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "Relationship": f"{join.left_module} ↔ {join.right_module}",
                    "Join key": join.join_key,
                    "Rows checked": join.rows_checked,
                    "Matched": join.matched_rows,
                    "Unmatched": join.unmatched_rows,
                    "Status": join.status,
                    "Details": join.reason,
                }
                for join in self.join_health
            ]
        )
        for column in ("Rows checked", "Matched", "Unmatched"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        return frame


def _module_quality_warnings(module_name: str, dataframe: pd.DataFrame) -> list[str]:
    """Return practical data quality warnings for one normalized module."""
    definition = MODULE_REGISTRY[module_name]
    warnings: list[str] = []

    if dataframe.empty:
        return [f"{module_name} has no rows."]

    if definition.id_column in dataframe.columns:
        duplicate_count = int(dataframe[definition.id_column].dropna().duplicated().sum())
        if duplicate_count:
            warnings.append(
                f"{module_name} has {duplicate_count} duplicate value(s) in {definition.id_column}."
            )

    important_columns = [
        column
        for column in (*definition.required_columns, *definition.optional_columns)
        if column in dataframe.columns
    ]
    for column in important_columns:
        null_ratio = float(dataframe[column].isna().mean())
        if null_ratio >= NULL_WARNING_THRESHOLD:
            warnings.append(
                f"{module_name}.{column} is {null_ratio:.0%} empty."
            )

    for column in DATE_COLUMNS:
        if column in dataframe.columns:
            invalid_count = int(dataframe[column].isna().sum())
            if invalid_count:
                warnings.append(
                    f"{module_name}.{column} has {invalid_count} blank or unparseable date value(s)."
                )
    return warnings


def _module_status(
    dataframe: pd.DataFrame,
    missing_required_columns: tuple[str, ...],
    warnings: list[str],
) -> str:
    if missing_required_columns or dataframe.empty:
        return "Warning"
    if warnings:
        return "Partial"
    return "Healthy"


def _build_join_health(
    loaded_modules: dict[str, pd.DataFrame],
    relationships: Iterable[RelationshipStatus],
) -> tuple[JoinHealthStatus, ...]:
    relationship_lookup = {
        (relationship.left_module, relationship.right_module): relationship
        for relationship in relationships
    }
    join_health: list[JoinHealthStatus] = []

    for left_module, right_module, join_key in PREFERRED_RELATIONSHIPS:
        relationship = relationship_lookup.get((left_module, right_module))
        left_df = loaded_modules.get(left_module)
        right_df = loaded_modules.get(right_module)

        if left_df is None or right_df is None:
            join_health.append(
                JoinHealthStatus(
                    left_module,
                    right_module,
                    join_key,
                    "Not available",
                    None,
                    None,
                    None,
                    "One or both modules are not loaded.",
                )
            )
            continue
        if join_key not in left_df.columns or join_key not in right_df.columns:
            join_health.append(
                JoinHealthStatus(
                    left_module,
                    right_module,
                    join_key,
                    "Warning",
                    None,
                    None,
                    None,
                    f"Missing join key `{join_key}` in one or both modules.",
                )
            )
            continue

        left_keys = left_df[join_key].dropna()
        right_keys = right_df[join_key].dropna()
        rows_checked = len(left_keys)
        matched_rows = int(left_keys.isin(set(right_keys)).sum())
        unmatched_rows = rows_checked - matched_rows
        lookup_has_duplicates = right_keys.duplicated().any()
        if lookup_has_duplicates:
            status = "Warning"
            reason = f"Lookup-side {join_key} has duplicates."
        elif rows_checked == 0:
            status = "Warning"
            reason = f"No non-empty {join_key} values are available to check."
        elif unmatched_rows:
            status = "Partial"
            reason = f"{unmatched_rows} row(s) do not resolve to {right_module}."
        else:
            status = "Healthy"
            reason = f"All checked {join_key} values resolve to {right_module}."

        if relationship and not relationship.available and status == "Healthy":
            status = "Warning"
            reason = relationship.reason
        join_health.append(
            JoinHealthStatus(
                left_module,
                right_module,
                join_key,
                status,
                rows_checked,
                matched_rows,
                unmatched_rows,
                reason,
            )
        )
    return tuple(join_health)


def build_data_health_report(
    loaded_modules: dict[str, pd.DataFrame],
    capabilities: Iterable[object],
    relationships: Iterable[RelationshipStatus],
    module_diagnostics: dict[str, ModuleDiagnostics] | None = None,
) -> DataHealthReport:
    """Build readiness details while reusing existing module and capability systems."""
    diagnostics = module_diagnostics or {}
    all_warnings: list[str] = []
    module_health: list[ModuleHealthStatus] = []
    resolved_mappings: dict[str, dict[str, str]] = {}

    for module_name, dataframe in loaded_modules.items():
        definition = MODULE_REGISTRY.get(module_name)
        if definition is None:
            continue
        missing_required = tuple(
            sorted(set(definition.required_columns).difference(dataframe.columns))
        )
        warnings = _module_quality_warnings(module_name, dataframe)
        diagnostic = diagnostics.get(module_name)
        if diagnostic:
            warnings.extend(
                warning
                for warning in (
                    f"{module_name} is missing required columns: {', '.join(diagnostic.missing_required_columns)}."
                    if diagnostic.missing_required_columns
                    else None,
                )
                if warning
            )
            if diagnostic.mapped_columns:
                resolved_mappings[module_name] = diagnostic.mapped_columns

        status = _module_status(dataframe, missing_required, warnings)
        module_health.append(
            ModuleHealthStatus(
                module_name=module_name,
                row_count=len(dataframe),
                required_columns=definition.required_columns,
                missing_required_columns=missing_required,
                status=status,
                warnings=tuple(warnings),
            )
        )
        all_warnings.extend(warnings)

    join_health = _build_join_health(loaded_modules, relationships)
    for join in join_health:
        if join.status in {"Partial", "Warning"}:
            all_warnings.append(
                f"{join.left_module} ↔ {join.right_module}: {join.reason}"
            )

    capability_health = tuple(
        CapabilityReadinessStatus(
            name=capability.name,
            status="Enabled" if capability.available else "Disabled",
            reason=capability.reason,
        )
        for capability in capabilities
    )
    return DataHealthReport(
        module_health=tuple(module_health),
        join_health=join_health,
        capability_health=capability_health,
        warnings=tuple(dict.fromkeys(all_warnings)),
        resolved_mappings=resolved_mappings,
    )
