from __future__ import annotations

import hashlib
import json
import re
import shutil
import textwrap
from pathlib import Path

SLICE = "27A/27B"

ROOT = Path.cwd()


def find_root() -> Path:
    here = Path.cwd()
    candidates = [here, *here.parents]
    for candidate in candidates:
        if (candidate / "src" / "pegasus").is_dir() and (candidate / "config" / "registries").is_dir():
            return candidate
    raise SystemExit("Could not locate PegaSUS repository root. Run this script from inside C:\\Users\\Galaxy\\LEVI\\PegaSUS.")


ROOT = find_root()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8", newline="\n")


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def patch(path: str, transform) -> None:
    original = read(path)
    updated = transform(original)
    if updated != original:
        (ROOT / path).write_text(updated, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


REGISTRY_CONTENT: dict[str, str] = {
    "health/diagnostic_topology.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A diagnostic topology registry. Defines ICD-bearing fields as topology-bearing observer/event selectors, not generic additive numeric measures.
        entries:
        - id: sim_underlying_cause_causabas
          status: active
          source_system: SIM-DO
          field_patterns: [CAUSABAS, causa_basica, underlying_cause]
          topology_role: underlying_cause
          icd_system: ICD-10
          field_kind: observer_proxy
          carrier: Deaths
          unit: ICD10
          aggregation: non_aggregable
          allowed_operators: [diagnostic_topology_annotation, icd_chapter_projection, icd_block_projection, icd_leaf_count]
          forbidden_operators: [numeric_sum, generic_rate_numerator, generic_cost_component]
          warnings: [diagnostic_topology_not_numeric_measure]
        - id: sim_causal_chain_linhas
          status: active
          source_system: SIM-DO
          field_patterns: [LINHAA, LINHAB, LINHAC, LINHAD, LINHAII, linha]
          topology_role: causal_chain
          icd_system: ICD-10
          field_kind: observer_proxy
          carrier: Deaths
          unit: ICD10
          aggregation: non_aggregable
          allowed_operators: [diagnostic_topology_annotation, icd_chapter_projection, icd_block_projection]
          forbidden_operators: [numeric_sum, generic_rate_numerator]
          warnings: [causal_chain_not_underlying_cause]
        - id: sih_principal_diagnosis
          status: active
          source_system: SIH-RD
          field_patterns: [DIAG_PRINC, diag_princ, principal_diagnosis]
          topology_role: principal_diagnosis
          icd_system: ICD-10
          field_kind: observer_proxy
          carrier: HospitalAdmissions
          unit: ICD10
          aggregation: non_aggregable
          allowed_operators: [diagnostic_topology_annotation, icd_chapter_projection, icd_block_projection, admission_diagnosis_count]
          forbidden_operators: [numeric_sum, generic_cost_component]
          warnings: [hospital_diagnosis_topology]
        - id: sih_secondary_diagnosis
          status: active
          source_system: SIH-RD
          field_patterns: [DIAG_SECUN, diag_secun, secondary_diagnosis]
          topology_role: secondary_diagnosis
          icd_system: ICD-10
          field_kind: observer_proxy
          carrier: HospitalAdmissions
          unit: ICD10
          aggregation: non_aggregable
          allowed_operators: [diagnostic_topology_annotation, icd_chapter_projection, icd_block_projection]
          forbidden_operators: [numeric_sum, generic_cost_component]
          warnings: [secondary_diagnosis_sparse_optional]
        - id: sinasc_congenital_anomaly_icd
          status: active
          source_system: SINASC
          field_patterns: [CODANOMAL, anomaly_icd, congenital_anomaly]
          topology_role: congenital_anomaly
          icd_system: ICD-10-Q
          field_kind: observer_proxy
          carrier: LiveBirths
          unit: ICD10
          aggregation: non_aggregable
          allowed_operators: [diagnostic_topology_annotation, icd_chapter_projection, icd_block_projection, anomaly_prevalence_numerator]
          forbidden_operators: [numeric_sum, generic_rate_numerator_without_birth_denominator]
          warnings: [congenital_anomaly_requires_birth_denominator]
    ''',
    "health/cnes_capacity_registry.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A CNES-ST capacity registry. Protects indexed facility-capacity components from generic count semantics.
        entries:
        - id: cnes_facility_stock
          status: active
          source_system: CNES-ST
          field_patterns: [CNES, facility_id, estabelecimento, TP_UNID]
          capacity_family: facility_stock
          carrier: Facilities
          unit: counts
          aggregation: additive
          support_axes: [facility, municipality, period]
          allowed_operators: [facility_count, facility_type_stratification, capacity_density_with_population]
          forbidden_operators: [generic_admission_count, sih_cost_sum]
          protected_non_equivalence: [facility_type, sus_binding, management_regime]
          warnings: []
        - id: cnes_bed_capacity_qtleit
          status: active
          source_system: CNES-ST
          field_patterns: [QTLEIT, LEITOS, beds, bed_capacity]
          capacity_family: bed_capacity
          capacity_index: QTLEIT
          carrier: Facilities
          unit: beds
          aggregation: additive
          support_axes: [facility, municipality, period, bed_type]
          allowed_operators: [bed_capacity_sum, bed_capacity_density, admission_per_bed_ratio]
          forbidden_operators: [facility_count_equivalence, generic_population_denominator]
          protected_non_equivalence: [bed_type, sus_binding, specialty]
          warnings: [capacity_component_identity_protected]
        - id: cnes_installation_capacity_qtinst
          status: active
          source_system: CNES-ST
          field_patterns: [QTINST, INSTAL, installation_capacity, equipment_capacity]
          capacity_family: installation_capacity
          capacity_index: QTINST
          carrier: Facilities
          unit: counts
          aggregation: additive
          support_axes: [facility, municipality, period, installation_type]
          allowed_operators: [installation_capacity_sum, capacity_density]
          forbidden_operators: [bed_capacity_equivalence, facility_count_equivalence]
          protected_non_equivalence: [installation_type, equipment_type]
          warnings: [capacity_component_identity_protected]
        - id: cnes_sus_binding
          status: active
          source_system: CNES-ST
          field_patterns: [VINC_SUS, sus_binding]
          capacity_family: facility_attribute
          carrier: Facilities
          unit: binary_flag
          aggregation: non_aggregable
          support_axes: [facility, municipality, period]
          allowed_operators: [facility_filter, sus_capacity_stratification]
          forbidden_operators: [numeric_sum_without_filter_semantics]
          protected_non_equivalence: [sus_binding]
          warnings: [facility_attribute_not_capacity_measure]
    ''',
    "health/sih_cost_registry.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A SIH-RD cost-component registry. Protects AIH cost components from unsafe generic BRL collapse.
        entries:
        - id: sih_total_cost_val_tot
          status: active
          source_system: SIH-RD
          field_patterns: [VAL_TOT, total_cost, valor_total]
          cost_component: total
          carrier: HospitalAdmissions
          unit: BRL
          aggregation: additive
          allowed_operators: [cost_sum, cost_per_admission, cost_density_population]
          forbidden_operators: [component_equivalence_with_professional_or_hospital_service]
          protected_non_equivalence: [VAL_SH, VAL_SP, VAL_SADT, VAL_RN, VAL_UTI]
          warnings: [sih_cost_component_identity_protected]
        - id: sih_hospital_service_val_sh
          status: active
          source_system: SIH-RD
          field_patterns: [VAL_SH, hospital_service, servico_hospitalar]
          cost_component: hospital_service
          carrier: HospitalAdmissions
          unit: BRL
          aggregation: additive
          allowed_operators: [cost_component_sum, cost_component_share]
          forbidden_operators: [total_cost_equivalence, professional_service_equivalence]
          protected_non_equivalence: [VAL_TOT, VAL_SP]
          warnings: [sih_cost_component_identity_protected]
        - id: sih_professional_service_val_sp
          status: active
          source_system: SIH-RD
          field_patterns: [VAL_SP, professional_service, servico_profissional]
          cost_component: professional_service
          carrier: HospitalAdmissions
          unit: BRL
          aggregation: additive
          allowed_operators: [cost_component_sum, cost_component_share]
          forbidden_operators: [total_cost_equivalence, hospital_service_equivalence]
          protected_non_equivalence: [VAL_TOT, VAL_SH]
          warnings: [sih_cost_component_identity_protected]
        - id: sih_uti_cost_component
          status: active
          source_system: SIH-RD
          field_patterns: [VAL_UTI, UTI, uti_cost]
          cost_component: intensive_care
          carrier: HospitalAdmissions
          unit: BRL
          aggregation: additive
          allowed_operators: [uti_cost_sum, uti_cost_per_admission]
          forbidden_operators: [generic_total_cost_equivalence]
          protected_non_equivalence: [VAL_TOT, VAL_SH, VAL_SP]
          warnings: [sih_cost_component_identity_protected]
    ''',
    "health/clinical_event_definitions.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A clinical event definitions for source-agnostic EFG event carriers.
        entries:
        - id: sim_all_cause_death_event
          status: active
          source_system: SIM-DO
          event_carrier: Deaths
          numerator_unit: counts
          denominator_carriers: [Population, LiveBirths]
          default_aggregation: additive
          allowed_rates: [mortality_rate, infant_mortality_rate, neonatal_mortality_rate]
          warnings: []
        - id: sinasc_live_birth_event
          status: active
          source_system: SINASC
          event_carrier: LiveBirths
          numerator_unit: counts
          denominator_carriers: [Population]
          default_aggregation: additive
          allowed_rates: [crude_birth_rate, anomaly_prevalence, low_birth_weight_prevalence]
          warnings: []
        - id: sih_hospital_admission_event
          status: active
          source_system: SIH-RD
          event_carrier: HospitalAdmissions
          numerator_unit: counts
          denominator_carriers: [Population, Facilities]
          default_aggregation: additive
          allowed_rates: [hospitalization_rate, admission_per_bed_ratio]
          warnings: [administrative_admission_not_incidence]
        - id: cnes_facility_capacity_event
          status: active
          source_system: CNES-ST
          event_carrier: Facilities
          numerator_unit: counts
          denominator_carriers: [Population]
          default_aggregation: additive
          allowed_rates: [facility_density, bed_density]
          warnings: [capacity_snapshot_period_semantics]
    ''',
    "fields/bridge_grammars.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A bridge grammar registry for EFG declaration, support, and denominator bridge legality.
        entries:
        - id: race_admin_to_posterior_bridge
          status: active
          bridge_type: race_axis
          from_axis: administrative_race_color
          to_axis: posterior_self_declared_race_color
          operators: [Bridge_R, Bridge_R_fixedC_dynamicW, Bridge_R_posteriorC]
          required_metadata: [race_bridge_prior, posterior_uncertainty, missing_race_observer]
          warnings: [race_bridge_uncertainty_required]
        - id: geography_geneallocation_bridge
          status: active_artifact_required
          bridge_type: geospatial_support
          operators: [geneallocation_measure_pushforward, amc_contraction]
          required_metadata: [crosswalk_manifest_hash, mass_conservation_proof]
          forbidden_payloads: [direct_rate_allocation]
          warnings: [geospatial_artifact_required]
        - id: population_denominator_bridge
          status: active
          bridge_type: denominator_support
          operators: [RN, crude_rate, denominator_link]
          required_metadata: [support_intersection, denominator_fragility]
          warnings: []
        - id: facility_admission_support_bridge
          status: active_warning
          bridge_type: facility_admission_support
          operators: [admission_per_bed_ratio, admission_capacity_context]
          required_metadata: [facility_period_support, admission_period_support]
          warnings: [facility_capacity_not_case_denominator]
    ''',
    "carrier_registry.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A compatibility alias. Canonical carrier semantics live in ontology/carrier.yaml; this file remains required by registry validators.
        entries:
        - id: carrier_registry_alias
          status: active_alias
          canonical_registry: ontology/carrier.yaml
          description: Compatibility manifest for the canonical carrier registry.
          warnings: [canonical_registry_is_carrier_yaml]
    ''',
    "unit_registry.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A compatibility alias. Canonical unit semantics live in ontology/unit.yaml; this file remains required by registry validators.
        entries:
        - id: unit_registry_alias
          status: active_alias
          canonical_registry: ontology/unit.yaml
          description: Compatibility manifest for the canonical unit registry.
          warnings: [canonical_registry_is_unit_yaml]
    ''',
    "aggregation_registry.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A compatibility alias. Canonical aggregation semantics live in ontology/aggregation.yaml; this file remains required by registry validators.
        entries:
        - id: aggregation_registry_alias
          status: active_alias
          canonical_registry: ontology/aggregation.yaml
          description: Compatibility manifest for the canonical aggregation registry.
          warnings: [canonical_registry_is_aggregation_yaml]
    ''',
    "provenance_registry.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A compatibility alias. Canonical provenance semantics live in ontology/provenance.yaml; this file remains required by registry validators.
        entries:
        - id: provenance_registry_alias
          status: active_alias
          canonical_registry: ontology/provenance.yaml
          description: Compatibility manifest for the canonical provenance registry.
          warnings: [canonical_registry_is_provenance_yaml]
    ''',
    "demographic/race_axis_registry.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A race-axis declaration registry.
        entries:
        - id: administrative_race_color
          status: active_warning
          declaration_process: administrative_observed_or_recorded
          categories: [branca, preta, amarela, parda, indigena, ignored]
          allowed_direct_comparisons: [administrative_race_color]
          bridge_required_to: [posterior_self_declared_race_color, census_self_declared_race_color]
          warnings: [administrative_race_not_self_declared]
        - id: posterior_self_declared_race_color
          status: active
          declaration_process: bridge_posterior_distribution
          categories: [branca, preta, amarela, parda, indigena]
          allowed_direct_comparisons: [posterior_self_declared_race_color]
          bridge_required_to: [administrative_race_color]
          warnings: [race_bridge_posterior_uncertainty]
        - id: missing_race_observer
          status: active_observer
          declaration_process: missingness_observer
          categories: [missing, ignored, not_recorded]
          allowed_direct_comparisons: [missing_race_observer]
          bridge_required_to: []
          warnings: [missingness_not_population_race_axis]
    ''',
    "health/icd_catalog.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A minimal ICD-10 catalog sufficient for topology validation and chapter/block projection.
        entries:
        - id: icd10_chapter_I
          status: active
          range: [A00, B99]
          label: Certain infectious and parasitic diseases
          warnings: []
        - id: icd10_chapter_II
          status: active
          range: [C00, D48]
          label: Neoplasms
          warnings: []
        - id: icd10_chapter_IX
          status: active
          range: [I00, I99]
          label: Diseases of the circulatory system
          warnings: []
        - id: icd10_chapter_X
          status: active
          range: [J00, J99]
          label: Diseases of the respiratory system
          warnings: []
        - id: icd10_chapter_XVI
          status: active
          range: [P00, P96]
          label: Certain conditions originating in the perinatal period
          warnings: []
        - id: icd10_chapter_XVII
          status: active
          range: [Q00, Q99]
          label: Congenital malformations, deformations and chromosomal abnormalities
          warnings: []
        - id: icd10_chapter_XX
          status: active
          range: [V01, Y98]
          label: External causes of morbidity and mortality
          warnings: []
    ''',
    "health/icd_quality_groups.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A ICD quality grouping registry for diagnostic topology warnings.
        entries:
        - id: icd10_well_formed
          status: active
          predicate: normalized_icd10_code
          warnings: []
        - id: icd10_missing
          status: active_observer
          predicate: missing_or_blank_code
          warnings: [diagnostic_code_missing]
        - id: icd10_ill_defined
          status: active_warning
          predicate: R00_R99_or_unspecified
          warnings: [ill_defined_condition_group]
        - id: icd10_external_cause
          status: active
          predicate: V01_Y98
          warnings: [external_cause_topology]
    ''',
    "fields/join_affordances.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A support join affordance registry.
        entries:
        - id: sim_sidra_population_denominator
          status: active
          left_system: SIM-DO
          right_system: SIDRA
          support_axes: [municipality, year]
          allowed_operators: [RN, crude_mortality_rate]
          warnings: []
        - id: sinasc_sidra_population_denominator
          status: active
          left_system: SINASC
          right_system: SIDRA
          support_axes: [municipality, year]
          allowed_operators: [crude_birth_rate]
          warnings: []
        - id: sim_sinasc_birth_denominator
          status: active
          left_system: SIM-DO
          right_system: SINASC
          support_axes: [municipality, year]
          allowed_operators: [infant_mortality_rate, neonatal_mortality_rate, postneonatal_mortality_rate]
          warnings: [birth_denominator_linkage]
        - id: sih_cnes_capacity_context
          status: active_warning
          left_system: SIH-RD
          right_system: CNES-ST
          support_axes: [municipality, period]
          allowed_operators: [admission_per_bed_ratio, facility_capacity_context]
          warnings: [capacity_context_not_case_denominator]
    ''',
    "spatial/municipality_crosswalk_sources.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A geospatial source registry. External crosswalk artifacts remain required for non-native support transformations.
        entries:
        - id: ibge_municipality_crosswalk_external_required
          status: active_artifact_required
          geography_system: IBGE
          source_code_system: IBGE7
          target_code_system: IBGE7
          allowed_operators: [amc_contraction, geneallocation_measure_pushforward]
          required_artifacts: [crosswalk_matrix, source_manifest_hash, mass_conservation_proof]
          forbidden_operators: [direct_rate_allocation]
          warnings: [external_crosswalk_artifact_required]
    ''',
    "inference/model_registry.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A model registry aligned to PIRS execution surfaces.
        entries:
        - id: pirs_ols_residual_model
          status: active
          family: gaussian_ols
          required_artifacts: [pirs_design_matrix_manifest]
          outputs: [coefficients, fitted_values, residual_values]
          warnings: []
        - id: pirs_poisson_offset_model
          status: planned_contract
          family: poisson_log_offset
          required_artifacts: [pirs_design_matrix_manifest, denominator_offset]
          outputs: [coefficients, fitted_values, residual_values]
          warnings: [requires_offset_contract]
    ''',
    "inference/residual_registry.yaml": r'''
        schema_version: '1.0'
        registry_version: v2.0
        created_at: '2026-06-06'
        updated_at: '2026-06-13'
        provenance: Macro-Slice 27A residual registry aligned to PIRS/HSIC residual scanner.
        entries:
        - id: pirs_ols_raw_residual
          status: active
          residual_type: observed_minus_fitted
          allowed_scanners: [exact_centered_hsic_v1, nystrom_hsic_v1, rff_hsic_v1]
          warnings: []
        - id: pirs_standardized_residual
          status: active_warning
          residual_type: standardized_observed_minus_fitted
          allowed_scanners: [exact_centered_hsic_v1, nystrom_hsic_v1, rff_hsic_v1]
          warnings: [standardization_depends_on_model_variance]
    ''',
}


REGISTRY_MODULES: dict[str, str] = {
    "src/pegasus/registries/semantic.py": r'''
        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        from pegasus.registries.loader import load_registry_file


        def registry_root_path(root: str | Path = "config/registries") -> Path:
            return Path(root)


        def registry_entries(name: str, *, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
            payload = load_registry_file(Path(registry_root) / name)
            entries = payload.get("entries", [])
            return [entry for entry in entries if isinstance(entry, dict)]


        def active_entries(name: str, *, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
            entries = registry_entries(name, registry_root=registry_root)
            return [entry for entry in entries if str(entry.get("status", "")).startswith("active") or str(entry.get("status", "")) in {"stable", "planned_contract"}]


        def field_text(field: Any) -> str:
            parts: list[str] = []
            for attr in (
                "field_id",
                "name",
                "source",
                "source_json",
                "role",
                "role_json",
                "metadata_json",
                "operator",
                "carrier",
                "unit",
                "aggregation",
            ):
                value = getattr(field, attr, None)
                if value is not None:
                    parts.append(str(value))
            try:
                dump = field.model_dump()
            except Exception:
                dump = None
            if isinstance(dump, dict):
                parts.extend(str(value) for value in dump.values() if value is not None)
            return " ".join(parts).upper()


        def match_entry(field: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
            haystack = field_text(field)
            for entry in entries:
                patterns = entry.get("field_patterns", []) or []
                if any(str(pattern).upper() in haystack for pattern in patterns):
                    return entry
                entry_id = str(entry.get("id", "")).upper()
                if entry_id and entry_id in haystack:
                    return entry
            return None


        def registry_is_scaffold_only(name: str, *, registry_root: str | Path = "config/registries") -> bool:
            entries = registry_entries(name, registry_root=registry_root)
            if not entries:
                return True
            if len(entries) == 1:
                entry = entries[0]
                warnings = {str(value) for value in entry.get("warnings", []) or []}
                status = str(entry.get("status", ""))
                return status == "deferred" and "scaffold_only" in warnings
            return all("scaffold_only" in {str(value) for value in entry.get("warnings", []) or []} for entry in entries)
    ''',
    "src/pegasus/registries/diagnostic_topology.py": r'''
        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        from pegasus.registries.semantic import active_entries, match_entry


        REGISTRY_FILE = "health/diagnostic_topology.yaml"


        def diagnostic_topology_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
            return active_entries(REGISTRY_FILE, registry_root=registry_root)


        def diagnostic_topology_for_field(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
            return match_entry(field, diagnostic_topology_entries(registry_root=registry_root))


        def is_diagnostic_topology_field(field: Any, *, registry_root: str | Path = "config/registries") -> bool:
            return diagnostic_topology_for_field(field, registry_root=registry_root) is not None


        def diagnostic_evidence(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
            entry = diagnostic_topology_for_field(field, registry_root=registry_root)
            if entry is None:
                return None
            return {
                "registry": REGISTRY_FILE,
                "entry_id": entry.get("id"),
                "topology_role": entry.get("topology_role"),
                "unit": entry.get("unit"),
                "aggregation": entry.get("aggregation"),
                "forbidden_operators": list(entry.get("forbidden_operators", []) or []),
            }
    ''',
    "src/pegasus/registries/cnes_capacity.py": r'''
        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        from pegasus.registries.semantic import active_entries, match_entry


        REGISTRY_FILE = "health/cnes_capacity_registry.yaml"


        def capacity_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
            return active_entries(REGISTRY_FILE, registry_root=registry_root)


        def capacity_entry_for_field(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
            return match_entry(field, capacity_entries(registry_root=registry_root))


        def capacity_evidence(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
            entry = capacity_entry_for_field(field, registry_root=registry_root)
            if entry is None:
                return None
            return {
                "registry": REGISTRY_FILE,
                "entry_id": entry.get("id"),
                "capacity_family": entry.get("capacity_family"),
                "capacity_index": entry.get("capacity_index"),
                "protected_non_equivalence": list(entry.get("protected_non_equivalence", []) or []),
            }
    ''',
    "src/pegasus/registries/sih_cost.py": r'''
        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        from pegasus.registries.semantic import active_entries, match_entry


        REGISTRY_FILE = "health/sih_cost_registry.yaml"


        def cost_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
            return active_entries(REGISTRY_FILE, registry_root=registry_root)


        def cost_component_for_field(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
            return match_entry(field, cost_entries(registry_root=registry_root))


        def cost_evidence(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
            entry = cost_component_for_field(field, registry_root=registry_root)
            if entry is None:
                return None
            return {
                "registry": REGISTRY_FILE,
                "entry_id": entry.get("id"),
                "cost_component": entry.get("cost_component"),
                "protected_non_equivalence": list(entry.get("protected_non_equivalence", []) or []),
            }
    ''',
    "src/pegasus/registries/bridge.py": r'''
        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        from pegasus.registries.semantic import active_entries


        REGISTRY_FILE = "fields/bridge_grammars.yaml"


        def bridge_grammar_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
            return active_entries(REGISTRY_FILE, registry_root=registry_root)


        def bridge_grammar_for_type(bridge_type: str, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
            for entry in bridge_grammar_entries(registry_root=registry_root):
                if entry.get("bridge_type") == bridge_type or entry.get("id") == bridge_type:
                    return entry
            return None


        def bridge_operator_is_registered(operator: str, *, registry_root: str | Path = "config/registries") -> bool:
            for entry in bridge_grammar_entries(registry_root=registry_root):
                if operator in set(entry.get("operators", []) or []):
                    return True
            return False
    ''',
    "src/pegasus/registries/icd.py": r'''
        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        from pegasus.registries.semantic import active_entries


        REGISTRY_FILE = "health/icd_catalog.yaml"


        def icd_catalog_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
            return active_entries(REGISTRY_FILE, registry_root=registry_root)


        def chapter_for_code(code: str, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
            normalized = str(code).upper().replace(".", "")[:3]
            if not normalized:
                return None
            for entry in icd_catalog_entries(registry_root=registry_root):
                bounds = entry.get("range") or []
                if len(bounds) != 2:
                    continue
                lo, hi = str(bounds[0]).upper(), str(bounds[1]).upper()
                if lo <= normalized <= hi:
                    return entry
            return None
    ''',
}


TEST_UNIT = r'''
    from __future__ import annotations

    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from pegasus.efg.legality import evaluate_delta
    from pegasus.registries.cnes_capacity import capacity_evidence
    from pegasus.registries.diagnostic_topology import diagnostic_evidence
    from pegasus.registries.semantic import registry_is_scaffold_only
    from pegasus.registries.sih_cost import cost_evidence


    CRITICAL_REGISTRIES = (
        "health/diagnostic_topology.yaml",
        "health/cnes_capacity_registry.yaml",
        "health/sih_cost_registry.yaml",
        "health/clinical_event_definitions.yaml",
        "fields/bridge_grammars.yaml",
        "demographic/race_axis_registry.yaml",
        "health/icd_catalog.yaml",
        "health/icd_quality_groups.yaml",
        "fields/join_affordances.yaml",
        "spatial/municipality_crosswalk_sources.yaml",
        "inference/model_registry.yaml",
        "inference/residual_registry.yaml",
    )


    def _state(value: str = "active"):
        return SimpleNamespace(value=value)


    def _field(**kwargs):
        payload = {
            "field_id": "field",
            "name": "field",
            "carrier": "Deaths",
            "unit": "counts",
            "aggregation": "additive",
            "state": _state(),
            "role_json": "[]",
            "source_json": "[]",
            "metadata_json": "{}",
            "operator": "raw",
        }
        payload.update(kwargs)
        return SimpleNamespace(**payload)


    def test_slice27a_critical_registries_are_not_scaffold_only() -> None:
        root = Path("config/registries")
        for name in CRITICAL_REGISTRIES:
            path = root / name
            assert path.exists(), name
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert payload["registry_version"] == "v2.0", name
            assert not registry_is_scaffold_only(name), name
            entries = payload.get("entries") or []
            assert len(entries) >= 1, name
            assert all("scaffold_only" not in set(entry.get("warnings") or []) for entry in entries), name


    def test_slice27a_semantic_registry_helpers_classify_special_fields() -> None:
        diagnostic = _field(field_id="sim_CAUSABAS", name="CAUSABAS", unit="ICD10", aggregation="non_aggregable")
        cnes = _field(field_id="cnes_QTLEIT", name="QTLEIT bed capacity", carrier="Facilities", unit="beds")
        sih = _field(field_id="sih_VAL_SH", name="VAL_SH hospital service cost", carrier="HospitalAdmissions", unit="BRL")
        assert diagnostic_evidence(diagnostic)["topology_role"] == "underlying_cause"
        assert capacity_evidence(cnes)["capacity_family"] == "bed_capacity"
        assert cost_evidence(sih)["cost_component"] == "hospital_service"


    def test_slice27b_delta_warnings_carry_registry_evidence() -> None:
        operator = SimpleNamespace(name="RAW")
        diagnostic = _field(field_id="sim_CAUSABAS", name="CAUSABAS", unit="ICD10", aggregation="non_aggregable")
        result = evaluate_delta(parents=[diagnostic], operator=operator)
        warnings = "\n".join(str(value) for value in result.warnings)
        assert "registry_evidence_attached" in warnings
        assert "diagnostic_topology=sim_underlying_cause_causabas" in warnings
'''


TEST_INTEGRATION = r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    import yaml

    from pegasus.registries.loader import load_registries
    from pegasus.registries.validators import validate_registry_tree


    def test_slice27a27b_registry_manifest_hashes_match_written_registries() -> None:
        errors = validate_registry_tree("config/registries")
        assert errors == []
        bundle = load_registries("config/registries")
        manifest = yaml.safe_load(Path("config/registries/registry_manifest.yaml").read_text(encoding="utf-8"))
        entries = manifest.get("registries") or manifest.get("entries") or {}
        for name in ("diagnostic_topology", "cnes_capacity_registry", "sih_cost_registry", "bridge_grammars"):
            assert name in bundle.hashes
            assert name in entries
            assert entries[name]["sha256"] == bundle.hashes[name]


    def test_slice27b_legality_source_contains_registry_evidence_hook() -> None:
        source = Path("src/pegasus/efg/legality.py").read_text(encoding="utf-8")
        assert "_append_registry_evidence" in source
        assert "registry_evidence_attached" in source
        assert "diagnostic_topology" in source
        assert "cnes_capacity" in source
        assert "sih_cost" in source
'''


AUDIT = r'''
    from __future__ import annotations

    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from pegasus.core.hashing import sha256_file
    from pegasus.efg.legality import evaluate_delta
    from pegasus.registries.cnes_capacity import capacity_evidence
    from pegasus.registries.diagnostic_topology import diagnostic_evidence
    from pegasus.registries.sih_cost import cost_evidence


    CRITICAL = [
        "health/diagnostic_topology.yaml",
        "health/cnes_capacity_registry.yaml",
        "health/sih_cost_registry.yaml",
        "health/clinical_event_definitions.yaml",
        "fields/bridge_grammars.yaml",
        "demographic/race_axis_registry.yaml",
        "health/icd_catalog.yaml",
        "health/icd_quality_groups.yaml",
        "fields/join_affordances.yaml",
        "spatial/municipality_crosswalk_sources.yaml",
        "inference/model_registry.yaml",
        "inference/residual_registry.yaml",
    ]


    def _state(value: str = "active"):
        return SimpleNamespace(value=value)


    def _field(**kwargs):
        payload = {
            "field_id": "field",
            "name": "field",
            "carrier": "Deaths",
            "unit": "counts",
            "aggregation": "additive",
            "state": _state(),
            "role_json": "[]",
            "source_json": "[]",
            "metadata_json": "{}",
            "operator": "raw",
        }
        payload.update(kwargs)
        return SimpleNamespace(**payload)


    def main() -> int:
        root = Path("config/registries")
        errors: list[str] = []
        manifest = yaml.safe_load((root / "registry_manifest.yaml").read_text(encoding="utf-8"))
        entries = manifest.get("registries") or manifest.get("entries") or {}
        for filename in CRITICAL:
            path = root / filename
            if not path.exists():
                errors.append(f"missing registry: {filename}")
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if payload.get("registry_version") != "v2.0":
                errors.append(f"registry not v2.0: {filename}")
            registry_entries = payload.get("entries") or []
            if not registry_entries:
                errors.append(f"registry has no entries: {filename}")
            for entry in registry_entries:
                warnings = set(entry.get("warnings") or [])
                if entry.get("status") == "deferred" and "scaffold_only" in warnings:
                    errors.append(f"registry still scaffold-only: {filename}:{entry.get('id')}")
            stem = filename.removesuffix(".yaml")
            if stem not in entries:
                errors.append(f"registry missing from manifest: {stem}")
            elif entries[stem].get("sha256") != sha256_file(path):
                errors.append(f"registry manifest hash mismatch: {stem}")

        diagnostic = _field(field_id="sim_CAUSABAS", name="CAUSABAS", unit="ICD10", aggregation="non_aggregable")
        cnes = _field(field_id="cnes_QTLEIT", name="QTLEIT bed capacity", carrier="Facilities", unit="beds")
        sih = _field(field_id="sih_VAL_SP", name="VAL_SP professional service cost", carrier="HospitalAdmissions", unit="BRL")
        if not diagnostic_evidence(diagnostic):
            errors.append("diagnostic evidence helper failed")
        if not capacity_evidence(cnes):
            errors.append("CNES capacity evidence helper failed")
        if not cost_evidence(sih):
            errors.append("SIH cost evidence helper failed")
        delta = evaluate_delta(parents=[diagnostic], operator=SimpleNamespace(name="RAW"))
        if "registry_evidence_attached" not in set(delta.warnings):
            errors.append("DeltaResult warnings missing registry evidence marker")

        payload = {"slice": "27A/27B", "errors": errors, "critical_registries": CRITICAL}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if errors else 0


    if __name__ == "__main__":
        raise SystemExit(main())
'''


def write_registries() -> None:
    for name, content in REGISTRY_CONTENT.items():
        write(f"config/registries/{name}", content)


def write_registry_modules() -> None:
    for path, content in REGISTRY_MODULES.items():
        write(path, content)


def update_registry_manifest() -> None:
    import yaml

    root = ROOT / "config" / "registries"
    registries: dict[str, dict[str, str]] = {}
    for path in sorted(root.glob("*.yaml")):
        if path.name == "registry_manifest.yaml":
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        registries[path.stem] = {
            "path": path.name,
            "schema_version": str(payload.get("schema_version", "1.0")),
            "registry_version": str(payload.get("registry_version", payload.get("registry_id", "v1.0"))),
            "sha256": sha256_file(path),
        }
    seed = root / "sidra_table_seed.jsonl"
    if seed.exists():
        registries["sidra_table_seed"] = {
            "path": seed.name,
            "schema_version": "jsonl",
            "registry_version": "seed",
            "sha256": sha256_file(seed),
        }
    manifest = {
        "schema_version": "1.0",
        "registry_version": "v2.0",
        "created_at": "2026-06-06",
        "updated_at": "2026-06-13",
        "provenance": "Macro-Slice 27A registry manifest regenerated after registry canonicalization.",
        "entries": [
            {
                "id": "registry_manifest_v2",
                "status": "active",
                "description": "Registry manifest with per-registry content hashes.",
                "warnings": [],
            }
        ],
        "registries": registries,
    }
    (root / "registry_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True), encoding="utf-8", newline="\n")


def patch_legality() -> None:
    path = "src/pegasus/efg/legality.py"
    target = ROOT / path
    if not target.exists():
        raise SystemExit(f"Missing expected file: {path}")
    source = target.read_text(encoding="utf-8")
    if "def _append_registry_evidence" not in source:
        helper = r'''


def _registry_root_for_evidence(registries) -> str:
    root = getattr(registries, "root", None)
    if root is not None:
        return str(root)
    if isinstance(registries, (str, bytes)):
        return str(registries)
    return "config/registries"



def _append_registry_evidence(parents, operator, warnings, registries=None):
    """Attach compact registry evidence without changing DeltaResult schema."""
    merged = list(warnings or [])
    root = _registry_root_for_evidence(registries)
    evidence: list[str] = []
    try:
        from pegasus.registries.diagnostic_topology import diagnostic_evidence
        from pegasus.registries.cnes_capacity import capacity_evidence
        from pegasus.registries.sih_cost import cost_evidence
    except Exception:
        return merged
    for parent in parents or []:
        diagnostic = diagnostic_evidence(parent, registry_root=root)
        if diagnostic:
            evidence.append(f"diagnostic_topology={diagnostic.get('entry_id')}")
        capacity = capacity_evidence(parent, registry_root=root)
        if capacity:
            evidence.append(f"cnes_capacity={capacity.get('entry_id')}")
        cost = cost_evidence(parent, registry_root=root)
        if cost:
            evidence.append(f"sih_cost={cost.get('entry_id')}")
    if evidence:
        if "registry_evidence_attached" not in merged:
            merged.append("registry_evidence_attached")
        for item in evidence:
            token = f"registry_evidence:{item}"
            if token not in merged:
                merged.append(token)
    return merged
'''
        marker = "\ndef evaluate_delta("
        if marker in source:
            source = source.replace(marker, helper + marker, 1)
        else:
            source += helper
    # Replace DeltaResult warning arguments in a conservative way.
    source = source.replace("warnings=warnings,", "warnings=_append_registry_evidence(parents, operator, warnings, registries),")
    source = source.replace("warnings=warnings\n", "warnings=_append_registry_evidence(parents, operator, warnings, registries)\n")
    target.write_text(source, encoding="utf-8", newline="\n")


def write_tests_and_audit() -> None:
    write("tests/unit/test_slice27a27b_registry_canonical_delta.py", TEST_UNIT)
    write("tests/integration/test_slice27a27b_registry_canonical_delta_integration.py", TEST_INTEGRATION)
    write("scripts/dev/audits/audit_slice27a27b_registry_canonical_delta.py", AUDIT)


def copy_self() -> None:
    src = Path(__file__).resolve()
    dst = ROOT / "scripts" / "dev" / "updaters" / "apply_slice27a27b_registry_canonical_delta.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        shutil.copy2(src, dst)


def main() -> None:
    write_registries()
    write_registry_modules()
    patch_legality()
    update_registry_manifest()
    write_tests_and_audit()
    copy_self()
    print("Applied Slice 27A/27B registry canonicalization and registry-backed Delta evidence.")


if __name__ == "__main__":
    main()
