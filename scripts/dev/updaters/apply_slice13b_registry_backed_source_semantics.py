from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")


def patch_cli() -> None:
    path = ROOT / "src/pegasus/cli.py"
    text = path.read_text(encoding="utf-8")
    marker = "# Slice 13B registry-backed source-field commands"
    if marker in text:
        path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
        return
    block = r'''

# Slice 13B registry-backed source-field commands
@registries_app.command("source-fields-summary")
def registries_source_fields_summary(
    registry_root: Path = typer.Option(Path("config/registries"), "--registry-root"),
) -> None:
    from pegasus.registries.source_fields import source_field_registry_summary

    typer.echo(json.dumps(source_field_registry_summary(registry_root=registry_root), indent=2, sort_keys=True))


@registries_app.command("source-field-resolve")
def registries_source_field_resolve(
    source_system: str = typer.Option(..., "--source-system"),
    column: str = typer.Option(..., "--column"),
    registry_root: Path = typer.Option(Path("config/registries"), "--registry-root"),
) -> None:
    from pegasus.she.source_registry import resolve_source_field

    result = resolve_source_field(source_system=source_system, column_name=column, registry_root=registry_root)
    typer.echo(json.dumps(result.as_manifest(), indent=2, sort_keys=True))
'''
    # Append at end. Keeps CLI command wiring simple and avoids disturbing existing command functions.
    text = text.rstrip() + block + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


write("config/registries/carrier.yaml", r'''
schema_version: "1.0"
registry_id: "carrier_registry_v1"
carriers:
  Deaths:
    label: "Death events"
    description: "Individual death events and derived death counts from SIM-DO."
    allowed_units: ["counts"]
    default_aggregation: "additive"
  LiveBirths:
    label: "Live birth events"
    description: "Individual live-birth events and derived live-birth counts from SINASC."
    allowed_units: ["counts"]
    default_aggregation: "additive"
  HospitalAdmissions:
    label: "Hospital admissions"
    description: "SIH-RD hospitalization/admission records."
    allowed_units: ["counts", "days", "BRL"]
    default_aggregation: "additive"
  Facilities:
    label: "Health facilities"
    description: "CNES-ST establishment/facility-period records."
    allowed_units: ["counts", "beds", "binary_flag"]
    default_aggregation: "additive"
  Population:
    label: "Population denominator"
    description: "SIDRA or population-tensor person-year denominator carrier."
    allowed_units: ["person_years", "persons"]
    default_aggregation: "additive"
  ContextCells:
    label: "SIDRA contextual cells"
    description: "Long-form SIDRA contextual fact cells before epidemiological projection."
    allowed_units: ["raw_sidra_value", "index", "percent", "BRL", "persons", "counts"]
    default_aggregation: "non_aggregable"
  AuditMetadata:
    label: "Audit metadata"
    description: "Identifiers, hashes, raw JSON payloads, parse states, and other non-analytic metadata."
    allowed_units: ["identifier", "state", "hash", "json", "none"]
    default_aggregation: "non_aggregable"
''')

write("config/registries/unit.yaml", r'''
schema_version: "1.0"
registry_id: "unit_registry_v1"
units:
  counts:
    dimension: "count"
    additive: true
    description: "Non-negative event or record count."
  person_years:
    dimension: "exposure"
    additive: true
    description: "Person-time denominator exposure."
  persons:
    dimension: "stock_population"
    additive: true
    description: "Population stock/count from census or estimate tables."
  days:
    dimension: "duration"
    additive: true
    description: "Hospital stay or ICU days when summed over admissions."
  BRL:
    dimension: "currency"
    additive: true
    description: "Brazilian reais; may require deflation before time comparison."
  beds:
    dimension: "capacity"
    additive: true
    description: "Vector-indexed CNES bed/capacity count."
  binary_flag:
    dimension: "indicator"
    additive: false
    description: "Boolean/sentinel health-establishment flag; never silently coerced."
  raw_sidra_value:
    dimension: "sidra_raw"
    additive: false
    description: "SIDRA value before table-specific semantic projection."
  percent:
    dimension: "ratio"
    additive: false
    description: "Percentage or proportion already intensive."
  index:
    dimension: "index"
    additive: false
    description: "Index-style contextual value."
  identifier:
    dimension: "identifier"
    additive: false
    description: "Opaque identifier; not an analytic measure."
  state:
    dimension: "state"
    additive: false
    description: "Quality/parse/missingness state."
  hash:
    dimension: "hash"
    additive: false
    description: "Content or lineage hash."
  json:
    dimension: "json"
    additive: false
    description: "Serialized structured payload."
  none:
    dimension: "none"
    additive: false
    description: "Non-measure placeholder."
''')

write("config/registries/aggregation.yaml", r'''
schema_version: "1.0"
registry_id: "aggregation_registry_v1"
aggregations:
  additive:
    description: "Can be summed over compatible support after legal alignment."
    allowed_for_rates_numerator: true
    allowed_for_denominator: true
  weighted_mean:
    description: "Requires explicit weights; cannot be naively summed."
    allowed_for_rates_numerator: false
    allowed_for_denominator: false
  statistical_functional:
    description: "Derived functional such as median, quantile, or model residual."
    allowed_for_rates_numerator: false
    allowed_for_denominator: false
  compositional:
    description: "Composition vector or share; requires simplex-aware handling."
    allowed_for_rates_numerator: false
    allowed_for_denominator: false
  non_aggregable:
    description: "Identifiers, states, JSON, raw symbols, and other audit metadata."
    allowed_for_rates_numerator: false
    allowed_for_denominator: false
''')

write("config/registries/provenance.yaml", r'''
schema_version: "1.0"
registry_id: "provenance_registry_v1"
provenance_tags:
  source_normalized:
    risk: 0.05
    description: "Field produced by source-specific normalization with manifest hash."
  administrative_declaration:
    risk: 0.15
    description: "Administrative declaration process field."
  diagnostic_topology:
    risk: 0.10
    description: "CID/diagnostic topology-bearing field."
  billing_record:
    risk: 0.20
    description: "SIH billing/admission record."
  facility_registry:
    risk: 0.20
    description: "CNES establishment registry record."
  sidra_contextual:
    risk: 0.25
    description: "SIDRA contextual or denominator fact."
  component_specific_cost:
    risk: 0.20
    description: "SIH cost component preserved without generic-cost collapse."
  vector_indexed_capacity:
    risk: 0.20
    description: "CNES capacity component preserving QTINST/QTLEIT index semantics."
  audit_metadata:
    risk: 0.70
    description: "Metadata-only field; not directly analytic."
  fixture:
    risk: 1.00
    description: "Fixture-backed field."
''')

write("config/registries/quality.yaml", r'''
schema_version: "1.0"
registry_id: "quality_registry_v1"
quality_roles:
  analytical_measure:
    substrate_admissible: true
    default_state: "fragile"
    description: "Source-level measure that may become a substrate candidate if variance/support permits."
  denominator_measure:
    substrate_admissible: true
    default_state: "fragile"
    description: "Population/exposure candidate; still requires denominator legality checks."
  diagnostic_code:
    substrate_admissible: true
    default_state: "fragile"
    description: "Diagnostic topology field; not a generic string."
  audit_only:
    substrate_admissible: false
    default_state: "quarantined_descriptive"
    description: "Identifier/hash/state/raw metadata."
  sentinel_state:
    substrate_admissible: false
    default_state: "quarantined_descriptive"
    description: "Parse, missingness, or validity state column."
  forbidden_generic_collapse:
    substrate_admissible: false
    default_state: "illegal_excluded"
    description: "Field would collapse component semantics such as generic beds or generic cost."
''')

write("config/registries/source_fields.yaml", r'''
schema_version: "1.0"
registry_id: "source_field_registry_v1"
default_unknown:
  carrier: "AuditMetadata"
  unit: "none"
  aggregation: "non_aggregable"
  field_kind: "observer_proxy"
  role: ["audit_unknown_source_field"]
  quality_role: "audit_only"
  provenance: ["audit_metadata"]
  admissible: false
  dashboard_safe: "False"
  axes: {}
  warning: "unknown_source_field_requires_registry_entry"
source_systems:
  SIM-DO:
    fields:
      event_id: {carrier: AuditMetadata, unit: identifier, aggregation: non_aggregable, field_kind: observer_proxy, role: [identifier], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {event_axis: death_event}}
      year: {carrier: Deaths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [time_axis_candidate], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {time: year}}
      death_date: {carrier: AuditMetadata, unit: state, aggregation: non_aggregable, field_kind: observer_proxy, role: [event_timestamp], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {time: date}}
      age_years: {carrier: Deaths, unit: counts, aggregation: statistical_functional, field_kind: marked_functional, role: [age_mark], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {age: continuous_years}}
      age_days: {carrier: Deaths, unit: counts, aggregation: statistical_functional, field_kind: marked_functional, role: [age_mark], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {age: continuous_days}}
      sex: {carrier: Deaths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [stratifier, sex_axis], quality_role: analytical_measure, provenance: [source_normalized, administrative_declaration], admissible: true, dashboard_safe: "warning", axes: {sex: administrative}}
      race_color_admin: {carrier: Deaths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [stratifier, administrative_race_axis], quality_role: analytical_measure, provenance: [source_normalized, administrative_declaration], admissible: true, dashboard_safe: "warning", axes: {race_axis_type: administrative_death_declaration}}
      race_missingness_state: {carrier: AuditMetadata, unit: state, aggregation: non_aggregable, field_kind: observer_proxy, role: [race_missingness_audit], quality_role: sentinel_state, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {race_axis_type: administrative_death_declaration}}
      mun_residence_cod6: {carrier: Deaths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [geography_axis, residence], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {geography: mun_residence_cod6}}
      underlying_icd_norm: {carrier: Deaths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [diagnostic_topology, underlying_cause], quality_role: diagnostic_code, provenance: [source_normalized, diagnostic_topology], admissible: true, dashboard_safe: "warning", axes: {icd_topology_role: underlying_cause}}
      underlying_icd_parse_state: {carrier: AuditMetadata, unit: state, aggregation: non_aggregable, field_kind: observer_proxy, role: [icd_parse_audit], quality_role: sentinel_state, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {icd_topology_role: underlying_cause}}
      cause_chain_norm: {carrier: Deaths, unit: json, aggregation: non_aggregable, field_kind: observer_proxy, role: [diagnostic_topology, terminal_chain], quality_role: audit_only, provenance: [source_normalized, diagnostic_topology, audit_metadata], admissible: false, dashboard_safe: "False", axes: {icd_topology_role: terminal_chain}}
      associated_conditions_norm: {carrier: Deaths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [diagnostic_topology, associated_condition], quality_role: diagnostic_code, provenance: [source_normalized, diagnostic_topology], admissible: true, dashboard_safe: "warning", axes: {icd_topology_role: associated_condition}}
      source_manifest_hash: {carrier: AuditMetadata, unit: hash, aggregation: non_aggregable, field_kind: observer_proxy, role: [source_manifest_hash], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      raw_json: {carrier: AuditMetadata, unit: json, aggregation: non_aggregable, field_kind: observer_proxy, role: [raw_payload], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      row_hash: {carrier: AuditMetadata, unit: hash, aggregation: non_aggregable, field_kind: observer_proxy, role: [row_hash], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
  SINASC:
    fields:
      event_id: {carrier: AuditMetadata, unit: identifier, aggregation: non_aggregable, field_kind: observer_proxy, role: [identifier], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {event_axis: live_birth_event}}
      birth_year: {carrier: LiveBirths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [time_axis_candidate], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {time: birth_year}}
      mun_residence_cod6: {carrier: LiveBirths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [geography_axis, residence], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {geography: mun_residence_cod6}}
      mother_age_years: {carrier: LiveBirths, unit: counts, aggregation: statistical_functional, field_kind: marked_functional, role: [maternal_age_mark], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {age: maternal_years}}
      adolescent_mother_flag: {carrier: LiveBirths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [maternal_child_indicator], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {indicator: adolescent_mother}}
      advanced_maternal_age_flag: {carrier: LiveBirths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [maternal_child_indicator], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {indicator: advanced_maternal_age}}
      low_birth_weight_flag: {carrier: LiveBirths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [birth_outcome_indicator], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {indicator: low_birth_weight}}
      prematurity_flag: {carrier: LiveBirths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [birth_outcome_indicator], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {indicator: prematurity}}
      cesarean_flag: {carrier: LiveBirths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [delivery_mode_indicator], quality_role: analytical_measure, provenance: [source_normalized], admissible: true, dashboard_safe: "warning", axes: {indicator: cesarean}}
      anomaly_icd_code: {carrier: LiveBirths, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [diagnostic_topology, congenital_anomaly], quality_role: diagnostic_code, provenance: [source_normalized, diagnostic_topology], admissible: true, dashboard_safe: "warning", axes: {icd_topology_role: congenital_anomaly}}
      anomaly_icd_state: {carrier: AuditMetadata, unit: state, aggregation: non_aggregable, field_kind: observer_proxy, role: [anomaly_icd_audit], quality_role: sentinel_state, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      source_manifest_hash: {carrier: AuditMetadata, unit: hash, aggregation: non_aggregable, field_kind: observer_proxy, role: [source_manifest_hash], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
  SIH-RD:
    fields:
      admission_id: {carrier: AuditMetadata, unit: identifier, aggregation: non_aggregable, field_kind: observer_proxy, role: [identifier], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {event_axis: hospital_admission}}
      admission_year: {carrier: HospitalAdmissions, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [time_axis_candidate], quality_role: analytical_measure, provenance: [source_normalized, billing_record], admissible: true, dashboard_safe: "warning", axes: {time: admission_year}}
      mun_residence_cod6: {carrier: HospitalAdmissions, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [geography_axis, residence], quality_role: analytical_measure, provenance: [source_normalized, billing_record], admissible: true, dashboard_safe: "warning", axes: {geography: mun_residence_cod6}}
      principal_icd_norm: {carrier: HospitalAdmissions, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [diagnostic_topology, sih_principal_diagnosis], quality_role: diagnostic_code, provenance: [source_normalized, billing_record, diagnostic_topology], admissible: true, dashboard_safe: "warning", axes: {icd_topology_role: sih_principal_diagnosis}}
      secondary_icd_norm_json: {carrier: HospitalAdmissions, unit: json, aggregation: non_aggregable, field_kind: observer_proxy, role: [diagnostic_topology, sih_secondary_diagnosis], quality_role: audit_only, provenance: [source_normalized, billing_record, diagnostic_topology, audit_metadata], admissible: false, dashboard_safe: "False", axes: {icd_topology_role: sih_secondary_diagnosis}}
      stay_length_days: {carrier: HospitalAdmissions, unit: days, aggregation: additive, field_kind: extensive_measure, role: [length_of_stay], quality_role: analytical_measure, provenance: [source_normalized, billing_record], admissible: true, dashboard_safe: "warning", axes: {measure: stay_length}}
      death_flag: {carrier: HospitalAdmissions, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [inpatient_death], quality_role: analytical_measure, provenance: [source_normalized, billing_record], admissible: true, dashboard_safe: "warning", axes: {indicator: inpatient_death}}
      hospital_service_cost_real: {carrier: HospitalAdmissions, unit: BRL, aggregation: additive, field_kind: extensive_measure, role: [sih_cost_component, VAL_SH], quality_role: analytical_measure, provenance: [source_normalized, billing_record, component_specific_cost], admissible: true, dashboard_safe: "warning", axes: {cost_component: VAL_SH}}
      professional_service_cost_real: {carrier: HospitalAdmissions, unit: BRL, aggregation: additive, field_kind: extensive_measure, role: [sih_cost_component, VAL_SP], quality_role: analytical_measure, provenance: [source_normalized, billing_record, component_specific_cost], admissible: true, dashboard_safe: "warning", axes: {cost_component: VAL_SP}}
      icu_cost_real: {carrier: HospitalAdmissions, unit: BRL, aggregation: additive, field_kind: extensive_measure, role: [sih_cost_component, VAL_UTI], quality_role: analytical_measure, provenance: [source_normalized, billing_record, component_specific_cost], admissible: true, dashboard_safe: "warning", axes: {cost_component: VAL_UTI}}
      total_admission_cost_real: {carrier: HospitalAdmissions, unit: BRL, aggregation: additive, field_kind: extensive_measure, role: [sih_cost_component, VAL_TOT], quality_role: analytical_measure, provenance: [source_normalized, billing_record, component_specific_cost], admissible: true, dashboard_safe: "warning", axes: {cost_component: VAL_TOT}}
      cost_state_json: {carrier: AuditMetadata, unit: json, aggregation: non_aggregable, field_kind: observer_proxy, role: [sih_cost_state_audit], quality_role: sentinel_state, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      source_manifest_hash: {carrier: AuditMetadata, unit: hash, aggregation: non_aggregable, field_kind: observer_proxy, role: [source_manifest_hash], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
  CNES-ST:
    field_patterns:
      "^QTINST[0-9A-Z_]*$": {carrier: Facilities, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [cnes_capacity_component, QTINST], quality_role: analytical_measure, provenance: [source_normalized, facility_registry, vector_indexed_capacity], admissible: true, dashboard_safe: "warning", axes: {capacity_family: QTINST}}
      "^QTLEIT[0-9A-Z_]*$": {carrier: Facilities, unit: beds, aggregation: additive, field_kind: extensive_measure, role: [cnes_capacity_component, QTLEIT], quality_role: analytical_measure, provenance: [source_normalized, facility_registry, vector_indexed_capacity], admissible: true, dashboard_safe: "warning", axes: {capacity_family: QTLEIT}}
      ".*_state$": {carrier: AuditMetadata, unit: state, aggregation: non_aggregable, field_kind: observer_proxy, role: [state_audit], quality_role: sentinel_state, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
    fields:
      facility_id: {carrier: AuditMetadata, unit: identifier, aggregation: non_aggregable, field_kind: observer_proxy, role: [identifier], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {event_axis: cnes_facility_period}}
      year: {carrier: Facilities, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [time_axis_candidate], quality_role: analytical_measure, provenance: [source_normalized, facility_registry], admissible: true, dashboard_safe: "warning", axes: {time: year}}
      mun_facility_cod6: {carrier: Facilities, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [geography_axis, facility], quality_role: analytical_measure, provenance: [source_normalized, facility_registry], admissible: true, dashboard_safe: "warning", axes: {geography: mun_facility_cod6}}
      capacity_total_observed: {carrier: Facilities, unit: counts, aggregation: additive, field_kind: extensive_measure, role: [cnes_capacity_total_observer], quality_role: analytical_measure, provenance: [source_normalized, facility_registry, vector_indexed_capacity], admissible: true, dashboard_safe: "warning", axes: {capacity_family: observed_total}}
      invalid_flag_count: {carrier: Facilities, unit: counts, aggregation: additive, field_kind: observer_proxy, role: [cnes_flag_outlier_observer], quality_role: analytical_measure, provenance: [source_normalized, facility_registry], admissible: true, dashboard_safe: "warning", axes: {indicator: invalid_flag_count}}
      capacity_vector_json: {carrier: AuditMetadata, unit: json, aggregation: non_aggregable, field_kind: observer_proxy, role: [capacity_vector_audit], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      flag_vector_json: {carrier: AuditMetadata, unit: json, aggregation: non_aggregable, field_kind: observer_proxy, role: [flag_vector_audit], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      source_manifest_hash: {carrier: AuditMetadata, unit: hash, aggregation: non_aggregable, field_kind: observer_proxy, role: [source_manifest_hash], quality_role: audit_only, provenance: [source_normalized, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
  SIDRA:
    fields:
      table_id: {carrier: AuditMetadata, unit: identifier, aggregation: non_aggregable, field_kind: observer_proxy, role: [sidra_table_id], quality_role: audit_only, provenance: [sidra_contextual, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      variable_id: {carrier: AuditMetadata, unit: identifier, aggregation: non_aggregable, field_kind: observer_proxy, role: [sidra_variable_id], quality_role: audit_only, provenance: [sidra_contextual, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      period: {carrier: ContextCells, unit: raw_sidra_value, aggregation: non_aggregable, field_kind: observer_proxy, role: [sidra_period_axis], quality_role: audit_only, provenance: [sidra_contextual], admissible: false, dashboard_safe: "False", axes: {time: sidra_period}}
      locality_id: {carrier: ContextCells, unit: raw_sidra_value, aggregation: non_aggregable, field_kind: observer_proxy, role: [sidra_locality_axis], quality_role: audit_only, provenance: [sidra_contextual], admissible: false, dashboard_safe: "False", axes: {geography: sidra_locality}}
      classification_tuple: {carrier: ContextCells, unit: json, aggregation: non_aggregable, field_kind: observer_proxy, role: [sidra_classification_axis], quality_role: audit_only, provenance: [sidra_contextual, audit_metadata], admissible: false, dashboard_safe: "False", axes: {classification: tuple}}
      value_numeric: {carrier: ContextCells, unit: raw_sidra_value, aggregation: non_aggregable, field_kind: latent_context, role: [sidra_context_value], quality_role: analytical_measure, provenance: [sidra_contextual], admissible: true, dashboard_safe: "warning", axes: {measure: sidra_value}}
      value_status: {carrier: AuditMetadata, unit: state, aggregation: non_aggregable, field_kind: observer_proxy, role: [sidra_value_status], quality_role: sentinel_state, provenance: [sidra_contextual, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      request_hash: {carrier: AuditMetadata, unit: hash, aggregation: non_aggregable, field_kind: observer_proxy, role: [request_hash], quality_role: audit_only, provenance: [sidra_contextual, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
      metadata_hash: {carrier: AuditMetadata, unit: hash, aggregation: non_aggregable, field_kind: observer_proxy, role: [metadata_hash], quality_role: audit_only, provenance: [sidra_contextual, audit_metadata], admissible: false, dashboard_safe: "False", axes: {}}
''')

write("src/pegasus/registries/carrier.py", r'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class CarrierRegistryError(ValueError):
    """Raised when a carrier request cannot be resolved by the carrier registry."""


@dataclass(frozen=True)
class CarrierSpec:
    carrier_id: str
    label: str
    description: str
    allowed_units: tuple[str, ...]
    default_aggregation: str
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "carrier_id": self.carrier_id,
            "label": self.label,
            "description": self.description,
            "allowed_units": list(self.allowed_units),
            "default_aggregation": self.default_aggregation,
            "registry_hash": self.registry_hash,
        }


def load_carrier_registry(registry_root: str | Path = "config/registries") -> dict[str, CarrierSpec]:
    path = Path(registry_root) / "carrier.yaml"
    payload = load_yaml(path)
    raw = payload.get("carriers", {})
    if not isinstance(raw, dict) or not raw:
        raise CarrierRegistryError(f"carrier registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, CarrierSpec] = {}
    for carrier_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise CarrierRegistryError(f"carrier spec must be a mapping: {carrier_id}")
        out[str(carrier_id)] = CarrierSpec(
            carrier_id=str(carrier_id),
            label=str(spec.get("label", carrier_id)),
            description=str(spec.get("description", "")),
            allowed_units=tuple(str(x) for x in spec.get("allowed_units", [])),
            default_aggregation=str(spec.get("default_aggregation", "non_aggregable")),
            registry_hash=registry_hash,
        )
    return out


def get_carrier(carrier_id: str, *, registry_root: str | Path = "config/registries") -> CarrierSpec:
    registry = load_carrier_registry(registry_root)
    try:
        return registry[carrier_id]
    except KeyError as exc:
        raise CarrierRegistryError(f"unknown carrier: {carrier_id}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_carrier_registry(registry_root)
    return {"carriers": {k: v.as_manifest() for k, v in sorted(registry.items())}}
''')

write("src/pegasus/registries/unit.py", r'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class UnitRegistryError(ValueError):
    """Raised when a unit request cannot be resolved by the unit registry."""


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    dimension: str
    additive: bool
    description: str
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "dimension": self.dimension,
            "additive": self.additive,
            "description": self.description,
            "registry_hash": self.registry_hash,
        }


def load_unit_registry(registry_root: str | Path = "config/registries") -> dict[str, UnitSpec]:
    path = Path(registry_root) / "unit.yaml"
    payload = load_yaml(path)
    raw = payload.get("units", {})
    if not isinstance(raw, dict) or not raw:
        raise UnitRegistryError(f"unit registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, UnitSpec] = {}
    for unit_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise UnitRegistryError(f"unit spec must be a mapping: {unit_id}")
        out[str(unit_id)] = UnitSpec(
            unit_id=str(unit_id),
            dimension=str(spec.get("dimension", "unknown")),
            additive=bool(spec.get("additive", False)),
            description=str(spec.get("description", "")),
            registry_hash=registry_hash,
        )
    return out


def get_unit(unit_id: str, *, registry_root: str | Path = "config/registries") -> UnitSpec:
    registry = load_unit_registry(registry_root)
    try:
        return registry[unit_id]
    except KeyError as exc:
        raise UnitRegistryError(f"unknown unit: {unit_id}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_unit_registry(registry_root)
    return {"units": {k: v.as_manifest() for k, v in sorted(registry.items())}}
''')

write("src/pegasus/registries/aggregation.py", r'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class AggregationRegistryError(ValueError):
    """Raised when an aggregation law is missing or invalid."""


@dataclass(frozen=True)
class AggregationSpec:
    aggregation_id: str
    description: str
    allowed_for_rates_numerator: bool
    allowed_for_denominator: bool
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "aggregation_id": self.aggregation_id,
            "description": self.description,
            "allowed_for_rates_numerator": self.allowed_for_rates_numerator,
            "allowed_for_denominator": self.allowed_for_denominator,
            "registry_hash": self.registry_hash,
        }


def load_aggregation_registry(registry_root: str | Path = "config/registries") -> dict[str, AggregationSpec]:
    path = Path(registry_root) / "aggregation.yaml"
    payload = load_yaml(path)
    raw = payload.get("aggregations", {})
    if not isinstance(raw, dict) or not raw:
        raise AggregationRegistryError(f"aggregation registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, AggregationSpec] = {}
    for aggregation_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise AggregationRegistryError(f"aggregation spec must be a mapping: {aggregation_id}")
        out[str(aggregation_id)] = AggregationSpec(
            aggregation_id=str(aggregation_id),
            description=str(spec.get("description", "")),
            allowed_for_rates_numerator=bool(spec.get("allowed_for_rates_numerator", False)),
            allowed_for_denominator=bool(spec.get("allowed_for_denominator", False)),
            registry_hash=registry_hash,
        )
    return out


def get_aggregation(aggregation_id: str, *, registry_root: str | Path = "config/registries") -> AggregationSpec:
    registry = load_aggregation_registry(registry_root)
    try:
        return registry[aggregation_id]
    except KeyError as exc:
        raise AggregationRegistryError(f"unknown aggregation law: {aggregation_id}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_aggregation_registry(registry_root)
    return {"aggregations": {k: v.as_manifest() for k, v in sorted(registry.items())}}
''')

write("src/pegasus/registries/provenance.py", r'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class ProvenanceRegistryError(ValueError):
    """Raised when provenance tags cannot be resolved."""


@dataclass(frozen=True)
class ProvenanceSpec:
    tag: str
    risk: float
    description: str
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {"tag": self.tag, "risk": self.risk, "description": self.description, "registry_hash": self.registry_hash}


def load_provenance_registry(registry_root: str | Path = "config/registries") -> dict[str, ProvenanceSpec]:
    path = Path(registry_root) / "provenance.yaml"
    payload = load_yaml(path)
    raw = payload.get("provenance_tags", {})
    if not isinstance(raw, dict) or not raw:
        raise ProvenanceRegistryError(f"provenance registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, ProvenanceSpec] = {}
    for tag, spec in raw.items():
        if not isinstance(spec, dict):
            raise ProvenanceRegistryError(f"provenance spec must be a mapping: {tag}")
        out[str(tag)] = ProvenanceSpec(
            tag=str(tag),
            risk=float(spec.get("risk", 1.0)),
            description=str(spec.get("description", "")),
            registry_hash=registry_hash,
        )
    return out


def get_provenance(tag: str, *, registry_root: str | Path = "config/registries") -> ProvenanceSpec:
    registry = load_provenance_registry(registry_root)
    try:
        return registry[tag]
    except KeyError as exc:
        raise ProvenanceRegistryError(f"unknown provenance tag: {tag}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_provenance_registry(registry_root)
    return {"provenance_tags": {k: v.as_manifest() for k, v in sorted(registry.items())}}
''')

write("src/pegasus/registries/quality.py", r'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class QualityRegistryError(ValueError):
    """Raised when a quality role cannot be resolved."""


@dataclass(frozen=True)
class QualityRoleSpec:
    role_id: str
    substrate_admissible: bool
    default_state: str
    description: str
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "substrate_admissible": self.substrate_admissible,
            "default_state": self.default_state,
            "description": self.description,
            "registry_hash": self.registry_hash,
        }


def load_quality_registry(registry_root: str | Path = "config/registries") -> dict[str, QualityRoleSpec]:
    path = Path(registry_root) / "quality.yaml"
    payload = load_yaml(path)
    raw = payload.get("quality_roles", {})
    if not isinstance(raw, dict) or not raw:
        raise QualityRegistryError(f"quality registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, QualityRoleSpec] = {}
    for role_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise QualityRegistryError(f"quality role spec must be a mapping: {role_id}")
        out[str(role_id)] = QualityRoleSpec(
            role_id=str(role_id),
            substrate_admissible=bool(spec.get("substrate_admissible", False)),
            default_state=str(spec.get("default_state", "quarantined_descriptive")),
            description=str(spec.get("description", "")),
            registry_hash=registry_hash,
        )
    return out


def get_quality_role(role_id: str, *, registry_root: str | Path = "config/registries") -> QualityRoleSpec:
    registry = load_quality_registry(registry_root)
    try:
        return registry[role_id]
    except KeyError as exc:
        raise QualityRegistryError(f"unknown quality role: {role_id}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_quality_registry(registry_root)
    return {"quality_roles": {k: v.as_manifest() for k, v in sorted(registry.items())}}
''')

write("src/pegasus/registries/source_fields.py", r'''
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash
from pegasus.registries.aggregation import get_aggregation
from pegasus.registries.carrier import get_carrier
from pegasus.registries.provenance import get_provenance
from pegasus.registries.quality import get_quality_role
from pegasus.registries.unit import get_unit


class SourceFieldRegistryError(ValueError):
    """Raised when source-field registry resolution fails."""


@dataclass(frozen=True)
class SourceFieldRegistryEntry:
    source_system: str
    column_name: str
    carrier: str
    unit: str
    aggregation: str
    field_kind: str
    role: tuple[str, ...]
    quality_role: str
    provenance: tuple[str, ...]
    admissible: bool
    dashboard_safe: str
    axes: dict[str, Any]
    warning: str | None
    registry_hash: str
    matched_pattern: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "column_name": self.column_name,
            "carrier": self.carrier,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "field_kind": self.field_kind,
            "role": list(self.role),
            "quality_role": self.quality_role,
            "provenance": list(self.provenance),
            "admissible": self.admissible,
            "dashboard_safe": self.dashboard_safe,
            "axes": self.axes,
            "warning": self.warning,
            "registry_hash": self.registry_hash,
            "matched_pattern": self.matched_pattern,
        }


@dataclass(frozen=True)
class SourceFieldRegistry:
    registry_id: str
    schema_version: str
    registry_hash: str
    entries: dict[tuple[str, str], SourceFieldRegistryEntry]
    patterns: tuple[tuple[str, re.Pattern[str], dict[str, Any]], ...]
    default_unknown: dict[str, Any]

    def resolve(self, *, source_system: str, column_name: str) -> SourceFieldRegistryEntry:
        system = normalize_source_system(source_system)
        key = (system, column_name)
        if key in self.entries:
            return self.entries[key]
        for pattern_system, pattern, spec in self.patterns:
            if pattern_system == system and pattern.fullmatch(column_name):
                return _entry_from_spec(
                    source_system=system,
                    column_name=column_name,
                    spec=spec,
                    registry_hash=self.registry_hash,
                    matched_pattern=pattern.pattern,
                )
        return _entry_from_spec(
            source_system=system,
            column_name=column_name,
            spec=self.default_unknown,
            registry_hash=self.registry_hash,
            matched_pattern=None,
        )


def normalize_source_system(value: str) -> str:
    token = str(value).strip().upper().replace("_", "-")
    aliases = {
        "SIM": "SIM-DO",
        "SIM-DO": "SIM-DO",
        "SIM-DO": "SIM-DO",
        "SINASC": "SINASC",
        "SIH": "SIH-RD",
        "SIH-RD": "SIH-RD",
        "CNES": "CNES-ST",
        "CNES-ST": "CNES-ST",
        "SIDRA": "SIDRA",
    }
    return aliases.get(token, token)


def _list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(x) for x in value)
    return (str(value),)


def _entry_from_spec(*, source_system: str, column_name: str, spec: dict[str, Any], registry_hash: str, matched_pattern: str | None) -> SourceFieldRegistryEntry:
    return SourceFieldRegistryEntry(
        source_system=source_system,
        column_name=column_name,
        carrier=str(spec.get("carrier", "AuditMetadata")),
        unit=str(spec.get("unit", "none")),
        aggregation=str(spec.get("aggregation", "non_aggregable")),
        field_kind=str(spec.get("field_kind", "observer_proxy")),
        role=_list(spec.get("role", [])),
        quality_role=str(spec.get("quality_role", "audit_only")),
        provenance=_list(spec.get("provenance", [])),
        admissible=bool(spec.get("admissible", False)),
        dashboard_safe=str(spec.get("dashboard_safe", "False")),
        axes=dict(spec.get("axes", {}) or {}),
        warning=None if spec.get("warning") is None else str(spec.get("warning")),
        registry_hash=registry_hash,
        matched_pattern=matched_pattern,
    )


def _validate_entry(entry: SourceFieldRegistryEntry, *, registry_root: str | Path) -> None:
    carrier = get_carrier(entry.carrier, registry_root=registry_root)
    unit = get_unit(entry.unit, registry_root=registry_root)
    get_aggregation(entry.aggregation, registry_root=registry_root)
    get_quality_role(entry.quality_role, registry_root=registry_root)
    for tag in entry.provenance:
        get_provenance(tag, registry_root=registry_root)
    if entry.unit not in carrier.allowed_units:
        raise SourceFieldRegistryError(
            f"source field {entry.source_system}.{entry.column_name} uses unit {entry.unit!r} not allowed for carrier {entry.carrier!r}"
        )
    if entry.dashboard_safe not in {"True", "False", "warning"}:
        raise SourceFieldRegistryError(
            f"source field {entry.source_system}.{entry.column_name} has invalid dashboard_safe={entry.dashboard_safe!r}"
        )


def load_source_field_registry(registry_root: str | Path = "config/registries") -> SourceFieldRegistry:
    path = Path(registry_root) / "source_fields.yaml"
    payload = load_yaml(path)
    registry_hash = content_hash(payload)
    source_systems = payload.get("source_systems", {})
    if not isinstance(source_systems, dict) or not source_systems:
        raise SourceFieldRegistryError(f"source field registry is empty or invalid: {path}")
    default_unknown = dict(payload.get("default_unknown", {}) or {})
    entries: dict[tuple[str, str], SourceFieldRegistryEntry] = {}
    patterns: list[tuple[str, re.Pattern[str], dict[str, Any]]] = []
    for source_system, raw_system in source_systems.items():
        system = normalize_source_system(str(source_system))
        if not isinstance(raw_system, dict):
            raise SourceFieldRegistryError(f"source system registry block must be a mapping: {source_system}")
        for column_name, spec in (raw_system.get("fields", {}) or {}).items():
            if not isinstance(spec, dict):
                raise SourceFieldRegistryError(f"field spec must be a mapping: {source_system}.{column_name}")
            entry = _entry_from_spec(
                source_system=system,
                column_name=str(column_name),
                spec=spec,
                registry_hash=registry_hash,
                matched_pattern=None,
            )
            _validate_entry(entry, registry_root=registry_root)
            entries[(system, str(column_name))] = entry
        for pattern, spec in (raw_system.get("field_patterns", {}) or {}).items():
            if not isinstance(spec, dict):
                raise SourceFieldRegistryError(f"field pattern spec must be a mapping: {source_system}.{pattern}")
            compiled = re.compile(str(pattern))
            probe_name = "QTINST01" if "QTINST" in str(pattern) else ("QTLEIT01" if "QTLEIT" in str(pattern) else "example_state")
            probe = _entry_from_spec(source_system=system, column_name=probe_name, spec=spec, registry_hash=registry_hash, matched_pattern=str(pattern))
            _validate_entry(probe, registry_root=registry_root)
            patterns.append((system, compiled, spec))
    unknown_probe = _entry_from_spec(source_system="UNKNOWN", column_name="unknown", spec=default_unknown, registry_hash=registry_hash, matched_pattern=None)
    _validate_entry(unknown_probe, registry_root=registry_root)
    return SourceFieldRegistry(
        registry_id=str(payload.get("registry_id", "source_field_registry")),
        schema_version=str(payload.get("schema_version", "1.0")),
        registry_hash=registry_hash,
        entries=entries,
        patterns=tuple(patterns),
        default_unknown=default_unknown,
    )


def resolve_source_field_entry(*, source_system: str, column_name: str, registry_root: str | Path = "config/registries") -> SourceFieldRegistryEntry:
    return load_source_field_registry(registry_root).resolve(source_system=source_system, column_name=column_name)


def source_field_registry_summary(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_source_field_registry(registry_root)
    by_system: dict[str, int] = {}
    admissible = 0
    audit_only = 0
    for entry in registry.entries.values():
        by_system[entry.source_system] = by_system.get(entry.source_system, 0) + 1
        if entry.admissible:
            admissible += 1
        else:
            audit_only += 1
    return {
        "registry_id": registry.registry_id,
        "schema_version": registry.schema_version,
        "registry_hash": registry.registry_hash,
        "entry_count": len(registry.entries),
        "pattern_count": len(registry.patterns),
        "admissible_entry_count": admissible,
        "audit_or_excluded_entry_count": audit_only,
        "by_source_system": dict(sorted(by_system.items())),
    }


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_source_field_registry(registry_root)
    return {
        "registry_id": registry.registry_id,
        "schema_version": registry.schema_version,
        "registry_hash": registry.registry_hash,
        "entries": [entry.as_manifest() for entry in sorted(registry.entries.values(), key=lambda e: (e.source_system, e.column_name))],
        "patterns": [{"source_system": s, "pattern": p.pattern} for s, p, _ in registry.patterns],
        "default_unknown": registry.default_unknown,
    }
''')

write("src/pegasus/she/source_registry.py", r'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.registries.source_fields import (
    SourceFieldRegistryEntry,
    normalize_source_system,
    resolve_source_field_entry,
    source_field_registry_summary,
)


@dataclass(frozen=True)
class SourceFieldSpec:
    """SHE-facing source-field semantics resolved from the registry layer."""

    source_system: str
    column_name: str
    carrier: str
    unit: str
    aggregation: str
    field_kind: str
    role: tuple[str, ...]
    quality_role: str
    provenance: tuple[str, ...]
    admissible: bool
    dashboard_safe: str
    axes: dict[str, Any]
    registry_hash: str
    warning: str | None = None
    matched_pattern: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "column_name": self.column_name,
            "carrier": self.carrier,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "field_kind": self.field_kind,
            "role": list(self.role),
            "quality_role": self.quality_role,
            "provenance": list(self.provenance),
            "admissible": self.admissible,
            "dashboard_safe": self.dashboard_safe,
            "axes": self.axes,
            "registry_hash": self.registry_hash,
            "warning": self.warning,
            "matched_pattern": self.matched_pattern,
        }


@dataclass(frozen=True)
class SourceRegistryResolution:
    source_system: str
    column_name: str
    spec: SourceFieldSpec
    known: bool
    registry_backed: bool
    warnings: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "column_name": self.column_name,
            "known": self.known,
            "registry_backed": self.registry_backed,
            "warnings": list(self.warnings),
            "spec": self.spec.as_manifest(),
        }


def _spec_from_entry(entry: SourceFieldRegistryEntry) -> SourceFieldSpec:
    return SourceFieldSpec(
        source_system=entry.source_system,
        column_name=entry.column_name,
        carrier=entry.carrier,
        unit=entry.unit,
        aggregation=entry.aggregation,
        field_kind=entry.field_kind,
        role=entry.role,
        quality_role=entry.quality_role,
        provenance=entry.provenance,
        admissible=entry.admissible,
        dashboard_safe=entry.dashboard_safe,
        axes=dict(entry.axes),
        registry_hash=entry.registry_hash,
        warning=entry.warning,
        matched_pattern=entry.matched_pattern,
    )


def resolve_source_field(
    *,
    source_system: str,
    column_name: str,
    registry_root: str | Path = "config/registries",
) -> SourceRegistryResolution:
    system = normalize_source_system(source_system)
    entry = resolve_source_field_entry(source_system=system, column_name=column_name, registry_root=registry_root)
    spec = _spec_from_entry(entry)
    known = entry.warning != "unknown_source_field_requires_registry_entry"
    warnings: list[str] = []
    if entry.warning:
        warnings.append(entry.warning)
    if not entry.admissible:
        warnings.append(f"source_field_not_substrate_admissible:{entry.quality_role}")
    return SourceRegistryResolution(
        source_system=system,
        column_name=column_name,
        spec=spec,
        known=known,
        registry_backed=True,
        warnings=tuple(warnings),
    )


def source_registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    return source_field_registry_summary(registry_root=registry_root)
''')

write("tests/unit/test_registry_source_fields_yaml.py", r'''
from __future__ import annotations

from pegasus.registries.aggregation import get_aggregation
from pegasus.registries.carrier import get_carrier
from pegasus.registries.source_fields import resolve_source_field_entry, source_field_registry_summary
from pegasus.registries.unit import get_unit


def test_slice13b_source_field_registry_loads_yaml_and_core_dimensions() -> None:
    summary = source_field_registry_summary()
    assert summary["entry_count"] >= 40
    assert summary["admissible_entry_count"] > 10
    assert summary["by_source_system"]["SIM-DO"] >= 10
    assert summary["pattern_count"] >= 2

    deaths = get_carrier("Deaths")
    assert "counts" in deaths.allowed_units
    assert deaths.default_aggregation == "additive"

    counts = get_unit("counts")
    assert counts.additive is True

    additive = get_aggregation("additive")
    assert additive.allowed_for_rates_numerator is True


def test_slice13b_registry_resolves_known_sim_and_sih_fields() -> None:
    sim = resolve_source_field_entry(source_system="SIM-DO", column_name="underlying_icd_norm")
    assert sim.carrier == "Deaths"
    assert sim.unit == "counts"
    assert sim.aggregation == "additive"
    assert sim.quality_role == "diagnostic_code"
    assert sim.admissible is True
    assert "diagnostic_topology" in sim.provenance

    sih_cost = resolve_source_field_entry(source_system="SIH-RD", column_name="hospital_service_cost_real")
    assert sih_cost.carrier == "HospitalAdmissions"
    assert sih_cost.unit == "BRL"
    assert sih_cost.axes["cost_component"] == "VAL_SH"
    assert "component_specific_cost" in sih_cost.provenance


def test_slice13b_registry_patterns_preserve_cnes_vector_indices() -> None:
    qtleit = resolve_source_field_entry(source_system="CNES-ST", column_name="QTLEIT05")
    assert qtleit.carrier == "Facilities"
    assert qtleit.unit == "beds"
    assert qtleit.axes["capacity_family"] == "QTLEIT"
    assert qtleit.matched_pattern is not None
    assert "vector_indexed_capacity" in qtleit.provenance


def test_slice13b_unknown_field_is_audit_only_not_analytic() -> None:
    unknown = resolve_source_field_entry(source_system="SIM-DO", column_name="UNDECLARED_COLUMN")
    assert unknown.carrier == "AuditMetadata"
    assert unknown.admissible is False
    assert unknown.warning == "unknown_source_field_requires_registry_entry"
''')

write("tests/unit/test_she_source_registry_yaml_backed.py", r'''
from __future__ import annotations

from pegasus.she.source_registry import resolve_source_field, source_registry_manifest


def test_slice13b_she_source_registry_is_yaml_backed_for_known_field() -> None:
    result = resolve_source_field(source_system="SIM-DO", column_name="race_color_admin")
    assert result.registry_backed is True
    assert result.known is True
    assert result.spec.carrier == "Deaths"
    assert result.spec.axes["race_axis_type"] == "administrative_death_declaration"
    assert result.spec.dashboard_safe == "warning"
    assert result.spec.registry_hash


def test_slice13b_she_source_registry_marks_unknown_as_not_admissible() -> None:
    result = resolve_source_field(source_system="SINASC", column_name="MYSTERY")
    assert result.registry_backed is True
    assert result.known is False
    assert result.spec.admissible is False
    assert "unknown_source_field_requires_registry_entry" in result.warnings
    assert "source_field_not_substrate_admissible:audit_only" in result.warnings


def test_slice13b_she_source_registry_manifest_exposes_counts() -> None:
    manifest = source_registry_manifest()
    assert manifest["entry_count"] >= 40
    assert manifest["registry_hash"]
    assert "SIM-DO" in manifest["by_source_system"]
''')

write("tests/integration/test_slice13b_registry_backed_substrate.py", r'''
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest


def test_slice13b_substrate_uses_registry_semantics_for_candidates_and_exclusions(tmp_path: Path) -> None:
    artifact = tmp_path / "sim.parquet"
    pl.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "year": [2020, 2020, 2021],
            "race_color_admin": ["1", "4", ""],
            "underlying_icd_norm": ["I10", "J18", "R99"],
            "underlying_icd_parse_state": ["valid", "valid", "ill-defined"],
            "constant_col": [1, 1, 1],
            "all_missing_col": [None, None, None],
        }
    ).write_parquet(artifact)

    bundle = build_substrate_bundle(
        artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
    )
    candidates = {candidate.column_name: candidate for candidate in bundle.candidate_fields}
    exclusions = {exclusion.column_name: exclusion for exclusion in bundle.excluded_fields}

    assert candidates["year"].carrier == "Deaths"
    assert candidates["race_color_admin"].axes["race_axis_type"] == "administrative_death_declaration"
    assert candidates["underlying_icd_norm"].quality_role == "diagnostic_code"
    assert exclusions["event_id"].reason in {"registry_not_admissible", "structural_or_audit_only"}
    assert exclusions["underlying_icd_parse_state"].reason in {"registry_not_admissible", "structural_or_audit_only"}
    assert exclusions["constant_col"].reason == "zero_variance_constant"
    assert exclusions["all_missing_col"].reason == "all_missing"

    manifest_path = write_substrate_bundle_manifest(bundle, tmp_path / "substrate.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] >= 3
    assert payload["excluded_count"] >= 4
    assert payload["registry_backed"] is True
    assert payload["source_reality_mode"] == "fixture_only"
''')

write("scripts/dev/audits/audit_slice13b_registry_backed_source_fields.py", r'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pegasus.registries.source_fields import resolve_source_field_entry, source_field_registry_summary
from pegasus.she.source_registry import resolve_source_field


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Slice 13B registry-backed source-field semantics.")
    parser.add_argument("--registry-root", default="config/registries")
    args = parser.parse_args()
    root = Path(args.registry_root)
    summary = source_field_registry_summary(registry_root=root)
    if summary["entry_count"] < 40:
        raise SystemExit(f"source-field registry too small: {summary['entry_count']}")
    required = [
        ("SIM-DO", "underlying_icd_norm", "Deaths", "counts"),
        ("SIM-DO", "race_color_admin", "Deaths", "counts"),
        ("SINASC", "low_birth_weight_flag", "LiveBirths", "counts"),
        ("SIH-RD", "hospital_service_cost_real", "HospitalAdmissions", "BRL"),
        ("CNES-ST", "QTLEIT05", "Facilities", "beds"),
        ("SIDRA", "value_numeric", "ContextCells", "raw_sidra_value"),
    ]
    for system, column, carrier, unit in required:
        entry = resolve_source_field_entry(source_system=system, column_name=column, registry_root=root)
        if entry.carrier != carrier or entry.unit != unit:
            raise SystemExit(f"unexpected registry entry for {system}.{column}: {entry.as_manifest()}")
        resolution = resolve_source_field(source_system=system, column_name=column, registry_root=root)
        if not resolution.registry_backed:
            raise SystemExit(f"SHE resolution is not registry-backed for {system}.{column}")
        if not resolution.spec.registry_hash:
            raise SystemExit(f"SHE resolution missing registry hash for {system}.{column}")
    unknown = resolve_source_field(source_system="SIM-DO", column_name="UNDECLARED_COLUMN", registry_root=root)
    if unknown.known or unknown.spec.admissible:
        raise SystemExit("unknown source field was not quarantined as audit-only")
    print(
        "AUDIT PASSED: Slice 13B registry-backed source-field semantics validated "
        + json.dumps(
            {
                "entry_count": summary["entry_count"],
                "pattern_count": summary["pattern_count"],
                "registry_hash": summary["registry_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')

patch_cli()

# Ensure EOF normalized for files touched by patch_cli.
for rel in ["src/pegasus/cli.py"]:
    p = ROOT / rel
    p.write_text(p.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8", newline="\n")

print("Slice 13B registry-backed source semantics applied.")
