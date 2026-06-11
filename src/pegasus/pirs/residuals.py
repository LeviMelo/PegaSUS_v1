from __future__ import annotations

from pegasus.pirs.schemas import ModelOutput, ResidualField


def residual_field_from_model(output: ModelOutput) -> ResidualField:
    if output.status != "fitted" or output.residual_field_id is None:
        raise ValueError("Cannot derive residual field from non-fitted model")
    return ResidualField(
        field_id=output.residual_field_id,
        parent_model_id=output.model_id,
        residual_type="pearson",
        support={"inherits_from_model": output.model_id},
    )


def assert_model_derived_provenance(residual: ResidualField) -> None:
    if residual.provenance != ("model_derived",):
        raise ValueError("PIRS residual fields must carry exactly model_derived provenance")
