"""Lattice Dependency Operator (LDO) — the re-founded PIRS engine (MSD-II §II.6)."""

from pegasus.ldo.assemble import LDOField, assemble_ldo_tensor
from pegasus.ldo.margins import GaussianField, gaussianize_field, randomized_pit_gaussianize
from pegasus.ldo.precision import PrecisionFit, build_spatial_precision, fit_contemporaneous_precision
from pegasus.ldo.lowrank import SparseLowRankFit, fit_sparse_plus_lowrank
from pegasus.ldo.lags import LaggedFit, LaggedLink, fit_lagged_links
from pegasus.ldo.edges import stability_select, to_link_records
from pegasus.ldo.residual_scan import joint_model_residuals, scan_residual_nonlinear_edges
from pegasus.ldo.certify import (
    LDOCertificationError,
    LDOCertificationPolicy,
    assert_ldo_edge_promotion_allowed,
    certify_links,
)
from pegasus.ldo.orchestrator import LDORun, run_ldo
from pegasus.ldo.resolution import MultiResolutionRun, restrict_variables, run_multiresolution_ldo
from pegasus.ldo.envelope import ScaleExceedsEnvelopeError, assert_within_envelope, estimate_ldo_bytes, load_compute_envelope
from pegasus.ldo.output import link_records_to_table, write_hypotheses
from pegasus.ldo.records import LINK_RECORD_COLUMNS, EdgeType, LinkRecord

__all__ = [
    "LDOField",
    "assemble_ldo_tensor",
    "GaussianField",
    "gaussianize_field",
    "randomized_pit_gaussianize",
    "PrecisionFit",
    "fit_contemporaneous_precision",
    "build_spatial_precision",
    "SparseLowRankFit",
    "fit_sparse_plus_lowrank",
    "LaggedFit",
    "LaggedLink",
    "fit_lagged_links",
    "stability_select",
    "to_link_records",
    "joint_model_residuals",
    "scan_residual_nonlinear_edges",
    "certify_links",
    "assert_ldo_edge_promotion_allowed",
    "LDOCertificationPolicy",
    "LDOCertificationError",
    "run_ldo",
    "LDORun",
    "run_multiresolution_ldo",
    "MultiResolutionRun",
    "restrict_variables",
    "assert_within_envelope",
    "estimate_ldo_bytes",
    "load_compute_envelope",
    "ScaleExceedsEnvelopeError",
    "link_records_to_table",
    "write_hypotheses",
    "LinkRecord",
    "EdgeType",
    "LINK_RECORD_COLUMNS",
]
