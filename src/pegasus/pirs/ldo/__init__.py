"""Lattice Dependency Operator (LDO) — the re-founded PIRS engine (MSD-II §II.6)."""

from pegasus.pirs.ldo.assemble import LDOField, assemble_ldo_tensor
from pegasus.pirs.ldo.margins import GaussianField, gaussianize_field, randomized_pit_gaussianize
from pegasus.pirs.ldo.precision import PrecisionFit, build_spatial_precision, fit_contemporaneous_precision
from pegasus.pirs.ldo.lowrank import SparseLowRankFit, fit_sparse_plus_lowrank
from pegasus.pirs.ldo.lags import LaggedFit, LaggedLink, fit_lagged_links
from pegasus.pirs.ldo.edges import stability_select, to_link_records
from pegasus.pirs.ldo.residual_scan import joint_model_residuals, scan_residual_nonlinear_edges
from pegasus.pirs.ldo.certify import (
    LDOCertificationError,
    LDOCertificationPolicy,
    assert_ldo_edge_promotion_allowed,
    certify_links,
)
from pegasus.pirs.ldo.orchestrator import LDORun, run_ldo
from pegasus.pirs.ldo.resolution import MultiResolutionRun, restrict_variables, run_multiresolution_ldo
from pegasus.pirs.ldo.records import LINK_RECORD_COLUMNS, EdgeType, LinkRecord

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
    "LinkRecord",
    "EdgeType",
    "LINK_RECORD_COLUMNS",
]
