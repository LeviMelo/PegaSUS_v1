"""Source artifact reality contracts for PegaSUS data-layer hardening."""

from pegasus.source_artifacts.compile_policy import (
    CompileSourceReality,
    CompileSourceRealityError,
    attach_compile_source_reality,
    resolve_compile_source_reality,
)
from pegasus.source_artifacts.contracts import (
    SourceArtifact,
    SourceArtifactError,
    inspect_source_artifact,
    load_source_artifact_manifest,
    source_manifest_summary,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)

__all__ = [
    "CompileSourceReality",
    "CompileSourceRealityError",
    "attach_compile_source_reality",
    "resolve_compile_source_reality",
    "SourceArtifact",
    "SourceArtifactError",
    "inspect_source_artifact",
    "load_source_artifact_manifest",
    "source_manifest_summary",
    "validate_source_artifact_manifest",
    "write_source_artifact_manifest",
]
