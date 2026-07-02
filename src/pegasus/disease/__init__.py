"""Disease Semantic Axis (MSD-II §II.13).

The disease dimension is a *variable-identity* axis (it determines what a variable
measures), not a support or stratification axis. This package supplies the deterministic
ICD/CID hierarchy (``icd_adapter``), the multi-label provenance-typed Disease Concept
Registry (``concept_registry``), and the DiseaseGraph structural priors (``graph``) that
let disease structure enter the LDO as a prior on the variable dimension, exactly parallel
to how the SpatialWeightGraph enters as a prior on the cell dimension.
"""
