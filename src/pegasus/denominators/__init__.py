"""Population denominator data plane.

Houses the two packages that build and reconstruct the population denominator
tensor: :mod:`pegasus.denominators.population` (SIDRA-anchored cube construction
and orchestration) and :mod:`pegasus.denominators.reconstruction` (the CTR
solver math). Moved here from ``pegasus.sidra.population_cube`` and
``pegasus.she.reconstruction`` respectively.
"""
