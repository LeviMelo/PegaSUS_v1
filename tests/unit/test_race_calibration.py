"""Proof-of-capability for the ecological RaceBridge calibration (W-RACE-2 wiring).

The calibration turns real admin-race counts + census self-declared population into a *calibrated*
(non-fixture) Bridge_R prior the existing local-pi bridge consumes unchanged. These tests pin: it
emits a legal prior in the right (reclassification) direction, it recovers a planted whitening (not
identity), it refuses when the cells cannot identify the confusion, and it marks itself calibrated so
a production run accepts it.
"""
from __future__ import annotations

import numpy as np
import pytest

from pegasus.measurement.race import validate_race_bridge_prior
from pegasus.measurement.race_calibration import (
    RaceCalibrationError,
    calibrate_race_bridge_prior,
    emission_of,
)

ADMIN_CODES = ["1", "2", "3", "4", "5"]
TARGETS = ["branca", "preta", "amarela", "parda", "indigena"]
BASE_COMP = np.array([0.43, 0.10, 0.45, 0.01, 0.01])
R_TRUE = np.array([1.0, 1.5, 1.2, 0.9, 1.3])
J = K = 5


def _C_whitening() -> np.ndarray:
    C = np.zeros((K, J))
    C[:, 0] = [0.95, 0.01, 0.03, 0.005, 0.005]
    C[:, 1] = [0.12, 0.68, 0.18, 0.01, 0.01]
    C[:, 2] = [0.25, 0.02, 0.70, 0.02, 0.01]   # 25% of true parda -> admin branca
    C[:, 3] = [0.05, 0.01, 0.04, 0.88, 0.02]
    C[:, 4] = [0.05, 0.02, 0.10, 0.03, 0.80]
    return C / C.sum(axis=0, keepdims=True)


def _gen_counts(S, alpha, seed, popmean=60000, base_rate=0.02):
    r = np.random.default_rng(seed)
    comp = r.dirichlet(alpha * BASE_COMP, size=S)
    pop = r.uniform(popmean * 0.5, popmean * 1.5, size=S)
    N = comp * pop[:, None]
    a_s = np.log(base_rate) + 0.3 * r.standard_normal(S)
    m = (np.exp(a_s)[:, None] * R_TRUE[None, :]) * N
    Y = r.poisson(m @ _C_whitening().T).astype(float)   # (S, K) admin counts
    return Y, N


def test_calibration_emits_a_legal_calibrated_prior():
    Y, N = _gen_counts(300, 5.0, seed=0)
    prior = calibrate_race_bridge_prior(
        admin_counts_by_cell=Y,
        census_pop_by_cell=N,
        admin_codes=ADMIN_CODES,
        target_categories=TARGETS,
        bridge_id="ecological_test_v1",
        calibration_scope="ecological_test",
    )
    # legal Bridge_R prior (row-stochastic reclassification), validated internally + here
    validate_race_bridge_prior(prior)
    for src in ADMIN_CODES:
        assert abs(sum(prior["matrix"][src].values()) - 1.0) < 1e-6
    # calibrated, NOT a fixture — a production run would accept it (pipeline refuses fixture priors)
    assert prior["metadata"]["epistemic_status"] == "ecologically_calibrated"
    assert prior["metadata"]["contextual_identifiability"] > 0.0


def test_calibration_recovers_the_planted_whitening_not_identity():
    Y, N = _gen_counts(300, 5.0, seed=0)
    prior = calibrate_race_bridge_prior(
        admin_counts_by_cell=Y, census_pop_by_cell=N,
        admin_codes=ADMIN_CODES, target_categories=TARGETS,
        bridge_id="ecological_test_v1",
    )
    # The reclassification must be materially non-identity: admin=branca carries reclassified amarela
    # (the dominant leak given the population is ~45% amarela and 25% of true amarela is admin-labeled
    # branca; parda/preta contribute little here because they are only ~1%/10% of this population).
    # matrix["1"] = P(self | admin=branca).
    assert prior["matrix"]["1"]["branca"] < 0.95          # not the identity 1.0
    assert prior["matrix"]["1"]["amarela"] > 0.05         # admin-branca includes reclassified amarela

    # Recover the emission from the stored prior with the observed admin marginal → ~ the planted C.
    p_admin_obs = Y.sum(axis=0) / Y.sum()
    C_hat = emission_of(_as_prior(prior), p_admin_obs)
    Ctrue = _C_whitening()
    assert C_hat[0, 2] > 0.12                          # amarela -> admin branca (true 0.25) materially recovered
    assert abs(C_hat[2, 1] - Ctrue[2, 1]) < 0.12       # preta -> admin amarela (true 0.18)


def test_calibration_refuses_when_unidentifiable():
    # near-constant composition across cells -> confusion not identifiable -> refuse (no silent
    # near-identity prior masquerading as calibrated).
    Y, N = _gen_counts(300, 5000.0, seed=1)
    with pytest.raises(RaceCalibrationError, match="collinear"):
        calibrate_race_bridge_prior(
            admin_counts_by_cell=Y, census_pop_by_cell=N,
            admin_codes=ADMIN_CODES, target_categories=TARGETS,
            bridge_id="ecological_test_v1",
        )


def test_calibration_validates_shapes():
    Y, N = _gen_counts(50, 5.0, seed=2)
    with pytest.raises(RaceCalibrationError):
        calibrate_race_bridge_prior(
            admin_counts_by_cell=Y[:, :4],  # wrong K
            census_pop_by_cell=N,
            admin_codes=ADMIN_CODES, target_categories=TARGETS,
            bridge_id="x",
        )


def _as_prior(payload):
    from pegasus.measurement.race import validate_race_bridge_prior
    return validate_race_bridge_prior(payload)
