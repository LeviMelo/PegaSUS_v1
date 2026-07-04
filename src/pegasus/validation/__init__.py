"""The standing validation battery (MSD-III Part IX).

No capability is certified until it passes its relevant prong. The strategy is to test
on cases where the answer is already known: synthetic ground truth (planted edges/lags/
factors → measured recovery), known-positive/known-negative controls, and temporal
holdout. A certified result is expressible as "recovers known truths; measured
false-alarm rate X%; synthetic-recovery accuracy Y%; holds out-of-sample" — the sentence
that makes output science rather than assertion (§IX.4).
"""
