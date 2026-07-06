"""Physical tensor executor for autonomous EFG.

Operators here execute arrays. Metadata-only operators are not sufficient for
MSD convergence.

This package was split from a single ``executor.py`` module into ``support`` /
``kernels`` / ``run`` submodules. The public import surface is preserved: every
symbol the former module exposed is re-exported here.
"""

from __future__ import annotations

from pegasus.efg.executor.support import *
from pegasus.efg.executor.kernels import *
from pegasus.efg.executor.run import *
