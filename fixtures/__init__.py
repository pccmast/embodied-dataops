"""Repository fixtures (SPEC 13).

Everything generated here is **synthetic** (``data_origin=synthetic``): the
material is deterministic, seeded with :data:`SEED` and provenance-recorded in
``fixtures/manifest.json``.  It is never presented as real robot capture.
"""

from __future__ import annotations

#: SPEC 13.1: every synthetic fixture uses this fixed seed.
SEED = 20261029

#: SPEC 0.2: synthetic files are always labelled.
DATA_ORIGIN = "synthetic"

__all__ = ["DATA_ORIGIN", "SEED"]
