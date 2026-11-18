"""DO-A02 conditional operations, receipts/fences and diagnostics (V2 D2-M4).

Modules
-------
* :mod:`contract`   - contract version and capability constants (DO-A02 C1)
* :mod:`etag`       - strong ETag parsing/formatting (C2)
* :mod:`revisions`  - ``business_revision`` CAS helpers (C2)
* :mod:`service`    - the single transaction that turns a conditional POST into
  an exactly-once business effect + operation receipt (C2/C3)
* :mod:`effects`    - the three concrete effects (reprocess / retry / analyze gaps)
* :mod:`receipts`   - receipt read models with the REAL downstream state (C4)
* :mod:`gap_analysis` - bounded processor that completes a QUEUED GapAnalysis
* :mod:`diagnostics` - bounded read-only timeline / failure analysis / aggregate (C5)
"""

from __future__ import annotations

__all__ = [
    "contract",
    "diagnostics",
    "effects",
    "etag",
    "gap_analysis",
    "receipts",
    "revisions",
    "service",
]
