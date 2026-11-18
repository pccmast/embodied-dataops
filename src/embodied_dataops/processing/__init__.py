"""Video/image processing primitives (SPEC 8.1, 5.2).

Modules
-------
``detection``
    Detector abstraction: real ``Yolo11nDetector`` (Ultralytics YOLO11n, CPU)
    and the deterministic, clearly labelled ``FakeDetector`` that exists only
    for contract/scale tests.
``preview``
    OpenCV frame extraction, 640x480 checks and JPEG thumbnails for canonical
    10 Hz videos.
``files``
    Scratch workspaces and bounded archive extraction (Lead-owned).
"""

from __future__ import annotations
