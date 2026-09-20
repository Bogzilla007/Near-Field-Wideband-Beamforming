"""
Shared project-wide constants.

Introduced as part of the Minimum Viable Paper plan's P1.1 (notation
table) work, fixing gap A5: CENTER_FREQ (and THETA) were previously
redeclared as a local constant in 15 separate files instead of being
defined in one place (see README.md Section 7, "Gotchas for Future
Stages"). Every file that used its own local `CENTER_FREQ = 28e9` /
`THETA = np.deg2rad(20.0)` now imports these from here instead.

This is a pure refactor: no numeric value changes. Every value below is
copied verbatim from the pre-refactor per-file constants, and every
validate_stageN.py script was re-run after this change to confirm zero
regression (see NOTATION.md).
"""

from __future__ import annotations

import numpy as np

# Carrier / center frequency, Hz. 28 GHz -- upper-midband-ish 6G
# reference point (comment carried over from validate_stage1.py, the
# original source of this constant).
CENTER_FREQ: float = 28e9

# Standard test/steering angle, radians. 20 degrees -- an arbitrary
# non-broadside angle used throughout the project to test generality
# (comment carried over from validate_stage2.py). Fixed for the whole
# project (see README.md's "What this plan deliberately does NOT
# include" -- a full angle sweep is explicitly out of scope, not an
# oversight).
THETA: float = np.deg2rad(20.0)
