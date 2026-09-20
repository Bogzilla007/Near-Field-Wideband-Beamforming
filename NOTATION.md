# Notation, Symbols & Assumptions

**Purpose:** single source of truth for every symbol and modeling
assumption used across this project, referenced (not restated) by every
other document. Per the Minimum Viable Paper plan's P1.1, this is a
transcription of what already exists in `README.md` (Sections 2 and 6)
and `SYNTHESIS.md` (Section 7) — no new claims, no new modeling
decisions, just one place to look instead of fifteen.

---

## 1. Symbols

| Symbol | Meaning | Where enforced in code |
|---|---|---|
| N | element count (ULA) or N_x×N_y (UPA) | `arraymodel.py` / `planar.py` |
| d | element spacing (λ/2 at center frequency) | `arraymodel.py` |
| D | aperture (D_diag = √(D_x² + D_y²) for UPA) | `arraymodel.py` / `planar.py` |
| fc | carrier / center frequency (28 GHz default) | `constants.py` (`CENTER_FREQ`) |
| B | bandwidth | `sweep.py` |
| B/fc | fractional bandwidth | derived; not yet a persisted column anywhere (see Section 3) |
| R_R | classical Rayleigh distance, 2D²/λ | `arraymodel.py` (`rayleigh_distance()`) / `planar.py` |
| r/R_R | normalized range (r expressed as a fraction of the array's own Rayleigh distance) | `sweep.py` (`r_frac_rayleigh`) |
| θ | observation / steering angle (azimuth) | `constants.py` (`THETA`), fixed at 20° everywhere — see Section 3, gap B1 |
| φ | elevation angle (2D/UPA only) | `planar.py` |
| N_RF | number of RF chains (hybrid architecture) | `hybrid.py` |
| b | phase-shifter bit depth | `hardware.py` |
| K | Rician K-factor (LOS power / scattered power) | `multipath.py` |

**Note on fc and θ:** as of the P1.1 refactor, `CENTER_FREQ` and
`THETA` are defined exactly once, in `constants.py`, and imported
everywhere they're used (`sweep.py`, `correction.py`, `sensing.py`,
`multiuser.py`, `synthesis.py`, and every `validate_stageN.py`). This
closes gap A5 (previously redeclared as a local constant in 15
separate files — see `README.md` Section 7). It does **not** close gap
B1: θ is still a single fixed value (20°) throughout the project, never
swept. Centralizing the constant makes it trivial to change fc or θ in
one place if a future stage needs to; it does not, by itself, add an
angle sweep. See "What this plan deliberately does NOT include" in the
Minimum Viable Paper plan for why a full B1 angle sweep is out of scope.

---

## 2. Explicit modeling assumptions

Lifted from `correction.py`'s cost-estimate docstring and
`SYNTHESIS.md` Section 7 ("Honest scope note"), made front-and-center
here rather than left implicit or buried in a closing section:

- **Isotropic elements.** No element radiation pattern is modeled;
  every array element is treated as an ideal isotropic radiator.
- **Ideal, continuous-phase phase shifters**, except where explicitly
  relaxed by Stage 9's quantized model (`hardware.py`,
  `quantize_phase`) — phase resolution is finite there (1–8 bits
  tested), amplitude is left untouched (continuous) even in the
  quantized case.
- **No mutual coupling.** Elements are modeled as independent; no
  element-to-element electromagnetic coupling is included.
- **No calibration error.** Every element's phase/amplitude response
  is assumed perfectly known and perfectly realized (up to Stage 9's
  quantization), with no residual hardware miscalibration modeled.
- **QPSK only.** The BER/EVM analysis (Stage 10, `noise.py`) is
  derived for QPSK specifically. Higher-order modulations would shift
  the absolute BER/EVM numbers; the underlying SNR-invariance argument
  (Section 4 below) is modulation-agnostic, but no other modulation was
  built or tested.
- **Rician (not ray-traced) multipath.** Stage 13's multipath model
  (`multipath.py`) combines a line-of-sight path with a small number
  (3, in the validated runs) of randomly-placed but per-trial-fixed
  reflectors, mixed via a Rician K-factor. This is a controlled,
  geometric approximation of multipath, not a physically simulated or
  measured propagation environment.
- **Single carrier per experiment**, unless explicitly stated
  otherwise. No inter-carrier or multi-band scenarios are modeled.

---

## 3. Known gaps (documented, not fixed, per plan scope)

These are explicitly *not* being closed as part of the Minimum Viable
Paper plan — flagging them here is the entire scope of that work, not
a to-do list:

- **B1 — θ fixed at 20° everywhere.** No angle sweep exists in the
  codebase. Per the plan's closing section, this stays out of scope
  "unless P5a or P5b reveals angle-dependence is specifically where the
  novelty gap is" — every fixed parameter is a stated assumption, not
  automatically a gap that must be closed before publishing.
- **A2 — B/fc as a derived column.** Fractional bandwidth is not
  currently persisted as its own column in any results CSV
  (`fine_sweep.csv`, `coarse_sweep.csv`, etc. store `bandwidth` and `N`
  separately). It can be computed post hoc (`bandwidth / CENTER_FREQ`)
  from existing data without re-running any sweep; whether it's worth
  adding as a materialized column is a P2/write-up decision, not a
  simulation gap.

---

## 4. Metric conventions (for reference, not new content)

Restated briefly here since every quantity above is meaningless without
it; full detail lives in `beamformers.py`'s module docstring and
`README.md` Section 6:

- **Gain metric:** `Gain = |a^H · h|^2`, using the raw (unnormalized,
  unit-per-element-amplitude) steering vector against the channel —
  the classical "power pattern peak" convention (validated against N²
  scaling in Stage 3), not an SNR-normalized array gain.
- **Steering-vector energy convention:** every beamformer mode is
  normalized to `sum|a_n|² = N` (or per-frequency-bin, for
  frequency-dependent modes), so gain comparisons across modes are
  fair. This is what makes Stage 10's SNR-independence result
  (Section 4 of `SYNTHESIS.md`) a provable consequence rather than an
  empirical coincidence.
- **Channel power convention:** every channel used in a gain-loss
  metric is power-normalized to `sum|h|² = N·F` before comparison, so
  the near-field `1/r` amplitude taper never contaminates a metric
  meant to isolate phase/steering-shape mismatch.
- **Range normalization:** distance is always swept as a fraction of
  *that array's own* Rayleigh distance (`r_frac_rayleigh`), not an
  absolute meter value — the normalization that made Stage 2's
  divergence curves collapse across array sizes, and (with the aspect
  ratio caveat in Stage 12) across 2D geometries too.

---

*This document supersedes no other file's content — it collects and
cross-references, per the plan's P1.1 scope. Source of truth for any
discrepancy remains the code itself and `README.md`/`SYNTHESIS.md`.*
