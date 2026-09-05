# Near-Field Wideband Beamforming Project — README

**Purpose of this file:** a single source of truth for what's been built,
what each result means, and what's left — so nothing gets lost across
sessions. Update the status table and "Last updated" line whenever a
stage completes.

**Last updated:** after Stage 9
**Status:** Stages 1–9, 11 complete. Stage 10, 5b, 12, 13, 14, 15 planned.

---

## 1. What This Project Is

Empirically maps where classical far-field/narrowband beamforming breaks
down as a function of array size (N), signal bandwidth (B), and user
distance (r) — cross-validated against the analytical Rayleigh distance
(`2D²/λ`) — and quantifies how much of that loss a corrected beamformer
can recover. Later stages (11+) reframe near-field curvature as a
**resource** (extra range degree-of-freedom for multi-user separation),
not just a failure mode.

Two failure mechanisms are modeled and kept isolated throughout:
- **Near-field breakdown** — spherical wavefront (not planar); phase
  *and* amplitude vary per element.
- **Beam squint** — wideband signal, single phase-shift only steers the
  carrier correctly.

**Discipline used throughout:** build → hard-gate checkpoint script with
numeric assertions → generate a plot → only then proceed. Every stage's
`validate_stageN.py` is a standalone script with `assert`s tied to
specific numeric claims — not just "looks about right" plots.

---

## 2. File Map

| File | Stage introduced | What it does |
|---|---|---|
| `arraymodel.py` | 1 | `ArrayGeometry` (ULA, λ/2 spacing, `rayleigh_distance()`), `generate_ofdm_signal`, `generate_setup` |
| `channel.py` | 2 | `far_field_channel` (phase-only), `near_field_channel` (phase+amplitude, exact per-element distance) |
| `beamformers.py` | 3/4 | 4 steering-vector modes (`conventional`, `nearfield`, `squint`, `combined`), `beamform_gain`, `beamform()` dispatcher |
| `visualization.py` | 4 | `compute_beam_pattern`/`plot_beam_pattern` — polar beam-pattern plots, reused by Stage 4's checkpoint and (planned) Stage 5b dashboard |
| `sweep.py` | 5 | `gain_loss_at_point`, `run_sweep` — the (N, bandwidth, r/Rayleigh) grid engine behind the failure map |
| `correction.py` | 6 | `recovery_table`, `cost_estimate` — combined-corrector recovery + O(N) vs O(N·F) cost comparison |
| `sensing.py` | 7 | `estimate_range` — matched-filter range estimator (scans candidate r, picks best `combined`-mode match) |
| `multiuser.py` | 11 | Two-user near-field scenario: `two_user_channels`, `separation_db`, `delay_only_steering_vector`, `superposition_sanity_check`, `run_scenario` |
| `hardware.py` | 9 | `quantize_phase`, `quantized_beamform` — finite-bit-depth phase-shifter constraint on top of any beamform() mode |
| `validate_stage1.py` … `validate_stage11.py` | — | Checkpoint scripts, one per stage, each self-contained with `assert`s |

**Shared conventions across all files** (important — see Section 6 before
extending anything):
- `CENTER_FREQ = 28e9` (28 GHz) is redeclared as a local constant in every
  validate script rather than imported from one place.
- `THETA = np.deg2rad(20.0)` is the standard test angle, used consistently
  in `sweep.py`, `sensing.py`, `multiuser.py`, and most validate scripts.
- All steering vectors are energy-normalized to `sum|a_n|² = N` (or per
  frequency bin, for `combined`) so gain comparisons across modes are fair.
- All channels used in gain-loss metrics are power-normalized to
  `sum|h|² = N·F` before comparison, so the near-field `1/r` amplitude
  taper never contaminates a metric that's supposed to isolate phase/
  steering-shape mismatch.
- Distance is always swept as a **fraction of that array's own Rayleigh
  distance** (`r_frac_rayleigh`), not an absolute meter value — this is
  what makes results comparable across different N.

---

## 3. Stage-by-Stage Results

### Stage 1 — Array & Signal Setup ✅
- `ArrayGeometry`, `generate_ofdm_signal` (QPSK-on-subcarriers, 120kHz
  5G-NR-like spacing).
- N=8→512 profiled: near-linear runtime scaling (~7.85–8.56x for 8x
  array growth). **Original N ceiling: 512.**

### Stage 2 — Channel Models ✅
- `far_field_channel` (planar, phase-only) vs `near_field_channel`
  (spherical, phase+amplitude).
- **Bug caught:** convergence metric needed global-phase removal (bulk
  propagation delay) — fixed with a phase-invariant chordal distance metric.
- Divergence curves for N=32/128/512 **collapse onto one curve** when
  distance is normalized by each array's own Rayleigh distance — strong
  confirmation that Rayleigh distance is the right normalization variable.

### Stage 3 — Baseline Conventional Beamformer ✅
- `conventional_steering_vector`, `beamform_gain` (pinned to the **N²
  power-pattern convention**, not N-scaled SNR gain).
- **Bug caught:** unnormalized `1/r` taper at r=50×Rayleigh (which itself
  scales as N²) was collapsing gain for large N — fixed via channel power
  normalization.
- Gain matches N² exactly (ratio = 1.0000) at N=8/16/32/64.

### Stage 4 — Advanced Beamformers ✅
- Added `nearfield` (curvature-only), `squint` (TTD/delay-only),
  `combined` (both) modes, plus `beamform()` dispatcher.
- **Bug caught:** near-field/combined steering vectors had different
  total energy than conventional/squint's unit-modulus convention,
  briefly making "broken" conventional score *higher* gain than the
  ideal combined corrector — fixed via per-frequency energy normalization.
- All isolation tests + 4 convergence limits passed.

### Stage 5 — Sweep Engine & Failure Map ✅
- Coarse (27-point) and fine (420-point, later extended in Stage 8)
  sweeps of the (N, bandwidth, r/Rayleigh) grid.
- **Key finding:** the empirical 3dB failure contour sits at
  **r ≈ 0.15–0.2× Rayleigh distance**, not 1× — the textbook Rayleigh
  distance is a *conservative* bound. At high bandwidth (4GHz), a
  separate **vertical** squint-driven failure boundary appears around
  N≈70–100, independent of distance.
- Results: `results/coarse_sweep.csv`, `results/fine_sweep.csv`.

### Stage 6 — Correction & Recovery ✅
- Applied the combined corrector to the worst region (N=512, 4GHz,
  r/Rayleigh = 0.02/0.05/0.1). **Result: 11.2–13.4 dB recovered**
  (mean 11.93 dB) across all three points.
- Cost: conventional O(N)=512 multiplies/update vs combined
  O(N·F)=16,896 (33×), 33 frequency bins.

### Stage 7 — Sensing Illustration ✅ (stretch tier)
- Matched-filter range estimator (scans candidate r, picks best
  `combined`-mode match). Error stayed 0.6–1.2% at all 3 tested
  distances (0.1×, 1×, 10× Rayleigh), including far-field.
- **Important caveat, carried forward into Stage 11:** this works even
  at 10×Rayleigh (far-field) because it exploits **bulk propagation
  delay's dependence on r** (a phase-vs-frequency slope), which is a
  *different* mechanism from near-field curvature — not an unexplained
  anomaly, but worth flagging explicitly.

### Stage 8 — N=1024 Re-validation ✅
- Extended Stage 1's profiling and Stage 5's fine sweep to N=1024.
- Timing: 512→1024 ratio ~2x (ideal linear), well within tolerance.
- Rayleigh distance: 1397.89m → 5602.53m, ratio 4.008x, matching the
  analytical N² scaling.
- **Coherence check caught a real (non-bug) finding:** a naive 3dB-
  crossing detector flagged a "discontinuity" at BW=400MHz between
  N=512 and N=1024. Root cause: the squint **floor** (asymptotic loss at
  large r, independent of distance) rose from 1.76dB (N=512, below
  threshold) to 4.52dB (N=1024, above threshold) — i.e. Stage 5's
  "vertical squint boundary," previously only visible at 4GHz/N~70-100,
  swept down to include 400MHz by N=1024. Real physics, not a bug — the
  diagnostic was fixed to separate the squint-floor mechanism from the
  genuine near-field crossing before re-asserting coherence.
- **N ceiling raised to 1024.**

### Stage 9 — Quantized Phase Shifters ✅
- Applied realistic finite-bit-depth phase quantization to **both** the
  broken conventional baseline **and** the combined corrector (not just
  the baseline), at Stage 6's worst region (N=512, 4GHz, r/Rayleigh =
  0.02/0.05/0.1). Amplitude left untouched, only phase quantized to
  1/2/3/4/6/8 bits.
- **Metric — "recovery efficiency":** the dB gap between quantized-
  conventional and quantized-combined, divided by Stage 6's original
  fully-idealized recovery. Both modes' loss vs. the unquantized ideal
  shrinks monotonically as bits increase (quantization behaves
  correctly); 1-bit resolution shows a clear, non-trivial 3.9dB penalty
  on the otherwise-perfect combined corrector.
- **Important nuance caught mid-stage:** the naive recovery-efficiency
  metric hits >99% even at **1 bit** for every point — which looks like
  "1-bit hardware is fine." It isn't: this is a ratio artifact — at
  1-bit, both conventional and combined degrade by similar absolute
  amounts, so their *gap* survives even though the combined corrector's
  own performance is already 3.9dB worse than its idealized self. A
  stricter, more honest criterion (corrector's own absolute loss vs.
  ideal < 1dB) was added and consistently lands at **2 bits** across all
  three worst-region points — the actual practical recommendation, not
  1 bit.
- Results: `results/stage9_quantization.csv`, `figures/stage9_quantization.png`.

### Stage 10 — SNR / Noise Floor ⬜ (not started)

### Stage 11 — Multi-User Near-Field Spatial Multiplexing ✅
**Reframe:** near-field curvature as a *resource* — can two users at the
same angle θ but different ranges r1, r2 be separated using range
information, where far-field/angle-only beamforming structurally cannot?

- Array: N=1024 (Stage 8's ceiling), θ=20°, bandwidth=2GHz for wideband
  conditions.
- **Design correction made mid-stage:** the first attempt at a "delay-
  only, no curvature" control used the existing `squint` mode — but that
  mode has **no `r` parameter at all**, so it structurally cannot encode
  range information, and produced numbers nearly identical to the
  angle-only baseline. This wasn't a finding, it was a flawed isolation.
  Replaced with a purpose-built `delay_only_steering_vector` (exact
  per-frequency r-dependent phase, but uniform amplitude — no curvature
  taper) to correctly isolate the delay/phase-slope mechanism.
- **Superposition sanity check** (flagged as a risk in the extension
  plan, since this is the first stage to sum two near-field channels):
  confirmed `near_field_channel`/`beamform_gain` compose correctly under
  superposition (finite values, power and Cauchy-Schwarz bounds
  respected) before trusting any downstream leakage numbers.
- **Headline result:** angle-only beamforming (`conventional`) cannot
  discriminate same-angle users by range — separation stays near 0 dB or
  even goes *negative* (a flat-wavefront assumption can actively favor
  the wrong user, since it happens to better match whichever user is
  less curved/farther away). Any near-field-phase-aware beamformer
  clears this baseline by a wide margin, with resolution that degrades
  as the range gap narrows:
  | r1/r2 (× Rayleigh) | Separation (dB), combined mode |
  |---|---|
  | 0.02 / 0.04 (wide gap) | 8.95 |
  | 0.05 / 0.08 (medium gap) | 2.73 |
  | 0.10 / 0.12 (narrow gap) | 0.13 |
- **Unexpected but genuine finding:** the three "r-aware" conditions
  (A = narrowband curvature-only, B = wideband combined, C = wideband
  delay-only) converge to nearly identical separation values at every
  range gap tested (spread <0.02 dB). At N=1024, the array aperture
  (~5.5m) is always ≪ any tested range (≥112m), so amplitude taper and
  multi-frequency delay-slope both turn out to be **negligible**
  contributors — the entire discrimination capability comes from
  near-field **phase curvature** alone (the Fresnel-quadratic deviation
  from far-field's linear phase), present even at a single frequency.
  This is a more precise, mechanism-level finding than the original
  hypothesis (which expected curvature and delay to be separately-sized
  contributors), and matches near-field/XL-MIMO literature — amplitude
  taper only matters within a few aperture-lengths of the array, far
  closer than any distance tested anywhere in this project.
- Results: `results/stage11_multiuser.csv`, `figures/stage11_multiuser.png`.

---

## 4. Bugs Caught & Fixed (running log)

| Stage | Bug | Root cause | Fix |
|---|---|---|---|
| 2 | Convergence metric never reached 0 | Didn't account for global phase (bulk delay) | Phase-invariant chordal distance metric |
| 3 | Gain shrank toward zero with N instead of growing | Unnormalized `1/r` taper at r=50×Rayleigh (scales as N²) swamped the signal | Channel power normalization |
| 4 | "Broken" conventional beamformer scored *higher* gain than the ideal combined corrector | Steering vectors had inconsistent total energy across modes | Per-frequency energy normalization on all modes |
| 4/5 | Isolation-test assertions initially failed | Test conditions too mild to produce a meaningfully broken baseline | Tuned test conditions to a regime where the effect is actually large |
| 8 | 3dB-crossing "discontinuity" flagged between N=512/1024 | Naive detector conflated the near-field crossing with a separate squint-floor breach | Diagnostic split into two mechanisms (squint floor vs. near-field-driven excess) before re-checking coherence |
| 9 | "Recovery efficiency" metric said 1-bit phase resolution was >99% as good as ideal | Ratio metric hides absolute degradation when both sides degrade similarly — gap survives even though both are individually much worse | Added a stricter absolute-loss criterion (corrector's own loss < 1dB), which correctly identifies 2 bits as the real minimum, not 1 |
| 11 | Condition C ("delay-only") showed ~0 separation, matching the angle-only baseline | `squint` mode has no `r` parameter — cannot encode range info by construction; this was a flawed test, not a real finding | Built `delay_only_steering_vector`: exact r-dependent phase, uniform amplitude |

None of these were caught by "it looks about right" — every one was
caught by a specific, falsifiable numeric assertion failing.

---

## 5. Remaining / Planned Work

Per the current Extension Plan, order is: **8 → 11 → 9 → 10 → 5b → 12 →
13 → 14 → 15.** Stages 8, 9, and 11 are done; next up is Stage 10.

- **Stage 10 — SNR/noise floor:** AWGN at the receiver, reframe gain-loss
  as effective SNR, add BER/EVM, sweep SNR as a 4th axis.
- **Stage 5b — Interactive dashboard:** Streamlit app, placed after
  8–11/9/10 so it has a richer result set to showcase. Reuses
  `visualization.py`'s beam-pattern functions.
- **Stage 12 — 2D UPA:** generalize `ArrayGeometry` to 2D; open question
  is the correct normalization variable in 2D (possibly
  `(r/R_Rayleigh, θ, φ, N_x/N_y)`).
- **Stage 13 — Multipath:** controlled via a single Rician-K-factor
  sweep (not random reflections).
- **Stage 14 — Hybrid analog+digital beamforming:** last, deliberately
  open-ended/loosely scoped.
- **Stage 15 — Capstone synthesis:** depends on ALL of 8–14; 2-3
  headline slices classifying regions into qualitative bands; no new
  unvalidated claims, only recombination of prior results.

---

## 6. Gotchas for Future Stages (read before extending)

- **`mode="squint"` has no `r` parameter.** It's purely angle-based TTD
  correction. Don't use it as a stand-in for "delay-based ranging" in
  any future multi-user/sensing work — it structurally cannot encode
  range. Use `delay_only_steering_vector` (multiuser.py) instead, or
  build an equivalent for the specific mechanism you actually need.
- **Amplitude taper (`1/r_n`) is negligible whenever aperture ≪ r** —
  true for essentially every scenario tested in this project (Rayleigh
  distance scales as N², so even "deep near-field" fractions of it are
  still physically far past the array's few-meter aperture). Don't
  expect amplitude-based effects to show up unless deliberately testing
  r comparable to the aperture size itself (reactive near-field —
  outside this project's realistic 6G-user framing).
- **A "recovery efficiency" or similar ratio metric can hide absolute
  degradation.** If two quantities degrade by similar absolute amounts
  under some constraint (e.g. phase quantization), their *ratio* or
  *gap* can look nearly unchanged even though both are individually far
  worse than ideal (see Stage 9). Always check at least one absolute
  metric alongside any ratio-based "efficiency" claim before calling a
  low resource level (bits, SNR, etc.) "sufficient."
- **3dB-crossing detection needs mechanism separation.** If a future
  stage's failure map shows an unexpected "discontinuity," check whether
  it's actually a squint-floor effect (constant across r) crossing
  threshold, not a genuine near-field-boundary shift, before assuming a
  bug (see Stage 8's bug log entry).
- **`CENTER_FREQ`/`THETA` are redeclared per-file, not imported from one
  place.** If either ever needs to change project-wide, it must be
  updated in every validate script and `sweep.py`/`sensing.py`/
  `multiuser.py` individually.

---

## 7. How to Reproduce

```bash
pip install numpy pandas matplotlib --break-system-packages

# Run stages in order (each depends on artifacts from earlier ones):
python3 validate_stage1.py
python3 validate_stage2.py
python3 validate_stage3.py
python3 validate_stage4.py
python3 validate_stage5.py    # also runs Stage 8's N=1024 extension
python3 validate_stage6.py
python3 validate_stage7.py
python3 validate_stage9.py
python3 validate_stage11.py
```

Each script prints a full checkpoint log and saves its outputs to
`results/*.csv` and `figures/*.png`. A non-zero exit / raised
`AssertionError` means that stage's hard gate failed — do not proceed to
a dependent stage until it's fixed.
