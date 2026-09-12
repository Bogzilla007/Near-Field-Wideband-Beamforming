# Near-Field Wideband Beamforming — Capstone Synthesis

This document is the closing narrative for the project, written after
Stage 14. It makes no new claims — every number here traces back to a
result already validated in Stages 1–14 (see `README.md` for the
full stage-by-stage record and `validate_stage15.py` for the checks
confirming these numbers are quoted faithfully).

---

## 1. The elevator pitch

Classical beamforming assumes the incoming signal looks like a flat
wave arriving from a single direction. That assumption quietly breaks
down as arrays get large and bandwidths get wide — closer users produce
a genuinely *curved* wavefront, and wide bandwidths mean a single phase
shift no longer steers every frequency correctly. This project mapped
exactly where that breakdown happens, built and validated a corrector,
priced out cheaper approximations of that corrector, checked whether
any of it survives real hardware and real channels, and then found that
the very effect that breaks conventional beamforming — near-field
curvature — is also a resource: it lets an array tell apart two users
standing in the same direction, which angle-only beamforming can never
do.

---

## 2. The core empirical law

**Finding (Stages 2, 5, 8):** the textbook Rayleigh distance
(`2D²/λ`) is a *conservative* boundary. The actual 3dB gain-loss
threshold — where a conventional beamformer starts failing —
sits at **r ≈ 0.15–0.2× Rayleigh distance**, not 1×. Independently, at
high bandwidth, a **second, distance-independent failure boundary**
appears once the array crosses roughly N≈70–100 elements, driven purely
by beam squint. Confirmed to hold, with the same normalization
(`r/Rayleigh(N)`), from N=8 up to **N=1024** (Stage 8) — including a
non-obvious detail: the squint-driven boundary itself shifts to lower
bandwidths as N grows, since squint severity scales with N at fixed
bandwidth.

---

## 3. What fixes it, and at what cost

**Full digital correction (Stage 6):** the combined near-field+squint
corrector recovers **11.2–13.4 dB** in the worst-tested region
(N=512, 4GHz, r≤0.1×Rayleigh) — consistently, not at one cherry-picked
point — at a **33× compute cost** (O(N·F) vs. O(N) multiplies per
beam-update).

**Cheaper approximations, tested at that same worst-region point:**
- **2-bit phase-shifter quantization (Stage 9)** keeps the corrector
  within 1dB of its own idealized performance. (A naive "recovery
  efficiency" ratio metric said even 1 bit was >99% as good — that
  was a metric artifact, not a real result; the stricter, honest
  criterion lands at 2 bits.)
- **16 RF chains, out of 512 elements (Stage 14)**, in a fully-connected
  hybrid analog+digital architecture, reaches within **0.1dB** of the
  full digital corrector's gain — a sharp transition, not a slow
  approach, because the near-field amplitude taper turns out to be a
  smooth function that a modest number of phase-only beams can
  reconstruct via interference.

**Important scope caveat:** both of these cheaper-hardware findings were
validated at **one specific (N, bandwidth, r) point**, not swept across
the whole failure map. Treat them as "here is a concrete, proven
existence result at this operating point," not "this generalizes
everywhere" — see the decision map in Section 8.

---

## 4. Does it survive contact with reality?

**Noise (Stage 10):** every beamformer mode in this project carries
identical total steering-vector energy, which has a provable
consequence — **the dB gap between beamformer modes is exactly
independent of the noise floor.** That's not a hand-wave; it was
confirmed numerically (the 11.2284dB gap held bit-identical across a
-35dB to 0dB input-SNR sweep). What *does* depend on absolute SNR is the
downstream nonlinear consequence: at -20dB input SNR, that fixed gap is
the difference between a working link (combined: BER 6.9×10⁻⁴) and a
broken one (conventional: BER 0.19). At -35dB, both are near the BER
ceiling — the *practical* difference compresses there, even though the
dB number never does.

**Multipath (Stage 13):** even the ideal corrector's own absolute gain
degrades substantially under scattering (a ~6.2dB drop at K=-5dB, where
scattered power exceeds line-of-sight power) — it has no model of the
reflectors. But its *advantage over conventional* barely erodes (11.23dB
→ 10.88dB, a 0.35dB shrink) across the same range, because both
beamformers absorb similar contamination from the same random-phase
scattered paths, so their *ratio* is far more stable than either's
absolute performance.

**Pattern worth naming:** twice now (Stage 10's provable noise-invariance,
Stage 13's empirical multipath-robustness), a *relative* comparison
between beamformer modes turned out to be far more robust to a shared
real-world impairment than either mode's absolute performance. That's
not a coincidence of this project's metric choices — it's a genuine
structural property worth remembering when evaluating any beamforming
comparison under non-ideal conditions.

---

## 5. From bug to feature

**Multi-user spatial multiplexing (Stage 11):** the same near-field
curvature that breaks single-user angle-only beamforming can *separate*
two users standing at the same angle but different ranges — something
far-field beamforming cannot do at all (angle-only steering is
literally blind to range). At N=1024, deep near-field (r=0.02×
Rayleigh), the achievable separation is **8.95dB**; it shrinks to
0.13dB as the two users' ranges converge (0.10×/0.12×Rayleigh) —
finite range resolution, as physically expected.

**Unexpected mechanism finding:** curvature-only (narrowband), delay-only
(wideband, uniform amplitude), and the full combined corrector all
achieve nearly identical separation (spread <0.02dB) at every range gap
tested. At N=1024, the array aperture (~5.5m) is always tiny relative to
the tested ranges, so amplitude taper and multi-frequency delay-slope
are both negligible — **the entire discrimination capability comes from
near-field phase curvature alone.** The same "amplitude taper is
negligible when aperture ≪ r" theme independently resurfaces in Stage
14's hybrid-beamforming result.

Section 8's Slice 3 makes the "bug ↔ feature" connection directly
visual: the exact same (N=1024, deep-near-field) conditions that
maximize conventional beamforming's failure are exactly where
near-field-aware multi-user separation is largest.

---

## 6. What generalizes, and with what caveats

**Array size (Stage 8):** the core failure law holds cleanly from N=8 to
N=1024, with runtime scaling and Rayleigh-distance scaling both matching
theory (2x/4x respectively for a doubling of N).

**2D geometry (Stage 12):** every channel model and beamformer mode
reduces to the validated 1D result to machine precision as a regression
gate. `r/Rayleigh(D_diag)` (diagonal-aperture-based Rayleigh distance)
collapses divergence curves cleanly across array size and off-axis
angle **for square arrays** — but **not fully across aspect ratio**:
elongated arrays (e.g. 128×32) show a consistently larger divergence
spread (~25% relative) than square arrays at the same total N and same
normalized distance. Aspect ratio is a genuine, separate factor a single
diagonal scalar doesn't fully absorb — a real but modest effect, not a
dramatic failure of the formula.

---

## 7. Honest scope note — what was deliberately never tested

- **Only phase quantization** was modeled (Stage 9); amplitude
  quantization was explicitly left as a stated follow-up in the
  original plan and never built.
- **No mutual coupling or calibration error** — every array element is
  treated as ideal and independent.
- **QPSK only** for the BER/EVM analysis (Stage 10); higher-order
  modulations would shift the absolute BER numbers, though the
  underlying SNR-invariance argument is modulation-agnostic.
- **Hybrid beamforming and quantization cost-tradeoffs were validated at
  a single (N, bandwidth, r) operating point each**, not swept across
  the full failure-map grid — see Section 3's caveat.
- **Multipath is Rician/geometric**, not ray-traced or measured — 3
  reflectors with randomized (but fixed-per-trial) geometry, not a
  physically simulated propagation environment.
- **The 2D aspect-ratio effect was found but not resolved** — Stage 12
  flagged that a single diagonal-based Rayleigh distance doesn't fully
  normalize aspect ratio, but a better candidate (e.g. per-axis Rayleigh
  distances) was never built or tested.
- **No real hardware validation anywhere** — this is a simulation study
  throughout; real phase-shifter nonlinearity, ADC/DAC quantization
  noise beyond the modeled phase resolution, and thermal effects are
  all out of scope.
- **Single-stream, single- or two-user scenarios only** — no
  multi-stream MIMO precoding, no more-than-2-user multiplexing.

---

## 8. The three headline figures

- **`figures/stage15_decision_map.png`** — the failure severity map
  (Safe/Marginal/Severe, using this project's own established 3dB/8dB
  thresholds) with the three validated remediation options overlaid at
  their exact tested coordinates.
- **`figures/stage15_robustness.png`** — the multipath and noise
  findings side by side, showing the core advantage surviving both
  independent impairments.
- **`figures/stage15_bug_to_feature.png`** — Stage 5/8's failure curve
  and Stage 11's multi-user separation curve, plotted together at
  matching (N=1024, r/Rayleigh) coordinates, rising in tandem.

All three are generated by `synthesis.py` and verified for faithfulness
to source data by `validate_stage15.py`.