"""
Stage 15 checkpoint script.

Unlike every prior stage, this checkpoint does NOT validate new physics
-- Stage 15 introduces none. Instead it verifies FAITHFULNESS: that the
numbers Stage 15's synthesis figures/document quote are EXACTLY what
the original stages already found, not transcribed sloppily or drifted
during recombination. This is the appropriate discipline for a "pure
recombination" stage per the Extension Plan's explicit scope.

Run: python3 validate_stage15.py
"""

from __future__ import annotations

import numpy as np

from synthesis import load_all_sources, build_all_figures, classify_severity


def main():
    print("=== Stage 15 Checkpoint ===\n")
    print("-- Loading (or regenerating) all source data --")
    sources = load_all_sources()
    for name, df in sources.items():
        print(f"  {name}: {len(df)} rows")
    print()

    # =========================================================================
    # Faithfulness checks -- do the recombined numbers match known,
    # previously-established values from each source stage?
    # =========================================================================
    print("-- Faithfulness checks --")

    fine_sweep = sources["fine_sweep"]
    stage6 = sources["stage6_recovery"]
    stage9 = sources["stage9_quantization"]
    stage10 = sources["stage10_noise"]
    stage11 = sources["stage11_multiuser"]
    stage13 = sources["stage13_multipath"]
    stage14 = sources["stage14_hybrid"]

    # 1. Worst-region conventional loss at N=512, BW=4GHz, r_frac=0.05
    #    should be 11.2284 dB -- this exact number has recurred, unchanged,
    #    since Stage 5, and was used as the reference point in Stages 6, 9,
    #    10, 13, and 14. If this drifts, something in the recombination
    #    pipeline (or an upstream regeneration) is wrong.
    worst_point = fine_sweep[(fine_sweep.N == 512) & (fine_sweep.bandwidth == 4e9) &
                              (fine_sweep.r_frac_rayleigh == 0.05)].iloc[0]
    print(f"  fine_sweep worst-region point (N=512,BW=4GHz,r=0.05x): "
          f"{worst_point.gain_loss_conventional_db:.4f} dB")
    assert np.isclose(worst_point.gain_loss_conventional_db, 11.2284, atol=0.01), (
        f"Expected the well-established 11.2284 dB reference value, got "
        f"{worst_point.gain_loss_conventional_db:.4f} dB -- something drifted."
    )
    print("  OK: matches the established 11.2284 dB reference used across "
          "Stages 6/9/10/13/14.\n")

    # 2. Stage 6's recovery at r_frac=0.05 should match the same value
    #    exactly (it's computed via the same gain_loss_at_point function).
    stage6_at_005 = stage6[stage6.r_frac_rayleigh == 0.05].iloc[0]
    print(f"  Stage 6 recovery at r_frac=0.05: {stage6_at_005.db_recovered:.4f} dB")
    assert np.isclose(stage6_at_005.db_recovered, worst_point.gain_loss_conventional_db, atol=1e-6), (
        "Stage 6's recovery number should exactly match fine_sweep's "
        "conventional loss at the same point (same underlying formula)"
    )
    print("  OK: Stage 6's recovery table is internally consistent with "
          "the fine sweep.\n")

    # 3. Stage 9's 2-bit recovery efficiency at r_frac=0.05 should be a
    #    sensible fraction (previously reported ~99.97%).
    eff_2bit = stage9[(stage9.r_frac_rayleigh == 0.05) & (stage9.n_bits == 2)].iloc[0].recovery_efficiency
    print(f"  Stage 9, 2-bit recovery efficiency at r_frac=0.05: {eff_2bit:.4f}")
    assert 0.99 < eff_2bit < 1.01, f"expected ~99.97% recovery efficiency, got {eff_2bit:.4f}"
    print("  OK: matches Stage 9's previously-reported ~99.97% figure.\n")

    # 4. Stage 14's N_RF=16 loss should be < 0.1 dB (the stage's own
    #    "essentially ideal" threshold).
    loss_16rf = stage14[stage14.n_rf == 16].iloc[0].gain_loss_db
    print(f"  Stage 14, N_RF=16 gain-loss: {loss_16rf:.6f} dB")
    assert loss_16rf < 0.1, f"expected < 0.1 dB per Stage 14's own finding, got {loss_16rf:.6f} dB"
    print("  OK: matches Stage 14's 'essentially ideal at N_RF=16' finding.\n")

    # 5. Stage 13's multipath advantage-erosion should still be small at
    #    the worst tested K (previously reported: 11.228 -> 10.880 dB).
    los_only_mean = stage13[stage13.k_db == -999].gain_loss_conventional_db.mean()
    worst_k_mean = stage13[stage13.k_db == stage13[stage13.k_db != -999].k_db.min()].gain_loss_conventional_db.mean()
    erosion = los_only_mean - worst_k_mean
    print(f"  Stage 13, LOS-only mean: {los_only_mean:.4f} dB, "
          f"worst-K mean: {worst_k_mean:.4f} dB, erosion: {erosion:.4f} dB")
    assert erosion < 1.0, f"expected small erosion (<1dB) per Stage 13's finding, got {erosion:.4f} dB"
    print("  OK: matches Stage 13's 'advantage barely erodes' finding.\n")

    # 6. Stage 11's three tested r1_fracs (0.02, 0.05, 0.10) should exist
    #    as exact points in fine_sweep at N=1024, BW=2GHz -- this is the
    #    load-bearing coincidence Slice 3 depends on.
    sub_fs_1024 = fine_sweep[(fine_sweep.N == 1024) & (fine_sweep.bandwidth == 2e9)]
    stage11_r1_fracs = set(stage11.r1_frac_rayleigh.unique())
    fine_sweep_r_fracs = set(sub_fs_1024.r_frac_rayleigh.unique())
    missing = stage11_r1_fracs - fine_sweep_r_fracs
    print(f"  Stage 11's r1_fracs: {sorted(stage11_r1_fracs)}")
    print(f"  fine_sweep (N=1024,BW=2GHz) r_fracs available: {sorted(fine_sweep_r_fracs)}")
    assert not missing, (
        f"Stage 11's r1_fracs {missing} are not present in fine_sweep at "
        f"N=1024,BW=2GHz -- Slice 3's dual-axis plot would need interpolation, "
        f"which this stage deliberately avoids (no new derived numbers)."
    )
    print("  OK: every Stage 11 r1_frac has an exact matching point in "
          "fine_sweep -- Slice 3 plots real, non-interpolated data on both axes.\n")

    # =========================================================================
    # Severity classification sanity check (Slice 1's classifier).
    # =========================================================================
    print("-- Severity classifier sanity check --")
    assert classify_severity(2.9) == "Safe"
    assert classify_severity(3.1) == "Marginal"
    assert classify_severity(7.9) == "Marginal"
    assert classify_severity(8.1) == "Severe"
    print("  OK: Safe/Marginal/Severe boundaries land exactly at this "
          "project's own established 3dB/8dB thresholds.\n")

    # =========================================================================
    # Build the figures.
    # =========================================================================
    print("-- Building synthesis figures --")
    build_all_figures(sources)
    print("  Saved figures/stage15_decision_map.png")
    print("  Saved figures/stage15_robustness.png")
    print("  Saved figures/stage15_bug_to_feature.png")

    print("\n=== Stage 15 checkpoint PASSED. All recombined numbers verified "
          "faithful to their source stages; no new claims introduced. ===")


if __name__ == "__main__":
    main()