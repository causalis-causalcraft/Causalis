def test_refutation_namespace_exports():
    import causalis.scenarios.unconfoundedness.refutation as ref

    # Overlap exports
    assert hasattr(ref, "positivity_overlap_checks")
    assert hasattr(ref, "calibration_report_m")

    # Score-based refutations
    assert hasattr(ref, "refute_placebo_outcome")
    assert hasattr(ref, "refute_subset")


    # SUTVA helper
    assert hasattr(ref, "print_sutva_questions")
