import importlib


def test_scenarios_rct_exports():
    # These are referenced in docs autosummary under causalis.scenarios.classic_rct
    mod = importlib.import_module('causalis.scenarios.classic_rct')
    for name in ['ttest', 'conversion_z_test', 'bootstrap_diff_means']:
        assert hasattr(mod, name), f"causalis.scenarios.classic_rct missing expected export: {name}"


def test_scenarios_unconfoundedness_exports():
    mod = importlib.import_module('causalis.scenarios.unconfoundedness')
    assert hasattr(mod, 'IRM')


def test_inference_subpackages_functions():
    # ATT/ATE/CATE/GATE functions referenced in docs by fully qualified names
    unconf = importlib.import_module('causalis.scenarios.unconfoundedness')
    assert hasattr(unconf, 'IRM')

    cate = importlib.import_module('causalis.scenarios.cate.cate')
    assert hasattr(cate, 'cate_esimand')

    gate = importlib.import_module('causalis.scenarios.cate.gate')
    assert hasattr(gate, 'gate_esimand')


def test_refutation_exports():
    ref = importlib.import_module('causalis.scenarios.unconfoundedness.refutation')
    for name in [
        'refute_placebo_outcome',
        'refute_placebo_treatment',
        'refute_subset',
        'refute_irm_orthogonality',
    ]:
        assert hasattr(ref, name), f"causalis.refutation missing expected export: {name}"
