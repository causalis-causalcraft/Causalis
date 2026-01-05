import importlib


def test_scenarios_rct_exports():
    # These are referenced in docs autosummary under causalis.scenarios.rct
    mod = importlib.import_module('causalis.scenarios.rct')
    for name in ['ttest', 'conversion_z_test', 'bootstrap_diff_means']:
        assert hasattr(mod, name), f"causalis.scenarios.rct missing expected export: {name}"


def test_scenarios_unconfoundedness_atte_exports():
    mod = importlib.import_module('causalis.scenarios.unconfoundedness.atte')
    assert hasattr(mod, 'dml_atte_source')
    assert hasattr(mod, 'dml_atte')


def test_inference_subpackages_functions():
    # ATT/ATE/CATE/GATE functions referenced in docs by fully qualified names
    atte = importlib.import_module('causalis.scenarios.unconfoundedness.atte')
    assert hasattr(atte, 'dml_atte')

    ate = importlib.import_module('causalis.scenarios.unconfoundedness.ate')
    assert hasattr(ate, 'dml_ate')

    cate = importlib.import_module('causalis.scenarios.unconfoundedness.cate')
    assert hasattr(cate, 'cate_esimand')

    gate = importlib.import_module('causalis.scenarios.unconfoundedness.gate')
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
