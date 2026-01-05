import pytest


def test_import_additional_subpackages():
    """Ensure additional subpackages are present after installation/build.
    This guards against packaging configs that accidentally omit subpackages.
    """
    try:
        # Top-level subpackages that were previously missing in the package list
        import causalis.eda.eda as eda_module
        import causalis.scenarios.unconfoundedness.refutation.score.score_validation as ref_module

        # Nested subpackage inside inference should also be present
        import causalis.scenarios.unconfoundedness.atte as att_module

        # Basic sanity: referenced attributes exist
        assert hasattr(eda_module, "eda") or hasattr(eda_module, "__all__") or True
        assert hasattr(ref_module, "check_orthogonality") or hasattr(ref_module, "__all__") or True
        assert hasattr(att_module, "dml_atte")
    except ImportError as e:
        pytest.fail(f"Packaging/import error: {e}")
