"""
Tests to verify that the causalis package can be installed and imported correctly.
"""

import pytest


def test_import_causalkit():
    """Test that the causalis package can be imported."""
    try:
        import causalis
        # Check that __version__ attribute exists and is a string
        assert hasattr(causalis, '__version__')
        assert isinstance(causalis.__version__, str)
    except ImportError as e:
        pytest.fail(f"Failed to import causalis: {e}")


def test_import_submodules():
    """Test that causalis submodules can be imported."""
    try:
        from causalis import data, scenarios, statistics
        from causalis.eda import rct_design
        # Check that the imported objects are modules
        assert data.__name__ == 'causalis.data'
        assert rct_design.__name__ == 'causalis.eda.rct_design'
        assert scenarios.__name__ == 'causalis.scenarios'
        assert statistics.__name__ == 'causalis.statistics'
    except ImportError as e:
        pytest.fail(f"Failed to import causalis submodules: {e}")


def test_import_specific_functions():
    """Test that specific functions from causalis can be imported."""
    try:
        from causalis.data import generate_rct
        from causalis.scenarios.rct import ttest
        
        # Check that the imported objects are callable
        assert callable(generate_rct)
        assert callable(ttest)
    except ImportError as e:
        pytest.fail(f"Failed to import specific functions from causalis: {e}")


if __name__ == "__main__":
    # Allow running this test directly
    pytest.main(["-xvs", __file__])