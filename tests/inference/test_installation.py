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