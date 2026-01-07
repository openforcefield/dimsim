"""
Unit tests for dimsim package.
"""

import dimsim


def test_import():
    """Test that the package can be imported."""
    assert dimsim is not None


def test_version():
    """Test that version is defined."""
    assert hasattr(dimsim, "__version__")
    assert isinstance(dimsim.__version__, str)


def test_author():
    """Test that author is defined."""
    assert hasattr(dimsim, "__author__")
    assert isinstance(dimsim.__author__, str)
