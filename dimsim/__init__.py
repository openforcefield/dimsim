"""
dimsim
Distributed simulation package
"""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__version__ = version("dimsim")
__author__ = "Lily Wang"
