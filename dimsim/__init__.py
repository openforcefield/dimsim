"""
dimsim
Distributed simulation package
"""

try:
    from importlib.metadata import version

    __version__ = version("dimsim")
except Exception:
    __version__ = "unknown"

__author__ = "Lily Wang"
