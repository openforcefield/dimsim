"""
dimsim
Distributed simulation package
"""
from openff.units import Quantity, Unit, unit

# from dimsim.plugins import register_default_plugins, register_external_plugins
#
# register_default_plugins()
# register_external_plugins()

try:
    from importlib.metadata import version

    __version__ = version("dimsim")
except Exception:
    __version__ = "unknown"

__author__ = "Lily Wang"

__all__ = (
    "Quantity",
    "unit",
    "Unit",
)
