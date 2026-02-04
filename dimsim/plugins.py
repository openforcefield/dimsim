"""
A collection of convenience utilities for loading the built-in
'plugins', such as workflow protocols, calculation layers and
physical properties.
"""

import importlib
import logging
import pkgutil
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)


def register_default_plugins():
    """Registers the built-in workflow protocols, calculation layers and
    physical properties with the plugin system.
    """

    # Import the default properties.
    importlib.import_module("dimsim.properties")

    # Import the default layers
    importlib.import_module("dimsim.layers.simulation")
    importlib.import_module("dimsim.layers.reweighting")

    # Import the default workflow protocols.
    protocols_module = importlib.import_module("dimsim.protocols")

    for _, module_name, _ in pkgutil.iter_modules(protocols_module.__path__):
        importlib.import_module(f"dimsim.protocols.{module_name}")


def register_external_plugins():
    """Registers any supported plugins found in external packages with the
    plugin system.
    """

    for entry_point in entry_points().select(group="dimsim.plugins"):
        try:
            entry_point.load()
        except ImportError:
            logger.exception(f"Could not load the {entry_point.name} plugin")
