import openmm
from openff.toolkit.topology.molecule import Molecule
from openff.toolkit.topology.topology import Topology
from openff.toolkit.typing.engines.smirnoff.io import ParameterIOHandler
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openff.toolkit.utils.base_wrapper import ToolkitWrapper
from openff.toolkit.utils.toolkit_registry import ToolkitRegistry


class ForceField:
    def __init__(
        self,
        *sources,
        aromaticity_model: str = ...,
        parameter_handler_classes: list[type[ParameterHandler]] | None = None,
        parameter_io_handler_classes: list[type[ParameterIOHandler]] | None = None,
        disable_version_check: bool = False,
        allow_cosmetic_attributes: bool = False,
        load_plugins: bool = False,
    ) -> None: ...

    def create_openmm_system(
        self,
        topology: Topology,
        *,
        toolkit_registry: ToolkitRegistry | ToolkitWrapper | None = None,
        charge_from_molecules: list["Molecule"] | None = None,
        partial_bond_orders_from_molecules: list["Molecule"] | None = None,
        allow_nonintegral_charges: bool = False,
    ) -> openmm.System: ...
