"""
Defines an API for defining thermodynamic states.
"""

from enum import Enum

from openff.toolkit import Quantity
from openff.units import Unit

from dimsim.attributes import UNDEFINED, Attribute, AttributeClass


class Ensemble(Enum):
    """An enum describing the supported thermodynamic ensembles."""

    NVT = "NVT"
    NPT = "NPT"


class ThermodynamicState(AttributeClass):
    """Data specifying a physical thermodynamic state obeying
    Boltzmann statistics.

    Notes
    -----
    Equality of two thermodynamic states is determined by comparing
    the temperature in kelvin to within 3 decimal places, and comparing
    the pressure (if defined) in pascals to within 3 decimal places.

    Examples
    --------
    Specify an NPT state at 298 K and 1 atm pressure.

    >>> state = ThermodynamicState(temperature=298.0*unit.kelvin, pressure=1.0*unit.atmospheres)

    Note that the pressure is only relevant for periodic systems.
    """

    temperature = Attribute(
        docstring="The external temperature.", type_hint=Quantity
    )
    pressure = Attribute(
        docstring="The external pressure.", type_hint=Quantity, optional=True
    )

    @property
    def inverse_beta(self):
        """
        Returns the temperature multiplied by the molar gas constant
        """
        return (self.temperature * Unit.molar_gas_constant).to("kilojoule/mole")

    @property
    def beta(self):
        """
        Returns one divided by the temperature multiplied by the molar gas constant
        """
        return 1.0 / self.inverse_beta

    def __init__(self, temperature=None, pressure=None):
        """
        Constructs a new ThermodynamicState object.
        """
        if temperature is not None:
            self.temperature = temperature
        if pressure is not None:
            self.pressure = pressure

    def validate(self, attribute_type=None):
        super(ThermodynamicState, self).validate(attribute_type)

        if self.pressure != UNDEFINED:
            self.pressure.to("pascals")
            assert self.pressure > Quantity(0.0, "pascals")

        self.temperature.to("kelvin")
        assert self.temperature > Quantity(0.0, "kelvin")

    def __repr__(self):
        return f"<ThermodynamicState {str(self)}>"

    def __str__(self):
        return_value = f"T={self.temperature:~}"

        if self.pressure != UNDEFINED:
            return_value += f" P={self.pressure:~}"

        return return_value

    def __hash__(self):
        temperature = self.temperature.m_as("kelvin").magnitude
        pressure = (
            None
            if self.pressure == UNDEFINED
            else self.pressure.m_as("pascals")
        )

        return hash(
            (f"{temperature:.3f}", None if pressure is None else f"{pressure:.3f}")
        )

    def __eq__(self, other):
        if not isinstance(other, ThermodynamicState):
            return False

        return hash(self) == hash(other)

    def __ne__(self, other):
        return not (self == other)
