"""A collection of utility functions for registering mappings between
ThermoML archive properties, and properties supported by the estimation
framework.
"""


class _ThermoMLPlugin:
    """Represents a property which may be extracted from a ThermoML archive."""

    def __init__(self, string_identifier, conversion_function, supported_phases):
        """Constructs a new ThermoMLPlugin object.

        Parameters
        ----------
        string_identifier: str
            The ThermoML string identifier (ePropName) for this property.
        conversion_function: function
            A function which maps a `ThermoMLProperty` into a
            `PhysicalProperty`.
        supported_phases: PropertyPhase:
            An enum which encodes all of the phases for which this
            property supports being estimated in.
        """

        self.string_identifier = string_identifier
        self.conversion_function = conversion_function

        self.supported_phases = supported_phases


def _default_mapping(property_class, property_to_map):
    """

    Parameters
    ----------
    property_class: type of PhysicalProperty
        The class to map this property into.
    property_to_map: ThermoMLProperty
        The ThermoML property to map.
    """

    mapped_property = property_class()

    mapped_property.value = property_to_map.value

    if property_to_map.uncertainty is not None:
        mapped_property.uncertainty = property_to_map.uncertainty

    mapped_property.phase = property_to_map.phase

    mapped_property.thermodynamic_state = property_to_map.thermodynamic_state
    mapped_property.substance = property_to_map.substance

    return mapped_property
