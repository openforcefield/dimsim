import typing

from dimsim.datasets.datasets import PropertyPhase

EntryTag = typing.Literal["density"]

"""
['Mass density, kg/m3',
 'Excess molar volume, m3/mol',
 'Relative permittivity at zero frequency',
 'Excess molar enthalpy (molar enthalpy of mixing), kJ/mol',
 'Molar enthalpy of vaporization or sublimation, kJ/mol']
"""


class DataEntry(typing.TypedDict):
    tag: EntryTag = "density"  # should not be this

    phases: list[PropertyPhase]

    smiles: list[str]

    x: list[float]

    temperature: float

    pressure: float

    value: float

    std: float | None

    units: str

    source: str


class DensityEntry(DataEntry):
    tag: EntryTag = "density"

    phases: list[PropertyPhase] = PropertyPhase.Liquid


class ExcessMolarVolumeEntry(DataEntry):
    tag: EntryTag = "excess_molar_volume"

    phases: list[PropertyPhase] = PropertyPhase.Liquid


class DielectricConstantEntry(DataEntry):
    tag: EntryTag = "dielectric_constant"

    phases: list[PropertyPhase] = PropertyPhase.Liquid


class EnthalpyOfMixingEntry(DataEntry):
    tag: EntryTag = "enthalpy_of_mixing"

    phases: list[PropertyPhase] = PropertyPhase.Liquid


class EnthalpyOfVaporizationEntry(DataEntry):
    tag: EntryTag = "enthalpy_of_vaporization"

    phases: list[PropertyPhase] = PropertyPhase.Liquid | PropertyPhase.Gas


"""
@thermoml_property(
    "Vapor or sublimation pressure, kPa",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class VaporPressure(PhysicalProperty):
    def default_unit(cls):
        return unit.kilopascal
"""


class VaporPressureEntry(DataEntry):
    tag: EntryTag = "pvap"

    phases: list[PropertyPhase] = PropertyPhase.Liquid | PropertyPhase.Gas
