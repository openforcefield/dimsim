"""Tests for pydantic utilities"""

import json

import openff.units
import pydantic
import pytest

from dimsim.utils.pydantic import OpenFFQuantity


class TestOpenFFQuantity:
    """Tests for OpenFFQuantity type annotation"""

    def test_validate_from_string(self):
        """Test that OpenFFQuantity can validate from a string"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        m = Model(temperature="300 kelvin")
        assert isinstance(m.temperature, openff.units.Quantity)
        assert m.temperature.magnitude == 300
        assert m.temperature.units == unit.kelvin

    def test_validate_from_quantity(self):
        """Test that OpenFFQuantity accepts existing Quantity objects"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        q = openff.units.Quantity(300, unit.kelvin)
        m = Model(temperature=q)
        assert m.temperature == q

    def test_validate_with_unit_conversion(self):
        """Test that OpenFFQuantity converts to expected units"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        # Provide celsius, should convert to kelvin
        m = Model(temperature="27 celsius")
        assert isinstance(m.temperature, openff.units.Quantity)
        assert m.temperature.units == unit.kelvin
        assert abs(m.temperature.magnitude - 300.15) < 0.01

    def test_validate_incompatible_units_raises(self):
        """Test that incompatible units raise ValidationError"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        # Length units are incompatible with temperature
        with pytest.raises(pydantic.ValidationError) as exc_info:
            Model(temperature="1.5 nanometer")

        assert "expected units of 'kelvin'" in str(exc_info.value).lower()

    def test_validate_incompatible_quantity_raises(self):
        """Test that a Quantity object with incompatible units raises ValidationError"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        # Create a Quantity with length units (incompatible with temperature)
        length_quantity = openff.units.Quantity(1.5, unit.nanometer)

        with pytest.raises(pydantic.ValidationError) as exc_info:
            Model(temperature=length_quantity)

        assert "expected units of 'kelvin'" in str(exc_info.value).lower()

    def test_validate_invalid_type_raises(self):
        """Test that invalid types raise ValidationError"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        with pytest.raises(pydantic.ValidationError):
            Model(temperature=123)  # Plain number without units

    def test_serialize_to_string(self):
        """Test that Quantity serializes to string"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        m = Model(temperature="300 kelvin")
        data = m.model_dump()

        # Serialization should produce a string
        assert isinstance(data["temperature"], str)
        assert "300" in data["temperature"]
        assert "kelvin" in data["temperature"]

    def test_json_serialization(self):
        """Test JSON serialization produces valid JSON with string values"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]
            pressure: OpenFFQuantity[unit.pascal]

        m = Model(temperature="300 kelvin", pressure="101325 pascal")
        json_str = m.model_dump_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data["temperature"], str)
        assert isinstance(data["pressure"], str)
        assert "300" in data["temperature"]
        assert "kelvin" in data["temperature"]

    def test_json_round_trip(self):
        """Test that models can be serialized and deserialized through JSON"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]
            distance: OpenFFQuantity[unit.nanometer]

        original = Model(temperature="300 kelvin", distance="1.5 nanometer")
        json_str = original.model_dump_json()

        # Deserialize from JSON
        restored = Model.model_validate_json(json_str)

        assert restored.temperature.magnitude == original.temperature.magnitude
        assert restored.temperature.units == original.temperature.units
        assert restored.distance.magnitude == original.distance.magnitude
        assert restored.distance.units == original.distance.units

    def test_multiple_quantity_fields(self):
        """Test model with multiple Quantity fields of different dimensions"""

        class SimulationParams(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]
            pressure: OpenFFQuantity[unit.bar]
            box_length: OpenFFQuantity[unit.nanometer]
            timestep: OpenFFQuantity[unit.femtosecond]

        params = SimulationParams(
            temperature="298.15 kelvin",
            pressure="1.0 bar",
            box_length="5.0 nanometer",
            timestep="2.0 femtosecond"
        )

        assert params.temperature.magnitude == 298.15
        assert params.pressure.magnitude == 1.0
        assert params.box_length.magnitude == 5.0
        assert params.timestep.magnitude == 2.0

    def test_optional_quantity_field(self):
        """Test that optional Quantity fields work correctly"""

        class Model(pydantic.BaseModel):
            required_temp: OpenFFQuantity[unit.kelvin]
            optional_temp: OpenFFQuantity[unit.kelvin] | None = None

        # With optional field provided
        m1 = Model(required_temp="300 kelvin", optional_temp="350 kelvin")
        assert m1.optional_temp is not None
        assert m1.optional_temp.magnitude == 350

        # Without optional field
        m2 = Model(required_temp="300 kelvin")
        assert m2.optional_temp is None

    def test_json_schema_generation(self):
        """Test that JSON schema is generated correctly"""

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        schema = Model.model_json_schema()

        # Should have properties defined
        assert "properties" in schema
        assert "temperature" in schema["properties"]

        # Temperature should be a string type in the schema
        temp_schema = schema["properties"]["temperature"]
        assert temp_schema["type"] == "string"

    def test_validate_string_with_different_formats(self):
        """Test various string formats for quantities"""

        class Model(pydantic.BaseModel):
            length: OpenFFQuantity[unit.nanometer]

        # Test different string formats
        test_cases = [
            "1.5 nanometer",
            "1.5 nm",
            "15 angstrom",  # Should convert to nm
            "0.0015 micrometer",  # Should convert to nm
        ]

        for test_str in test_cases:
            m = Model(length=test_str)
            assert isinstance(m.length, openff.units.Quantity)
            assert m.length.units == unit.nanometer

    def test_nested_model_with_quantities(self):
        """Test nested pydantic models with Quantity fields"""

        class InnerModel(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        class OuterModel(pydantic.BaseModel):
            name: str
            params: InnerModel

        outer = OuterModel(
            name="test",
            params={"temperature": "300 kelvin"}
        )

        assert outer.params.temperature.magnitude == 300

        # Test JSON round-trip
        json_str = outer.model_dump_json()
        restored = OuterModel.model_validate_json(json_str)
        assert restored.params.temperature.magnitude == 300

    def test_list_of_quantities(self):
        """Test a list of Quantity values"""

        class Model(pydantic.BaseModel):
            temperatures: list[OpenFFQuantity[unit.kelvin]]

        m = Model(temperatures=["300 kelvin", "350 kelvin", "400 kelvin"])

        assert len(m.temperatures) == 3
        assert all(isinstance(t, openff.units.Quantity) for t in m.temperatures)
        assert [t.magnitude for t in m.temperatures] == [300, 350, 400]

        # Test serialization
        data = m.model_dump()
        assert all(isinstance(t, str) for t in data["temperatures"])


class TestOpenMMQuantityConversion:
    """Tests for OpenMM Quantity conversion (if OpenMM is available)"""

    @pytest.fixture
    def openmm_available(self):
        """Check if OpenMM is available"""
        try:
            import openmm.unit
            return True
        except ImportError:
            pytest.skip("OpenMM not available")

    def test_validate_from_openmm_quantity(self, openmm_available):
        """Test that OpenFFQuantity can convert from OpenMM quantities"""
        import openmm.unit as openmm_unit

        class Model(pydantic.BaseModel):
            temperature: OpenFFQuantity[unit.kelvin]

        # Create an OpenMM quantity
        openmm_temp = 300 * openmm_unit.kelvin

        m = Model(temperature=openmm_temp)
        assert isinstance(m.temperature, openff.units.Quantity)
        assert m.temperature.magnitude == 300
        assert m.temperature.units == unit.kelvin

    def test_openmm_quantity_with_unit_conversion(self, openmm_available):
        """Test OpenMM quantity conversion with unit conversion"""
        import openmm.unit as openmm_unit

        class Model(pydantic.BaseModel):
            length: OpenFFQuantity[unit.nanometer]

        # Create OpenMM quantity in angstroms
        openmm_length = 15 * openmm_unit.angstrom

        m = Model(length=openmm_length)
        assert m.length.units == unit.nanometer
        assert abs(m.length.magnitude - 1.5) < 0.01
