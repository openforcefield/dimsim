from typing import Any

from pydantic import BaseModel


class MeasurementSource(BaseModel):

    doi: str

    reference: str

class CalculationSource(BaseModel):

    fidelity: str

    provenance: dict[str, Any]
