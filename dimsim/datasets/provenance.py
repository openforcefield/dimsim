from typing import Any

from pydantic import BaseModel


class MeasurementSource(BaseModel):

    doi: str | None = None

    reference: str | None = None


class CalculationSource(BaseModel):

    fidelity: str | None = None

    provenance: dict[str, Any] | None = None
