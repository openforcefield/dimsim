"""Base pydantic models"""

import hashlib
import sys

import pydantic


class BaseModel(pydantic.BaseModel):
    """Base pydantic model for all other models to inherit from"""

    @classmethod
    def from_yaml(cls, yaml_str: str):
        """Create an instance of the model from a YAML string"""
        import yaml
        data = yaml.safe_load(yaml_str)
        return cls(**data)

    def to_yaml(self) -> str:
        """Convert the model instance to a YAML string"""
        import yaml
        return yaml.safe_dump(self.model_dump())

    def _get_hash(self) -> int:
        """
        Get a hash that uniquely identifies this configuration
        and corresponds with the typical Python hash() function.
        """

        serialized = self.model_dump_json()
        raw = int(hashlib.sha256(serialized.encode("utf-8")).hexdigest(), 16)

        if sys.hash_info.width == 64:
            mod = (1 << 61) - 1  # Mersenne prime for 64-bit hash
        else:
            mod = (1 << 31) - 1

        return raw % mod
