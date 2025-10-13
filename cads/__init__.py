# Expose CADS callback and config at package level
from .cads import CADSConditionAnnealer as CADS, CADSConfig, CADSConditionAnnealer

__all__ = [
    "CADS", "CADSConfig", "CADSConditionAnnealer",
]
