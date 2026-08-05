"""Generador de datos sintéticos a escala (§20 F0, Anexo C)."""
from .generator import SyntheticGenerator, generate_universe
from .cnel_export import canonical_to_cnel, export_cnel_dataset

__all__ = ["SyntheticGenerator", "generate_universe",
           "canonical_to_cnel", "export_cnel_dataset"]
