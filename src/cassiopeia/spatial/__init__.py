"""Top level for spatial module."""

from .deprecated import impute_alleles_from_spatial_data
from .spatial_imputation import impute_alleles_spatial

__all__ = [
    "impute_alleles_spatial",
    "impute_alleles_from_spatial_data",
]
