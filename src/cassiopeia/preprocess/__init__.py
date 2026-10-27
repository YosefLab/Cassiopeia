"""Preprocessing: from sequencing reads to character matrices."""

from . import plotting as pl
from . import qc
from .align import align
from .alleles import call_alleles, call_read
from .characters import alleles_to_characters, to_allele_table
from .clones import call_clones, clone_sizes
from .construct import Construct, Site, TapeArray, list_presets, load_construct
from .correct import correct_intbcs
from .doublets import doublet_scores, filter_doublets
from .filter import filter_cells, filter_molecules, gmm_threshold
from .integrations import filter_integrations, integration_metrics
from .layout import LAYOUTS, ReadLayout, get_layout
from .pipeline import read_molecules, run
from .plotting import (
    plot_clone_intbc_heatmap,
    plot_doublet_scores,
    plot_edit_frequency,
    plot_intbc_detection,
    plot_integrations,
    plot_umi_reads,
)
from .reference import build_reference
from .utilities import (
    compute_empirical_indel_priors,
    convert_alleletable_to_character_matrix,
    convert_alleletable_to_lineage_profile,
    convert_character_matrix_to_allele_table,
    convert_lineage_profile_to_character_matrix,
)

__all__ = [
    "Construct",
    "LAYOUTS",
    "ReadLayout",
    "Site",
    "TapeArray",
    "align",
    "alleles_to_characters",
    "build_reference",
    "call_alleles",
    "call_clones",
    "call_read",
    "clone_sizes",
    "correct_intbcs",
    "doublet_scores",
    "filter_cells",
    "filter_doublets",
    "filter_integrations",
    "filter_molecules",
    "get_layout",
    "gmm_threshold",
    "integration_metrics",
    "list_presets",
    "load_construct",
    "pl",
    "plot_clone_intbc_heatmap",
    "plot_doublet_scores",
    "plot_edit_frequency",
    "plot_integrations",
    "plot_intbc_detection",
    "plot_umi_reads",
    "qc",
    "read_molecules",
    "run",
    "to_allele_table",
    "compute_empirical_indel_priors",
    "convert_alleletable_to_character_matrix",
    "convert_alleletable_to_lineage_profile",
    "convert_character_matrix_to_allele_table",
    "convert_lineage_profile_to_character_matrix",
]
