"""
Stores constants for the ProcessingPipeline module
"""

BAM_CONSTANTS = {
    "CELL_BC_TAG": "CB",
    "UMI_TAG": "UR",
    "NUM_READS_TAG": "ZR",
    "CLUSTER_ID_TAG": "ZC",
    "N_Q": 2,
    "HIGH_Q": 31,
    "LOW_Q": 10,
}

DNA_SUBSTITUTION_MATRIX = {
    "A": {"A": 5, "T": -4, "C": -4, "G": -4, "Z": 0, "N": 0},
    "T": {"A": -4, "T": 5, "C": -4, "G": -4, "Z": 0, "N": 0},
    "C": {"A": -4, "T": -4, "C": 5, "G": -4, "Z": 0, "N": 0},
    "G": {"A": -4, "T": -4, "C": -4, "G": 5, "Z": 0, "N": 0},
    "Z": {"A": 0, "T": 0, "C": 0, "G": 0, "Z": 0, "N": 0},
    "N": {"A": 0, "T": 0, "C": 0, "G": 0, "Z": 0, "N": 0},
}

DEFAULT_PIPELINE_PARAMETERS = {
    "general": {"entry": "'collapse'", "exit": "'call_lineages'"},
    "collapse": {"max_hq_mismatches": 3, "max_indels": 2, "force_sort": True},
    "resolve": {
        "min_avg_reads_per_umi": 2.0,
        "min_umi_per_cell": 10,
        "plot": True,
    },
    "align": {"gap_open_penalty": 20, "gap_extend_penalty": 1},
    "call_alleles": {
        "barcode_interval": (20, 34),
        "cutsite_locations": [112, 166, 220],
        "cutsite_width": 12,
        "context": True,
        "context_size": 5,
    },
    "error_correct": {
        "_id": "'batch1'",
        "max_umi_distance": 2,
        "verbose": False,
    },
    "filter_molecule_table": {
        "min_umi_per_cell": 10,
        "min_avg_reads_per_umi": 2.0,
        "umi_read_thresh": -1,
        "intbc_prop_thresh": 0.5,
        "intbc_umi_thresh": 10,
        "intbc_dist_thresh": 1,
        "doublet_threshold": 0.35,
        "plot": True,
        "verbose": False,
    },
    "call_lineages": {
        "min_umi_per_cell": 10,
        "min_avg_reads_per_umi": 2.0,
        "min_cluster_prop": 0.005,
        "min_intbc_thresh": 0.05,
        "inter_doublet_threshold": 0.35,
        "kinship_thresh": 0.25,
        "verbose": False,
        "plot": True,
    },
}
