"""
Main logic behind Cassiopeia-preprocess.

This file stores the main entry point for Cassiopeia-preprocess, and makes 
heavy use of the high level functionality in
cassiopeia.preprocess.pipeline. Here, we assume that the user
has already run CellRanger Count, or some equivalent, to obtain a BAM file that
relates cell barcodes and UMIs to sequences.

TODO(mattjones315@): include invocation instructions & pipeline specifics.
TODO(richardyz98@): create a .yml file including all necessary imports and 
dependencies
"""
import os

import argparse
import configparser
import logging
import pandas as pd
from typing import Any, Dict

from cassiopeia.preprocess import pipeline, setup_utilities

STAGES = {
    "collapse": pipeline.collapse_umis,
    "resolve": pipeline.resolve_umi_sequence,
    "align": pipeline.align_sequences,
    "call_alleles": pipeline.call_alleles,
    "error_correct": pipeline.error_correct_umis,
    "filter_molecule_table": pipeline.filter_molecule_table,
    "call_lineages": pipeline.call_lineage_groups,
}


def main():

    # --------------- Create Argument Parser & Read in Arguments -------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="Specify a config file for analysis."
    )

    args = parser.parse_args()

    config_filepath = args.config

    with open(config_filepath, "r") as f:
        pipeline_parameters = setup_utilities.parse_config(f.read())

    # pull out general parameters
    output_directory = pipeline_parameters["general"]["output_directory"]
    data_filepath = pipeline_parameters["general"]["input_file"]
    entry_point = pipeline_parameters["general"]["entry"]
    exit_point = pipeline_parameters["general"]["exit"]

    # set up output directory
    setup_utilities.setup(output_directory)

    # create pipeline plan
    pipeline_stages = setup_utilities.create_pipeline(entry_point, exit_point, STAGES)
    if entry_point == "collapse":
        data = data_filepath
    else:
        data = pd.read_csv(data_filepath, sep="\t")

    # ---------------------- Run Pipeline ---------------------- #
    for stage in pipeline_stages:

        procedure = STAGES[stage]
        data = procedure(data, **pipeline_parameters[stage])


if __name__ == "__main__":
    main()
