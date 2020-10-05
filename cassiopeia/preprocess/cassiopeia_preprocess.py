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
from typing import Any, Dict

import cassiopeia
from cassiopeia.pp import setup_utilities

def main():

    # --------------- Create Argument Parser & Read in Arguments -------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bam_file", type=str, help="Specify a BAM file to process."
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="Specify an output directory to store results.",
    )
    parser.add_argument(
        "config", type=str, help="Specify a config file for analysis."
    )

    args = parser.parse_args()
    
    config_filepath = args.config
    
    with open(config_filepath, 'r') as f:
        pipeline_parameters = setup_utilities.parse_config(f.read())

    # feed into base_dir
    setup_utilities.setup(pipeline_parameters['general']['output_directory'])
    

    # ---------------------- Run Pipeline ---------------------- #
    # Collapse UMIs
    cassiopeia.pp.collapse_umis(output_directory, bam_filepath)

    # Resolve Sequences
    cassiopeia.pp.resolve_umi_sequence()

    # align sequences
    cassiopeia.pp.align_sequences()

    # call alleles
    cassiopeia.pp.call_alleles()

    # error correct umis

    # filter molecule tables

    # call lineages


if __name__ == "__main__":
    main()
