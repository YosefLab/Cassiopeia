"""
Main logic behind Cassiopeia-preprocess.

This file stores the main entry point for Cassiopeia-preprocess, and makes 
heavy use of the high level functionality in
cassiopeia.ProcessingPipeline.process.pipeline. Here, we assume that the user
has already run CellRanger Count, or some equivalent, to obtain a BAM file that
relates cell barcodes and UMIs to sequences.

TODO(mattjones315@): include invocation instructions & pipeline specifics.
TODO(richardyz98@): create a .yml file including all necessary imports and 
dependencies
"""
import argparse
import logging
import os

from cassiopeia.ProcessingPipeline.process import pipeline


def setup(output_directory_location):

    if not os.path.isdir(output_directory_location):
        os.mkdir(output_directory_location)

    logging.basicConfig(
        filename=os.path.join(output_directory_location, "preprocess.log"),
        level=logging.INFO,
    )
    logging.basicConfig(
        filename=os.path.join(output_directory_location, "preprocess.err"),
        level=logging.ERROR,
    )


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

    bam_filepath = args.bam_file
    output_directory = args.output_directory
    config_filepath = args.config

    setup(output_directory)  # feed into base_dir

    # ---------------------- Run Pipeline ---------------------- #
    # Collapse UMIs
    pipeline.collapseUMIs(output_directory, bam_filepath)

    # Resolve Sequences
    pipeline.resolve_UMI_sequence()

    # align sequences
    pipeline.align_sequences()


if __name__ == "__main__":
    main()
