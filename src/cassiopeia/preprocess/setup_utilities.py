"""A file that stores setup utilities for Cassiopeia preprocessing.

This module supports the command line interface entry point in
``cassiopeia_preprocess.py``.
"""

import ast
import configparser
import logging
import os
from typing import Any

from cassiopeia.mixins import UnspecifiedConfigParameterError, logger
from cassiopeia.preprocess import constants


def setup(output_directory_location: str, verbose: bool) -> None:
    """
    Setup the environment for the preprocessing pipeline.

    Parameters
    ----------
    output_directory_location
        Directory to create or reuse for pipeline outputs.
    verbose
        Whether to enable verbose logging output.

    Returns
    -------
    None - Configures logging handlers and directory structure.
    """
    if not os.path.isdir(output_directory_location):
        os.mkdir(output_directory_location)

    # In addition to logging to the console, output logs to files.
    output_handler = logging.FileHandler(os.path.join(output_directory_location, "preprocess.log"))
    output_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(output_handler)

    error_handler = logging.FileHandler(os.path.join(output_directory_location, "preprocess.err"))
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)


def parse_config(config_string: str) -> dict[str, dict[str, Any]]:
    """
    Parse pipeline configuration settings from a string.

    Parameters
    ----------
    config_string
        Contents of the configuration file to interpret.

    Returns
    -------
    dict[str, dict[str, Any]] - Mapping of stage names to their parameter dictionaries.

    Raises
    ------
    UnspecifiedConfigParameterError
        Raised when required general configuration parameters are absent.
    """
    config = configparser.ConfigParser()

    # load in defaults
    config.read_dict(constants.DEFAULT_PIPELINE_PARAMETERS)

    config.read_string(config_string)

    parameters = {}
    for key in config:
        parameters[key] = {k: ast.literal_eval(v) for k, v in config[key].items()}

    # ensure that minimum items are present in config
    minimum_parameters = [
        "name",
        "output_directory",
        "reference_filepath",
        "input_files",
        "n_threads",
    ]
    for param in minimum_parameters:
        if param not in parameters["general"]:
            raise UnspecifiedConfigParameterError(
                "Please specify the following items for analysis: name, "
                "output_directory, reference_filepath, input_files, and n_threads"
            )

    # we need to add some extra parameters from the "general" settings
    parameters["convert"]["output_directory"] = parameters["general"]["output_directory"]
    parameters["convert"]["name"] = parameters["general"]["name"]
    parameters["convert"]["n_threads"] = parameters["general"]["n_threads"]
    parameters["filter_bam"]["output_directory"] = parameters["general"]["output_directory"]
    parameters["filter_bam"]["n_threads"] = parameters["general"]["n_threads"]
    parameters["error_correct_cellbcs_to_whitelist"]["output_directory"] = parameters["general"]["output_directory"]
    parameters["error_correct_cellbcs_to_whitelist"]["n_threads"] = parameters["general"]["n_threads"]
    parameters["collapse"]["output_directory"] = parameters["general"]["output_directory"]
    parameters["collapse"]["n_threads"] = parameters["general"]["n_threads"]
    parameters["resolve"]["output_directory"] = parameters["general"]["output_directory"]

    parameters["align"]["ref_filepath"] = parameters["general"]["reference_filepath"]
    parameters["align"]["ref"] = None
    parameters["align"]["n_threads"] = parameters["general"]["n_threads"]

    parameters["call_alleles"]["ref_filepath"] = parameters["general"]["reference_filepath"]
    parameters["call_alleles"]["ref"] = None

    parameters["error_correct_umis"]["allow_allele_conflicts"] = parameters["general"].get(
        "allow_allele_conflicts", False
    )
    parameters["error_correct_umis"]["n_threads"] = parameters["general"]["n_threads"]

    parameters["filter_molecule_table"]["output_directory"] = parameters["general"]["output_directory"]
    parameters["filter_molecule_table"]["allow_allele_conflicts"] = parameters["general"].get(
        "allow_allele_conflicts", False
    )

    parameters["call_lineages"]["output_directory"] = parameters["general"]["output_directory"]

    return parameters


def create_pipeline(entry, _exit, stages):
    """
    Create an ordered list of pipeline stages to execute.

    Parameters
    ----------
    entry
        Name of the first stage to run.
    _exit
        Name of the final stage to include.
    stages
        Ordered mapping of stage names to callables.

    Returns
    -------
    list[str] - Ordered names of procedures to run.
    """
    stage_names = list(stages.keys())
    start = stage_names.index(entry)
    end = stage_names.index(_exit)

    return stage_names[start : (end + 1)]
