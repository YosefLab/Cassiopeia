""""
A file that stores general functionality for setting up a Cassiopeia
preprocessing instance. This file supports the command line interface entrypoint
in cassiopeia_preprocess.py.
"""
import os

import ast
import configparser
import logging
from typing import Any, Dict

from cassiopeia.mixins import UnspecifiedConfigParameterError
from cassiopeia.preprocess import constants


def setup(output_directory_location: str) -> None:
    """Setup environment for pipeline

    Args:
        output_directory_location: Where to look for, or start a new, output
            directory
    """

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


def parse_config(config_string: str) -> Dict[str, Dict[str, Any]]:
    """Parse config for pipeline.

    Args:
        config_string: Configuration file rendered as a string.

    Returns:
        A dictionary mapping parameters for each preprocessing stage.

    Raises:
        UnspecifiedConfigParameterError
    """
    config = configparser.ConfigParser()

    # load in defaults
    config.read_dict(constants.DEFAULT_PIPELINE_PARAMETERS)

    config.read_string(config_string)

    parameters = {}
    for key in config:
        parameters[key] = {
            k: ast.literal_eval(v) for k, v in config[key].items()
        }

    # ensure that minimum items are present in config
    minimum_parameters = [
        "output_directory",
        "reference_filepath",
        "input_files",
    ]
    for param in minimum_parameters:
        if param not in parameters["general"]:
            raise UnspecifiedConfigParameterError(
                "Please specify the following items for analysis: "
                "output_directory, reference_filepath, and input_files"
            )

    # we need to add some extra parameters from the "general" settings
    parameters["convert"]["output_directory"] = parameters["general"][
        "output_directory"
    ]
    parameters["filter"]["output_directory"] = parameters["general"][
        "output_directory"
    ]
    parameters["error_correct_barcodes"]["output_directory"] = parameters[
        "general"
    ]["output_directory"]
    parameters["collapse"]["output_directory"] = parameters["general"][
        "output_directory"
    ]
    parameters["resolve"]["output_directory"] = parameters["general"][
        "output_directory"
    ]

    parameters["align"]["ref_filepath"] = parameters["general"][
        "reference_filepath"
    ]
    parameters["align"]["ref"] = None

    parameters["call_alleles"]["ref_filepath"] = parameters["general"][
        "reference_filepath"
    ]
    parameters["call_alleles"]["ref"] = None

    parameters["error_correct_umis"]["allow_allele_conflicts"] = parameters[
        "general"
    ].get("allow_allele_conflicts", False)

    parameters["filter_molecule_table"]["output_directory"] = parameters[
        "general"
    ]["output_directory"]
    parameters["filter_molecule_table"]["allow_allele_conflicts"] = parameters[
        "general"
    ].get("allow_allele_conflicts", False)

    parameters["call_lineages"]["output_directory"] = parameters["general"][
        "output_directory"
    ]

    return parameters


def create_pipeline(entry, _exit, stages):
    """Create pipeline given an entry point.

    Args:
        entry: a string representing a stage in start at.
        _exit: a string representing the stage to stop.
        stages: a list of stages in order of the general pipeline.

    Returns:
        A list of procedures to run.
    """

    stage_names = list(stages.keys())
    start = stage_names.index(entry)
    end = stage_names.index(_exit)

    return stage_names[start : (end + 1)]
