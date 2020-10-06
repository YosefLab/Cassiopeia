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

from cassiopeia.pp import constants


class UnspecifiedConfigParameterError(Exception):
    pass


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
        "input_file",
    ]
    for param in minimum_parameters:
        if param not in parameters["general"]:
            raise UnspecifiedConfigParameterError(
                "Please specify the following items for analysis: "
                "output_directory, reference_filepath, and input_file"
            )

    return parameters
