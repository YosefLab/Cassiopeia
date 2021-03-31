"""
Abstract class CassiopeiaSolver, for the phylogenetic inference module.

All algorithms are derived classes of this abstract class, and at a minimum
store an input character matrix and implement a method called `solve`. Each
derived class stores more information around the parameters necessary for
inferring a phylogenetic tree.
"""
import abc
import pandas as pd
from typing import Dict, Optional


class CassiopeiaSolver(abc.ABC):
    """
    CassiopeiaSolver is an abstract class that all inference algorithms derive
    from. At minimum, all CassiopeiaSolver subclasses will store a character
    matrix and implement a solver procedure.

    Args:
        prior_transformation: A function defining a transformation on the priors
            in forming weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative log
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p
    """

    def __init__(self, prior_transformation: str = "negative_log"):

        self.prior_transformation = prior_transformation

    @abc.abstractmethod
    def solve(self):
        """Solves the inference problem."""
        pass
