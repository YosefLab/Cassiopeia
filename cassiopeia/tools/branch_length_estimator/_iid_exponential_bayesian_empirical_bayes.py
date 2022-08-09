from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ray import tune

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins.errors import BranchLengthEstimatorError
from cassiopeia.model_selection import CharacterLevelCV, CharacterLevelCVError
from cassiopeia.tools.branch_length_estimator import (BranchLengthEstimator,
                                                      IIDExponentialBayesian,
                                                      IIDExponentialMLE)


def _create_space_iid_exponential_bayesian(
    tree: CassiopeiaTree, space: Dict
) -> Dict:
    """
    Args:
        space: Pre-computed values to use for the space.
    """
    space = space.copy()
    if "mutation_rate" not in space:
        mle = IIDExponentialMLE()
        mle.estimate_branch_lengths(deepcopy(tree))
        space["mutation_rate"] = tune.loguniform(
            mle.mutation_rate / 2.0, mle.mutation_rate * 2.0
        )
    if "e_pop_size" not in space:
        space["e_pop_size"] = tune.loguniform(
            max(tree.n_cell / 10.0, 3.0), tree.n_cell * 10.0
        )
    if "sampling_probability" not in space:
        space["sampling_probability"] = tune.loguniform(0.0000001, 1.0)
    return space


def _transform_hyperparameters_iid_exponential_bayesian(
    hyperparameters: Dict,
) -> Dict:
    return {
        "mutation_rate": hyperparameters["mutation_rate"],
        "birth_rate": np.log(hyperparameters["e_pop_size"])
        + np.log(1.0 / hyperparameters["sampling_probability"]),
        "sampling_probability": hyperparameters["sampling_probability"],
        "discretization_level": 600,
    }


class IIDExponentialBayesianEmpiricalBayes(
    BranchLengthEstimator, CharacterLevelCV
):
    """
    Empirical Bayes for IIDExponentialBayesian.

    We use n_folds=0 since we are using a training metric.
    """

    def __init__(
        self,
        n_hyperparams: int = 60,
        n_parallel_hyperparams: int = 6,
        random_seed: int = 0,
        verbose: bool = False,
        space: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            n_hyperparams=n_hyperparams,
            n_parallel_hyperparams=n_parallel_hyperparams,
            n_folds=0,
            n_parallel_folds=1,
            random_seed=random_seed,
            verbose=verbose,
        )
        if space is None:
            space = {}
        self._space = space

    def _create_space(self, tree: CassiopeiaTree):
        space = _create_space_iid_exponential_bayesian(tree, self._space)
        return space

    @staticmethod
    def _cv_metric(
        args: Tuple[Dict, CassiopeiaTree, Optional[pd.DataFrame]]
    ) -> float:
        hyperparameters, tree_train, cm_valid = args
        model = IIDExponentialBayesian(
            **_transform_hyperparameters_iid_exponential_bayesian(
                hyperparameters
            )
        )
        try:
            model.estimate_branch_lengths(tree_train)
            return model.log_likelihood
        except Exception:
            return -np.inf
        return

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        try:
            hyperparameters = self._tune(tree)
        except CharacterLevelCVError:
            raise BranchLengthEstimatorError("Ray tune failed.")
        model = IIDExponentialBayesian(
            **_transform_hyperparameters_iid_exponential_bayesian(
                hyperparameters
            )
        )
        model.estimate_branch_lengths(tree)
        self.mutation_rate = model.mutation_rate
        self.birth_rate = model.birth_rate
        self.sampling_probability = model.sampling_probability
        self.discretization_level = model.discretization_level
        self.log_likelihood = model.log_likelihood
