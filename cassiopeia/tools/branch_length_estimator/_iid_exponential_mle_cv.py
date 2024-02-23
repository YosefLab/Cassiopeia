from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins.errors import BranchLengthEstimatorError
from cassiopeia.model_selection import CharacterLevelCV, CharacterLevelCVError

from ._iid_exponential_mle import IIDExponentialMLE


def _cv_metric_ble(
    ble_class, args: Tuple[Dict, CassiopeiaTree, Optional[pd.DataFrame]]
) -> float:
    hyperparameters, tree_train, tree_valid = args
    model = ble_class(**hyperparameters)
    try:
        tree_train.set_character_states_at_leaves(tree_train.character_matrix)
        tree_train.reconstruct_ancestral_characters()
        tree_train.set_character_states(
            tree_train.root, [0] * tree_train.n_character
        )
        model.estimate_branch_lengths(tree_train)
    except Exception:
        return -np.inf
    if tree_valid is not None:
        tree_valid.set_times(tree_train.get_times())
        try:
            return IIDExponentialMLE.model_log_likelihood(
                tree=tree_valid, mutation_rate=model.mutation_rate
            )
        except Exception:
            raise ValueError(
                f"Failed to get model_log_likelihood."
                f"\ntree_valid = {tree_valid.get_newick()}"
                f"\nhyperparameters = {hyperparameters}"
            )
    else:
        # No CV - training log-likelihood
        return IIDExponentialMLE.model_log_likelihood(
            tree=tree_train, mutation_rate=model.mutation_rate
        )


def _create_space_iid_exponential_mle(
    tree: CassiopeiaTree, space: Dict
) -> Dict:
    """
    Args:
        space: Pre-computed values to use for the space.
    """
    space = space.copy()
    if "pseudomutations" not in space:
        space["pseudomutations"] = tune.grid_search(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        )
    return space


def _transform_hyperparameters_iid_exponential_mle(
    hyperparameters: Dict,
) -> Dict:
    return {
        "pseudo_mutations_per_edge": hyperparameters["pseudomutations"],
        "pseudo_non_mutations_per_edge": hyperparameters["pseudomutations"],
    }


class IIDExponentialMLECrossValidated(CharacterLevelCV):
    """
    Cross-validated IIDExponentialMLE.

    Uses the IIDExponentialMLE model log-likelihood as the metric.
    """

    def __init__(
        self,
        n_parallel_hyperparams: int = 6,
        n_folds: int = 6,
        n_parallel_folds: int = 6,
        verbose: bool = False,
        space: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            n_hyperparams=1,  # >1 repeats the grid, we don't want this.
            n_parallel_hyperparams=n_parallel_hyperparams,
            n_folds=n_folds,
            n_parallel_folds=n_parallel_folds,
            verbose=verbose,
        )
        if space is None:
            space = {}
        self._space = deepcopy(space)

    def _create_space(self, tree: CassiopeiaTree):
        space = _create_space_iid_exponential_mle(tree, self._space)
        return space

    def _search_alg(self) -> BasicVariantGenerator:
        return BasicVariantGenerator()

    @staticmethod
    def _cv_metric(
        args: Tuple[Dict, CassiopeiaTree, Optional[pd.DataFrame]]
    ) -> float:
        hyperparameters, tree_train, tree_valid = args
        hyperparameters = _transform_hyperparameters_iid_exponential_mle(
            hyperparameters
        )
        args = (hyperparameters, tree_train, tree_valid)
        return _cv_metric_ble(ble_class=IIDExponentialMLE, args=args)

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        try:
            hyperparameters = self._tune(tree)
        except CharacterLevelCVError:
            raise BranchLengthEstimatorError("Ray tune failed.")
        model = IIDExponentialMLE(
            **_transform_hyperparameters_iid_exponential_mle(hyperparameters)
        )
        model.estimate_branch_lengths(tree)
        self.mutation_rate = model.mutation_rate
        self.minimum_branch_length = model.minimum_branch_length
        self.pseudo_mutations_per_edge = model.pseudo_mutations_per_edge
        self.pseudo_non_mutations_per_edge = model.pseudo_non_mutations_per_edge
        self.penalized_log_likelihood = model.penalized_log_likelihood
        self.log_likelihood = model.log_likelihood
