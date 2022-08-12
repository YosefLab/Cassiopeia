import abc
import multiprocessing
import tempfile
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.suggest import Searcher
from ray.tune.suggest.hyperopt import HyperOptSearch

from cassiopeia.data import CassiopeiaTree


class CharacterLevelCVError(Exception):
    pass


def _character_level_cv_split(
    random_char_indices: List[int],
    n_folds: int,
):
    """
    Splits random_char_indices into n_folds by using modulo arithmetic.

    If n_folds == 0, all indices go to training.

    Args:
        random_char_indices: The list of indices to split
        n_folds: The number of folds

    Returns:
        The train and cv indices for each split, respectively.
    """
    if n_folds == 0:
        indices_train = [random_char_indices[:]]
        indices_cv = [None]
        return indices_train, indices_cv
    indices_train = []
    indices_cv = []
    n_characters = len(random_char_indices)
    for split_id in range(n_folds):
        indices_cv_ = [
            random_char_indices[i]
            for i in range(n_characters)
            if i % n_folds == split_id
        ]
        indices_train_ = [
            i for i in range(n_characters) if i not in indices_cv_
        ]
        indices_cv.append(indices_cv_[:])
        indices_train.append(indices_train_[:])
    return indices_train, indices_cv


class CharacterLevelCV(abc.ABC):
    """
    Hyperparameter selection via character-level cross validation.

    Subclasses should implement the following two methods:
    - _create_space, which given a CassiopeiaTree returns the
        Ray tune hyperparameter search space.
    - _evaluate_cv_metric, which given a dictionary containing the
        hyperparameter values, the training tree, and the held out character
        matrix, returns the CV metric resulting from fitting the model with the
        given hyperparameters to the training data. Note that in some
        applications (such as Empirical Bayes), all characters are used for
        training. This is achieved by using n_folds=0. In this case, the held
        out character matrix will be None.
    Once these methids are defined, the subclass should simply call the _tune
    method to run Ray tune and obtain the best performing hyperparameter
    setting. Finally, the subclass should refit with the found hyperparameters.

    Args:
        n_hyperparams: Budget for the number of hyperparameter settings the can
            be tried.
        n_parallel_hyperparams: How many hyperparameter values to evaluate in
            parallel.
        n_folds: How many folds to use for cross-validation.
            If n_folds = 0, then no cross-validation will be performed; instead,
            all characters will be used for training and the validation tree
            will be None. This is useful for model selecting with training-based
            metrics only, such as training data log-likelihood (used by
            Empirical Bayes) or when it is believed that there is no need for
            holding out characters.
            If n_folds = -1, then leave-one-out-CV will be performed.
        n_parallel_folds: How many folds to evaluate in parallel.
        random_seed: Random seed for reproducibility.
        verbose: Verbosity level.

    Attributes:
        best_cv_metric: Best CV metric value found.
        results_df: Dataframe will all hyperparameter values explored and their
            CV metric value. (Note that best_cv_metric can be deduced from this)
        indices_train: The training character indices for each split
        indices_cv: The cv character indices for each split
    """

    def __init__(
        self,
        n_hyperparams: int = 60,
        n_parallel_hyperparams: int = 6,
        n_folds: int = 6,
        n_parallel_folds: int = 6,
        random_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self._n_hyperparams = n_hyperparams
        self._n_parallel_hyperparams = n_parallel_hyperparams
        self._n_folds = n_folds
        self._n_parallel_folds = n_parallel_folds
        self._random_seed = random_seed
        self._verbose = verbose

    def _tune(self, tree: CassiopeiaTree) -> Dict:
        """
        Searches for the best performing hyperparameter values.

        Args:
            tree: The CassiopeiaTree containing the whole training data.

        Returns:
            The best performing hyperparameters.

        Raises:
            CharacterLevelCVError if Ray tune fails.
        """
        self._tree = tree  # Temporary variable
        random_char_indices = list(range(tree.n_character))
        np.random.shuffle(random_char_indices)
        self.indices_train, self.indices_cv = _character_level_cv_split(
            random_char_indices=random_char_indices,
            n_folds=self._n_folds,
        )
        ray.init(num_cpus=self._n_parallel_hyperparams)
        with tempfile.TemporaryDirectory() as ray_tune_local_dir:
            try:
                analysis = tune.run(
                    self._trainable,
                    config=self._create_space(tree),
                    num_samples=self._n_hyperparams,
                    search_alg=self._search_alg(),
                    metric="cv_metric",
                    mode="max",
                    verbose=0,
                    local_dir=ray_tune_local_dir,
                )
            except Exception:
                ray.shutdown()
                raise CharacterLevelCVError("Ray tune failed")
            ray.shutdown()
            del self._tree
            self.results_df = analysis.results_df
            self.best_cv_metric = analysis.best_result["cv_metric"]
            return analysis.best_config

    def _search_alg(self) -> Searcher:
        return HyperOptSearch(random_state_seed=self._random_seed)

    def _trainable(self, hyperparameters: Dict) -> None:
        """
        Reports to Ray the CV metric for the given hyperameter setting.

        Args:
            hyperparameters: The hyperparameters of the model.
        """
        tree = deepcopy(self._tree)
        tune.report(
            cv_metric=self._cross_validate_hyperparameters(
                tree=tree, hyperparameters=hyperparameters
            )
        )

    def _cross_validate_hyperparameters(
        self,
        tree: CassiopeiaTree,
        hyperparameters: Dict,
    ) -> float:
        """
        Returns the CV metric for the given hyperameter setting.

        Args:
            tree: The CassiopeiaTree containing the whole training data.
            hyperparameters: The hyperparameters of the model.

        Returns:
            The CV metric for the given hyperameter setting.
        """
        verbose = self._verbose
        processes = self._n_parallel_folds
        n_folds = self._n_folds
        if n_folds == -1:
            n_folds = tree.n_character
        if verbose:
            print(
                f"Cross-validating hyperparameters:"
                f"\nhyperparameters={hyperparameters}"
            )
        params = []
        for split_id in range(
            max(n_folds, 1)
        ):  # Want to loop exactly once if n_folds == 0
            train_indices = self.indices_train[split_id]
            cv_indices = self.indices_cv[split_id]
            tree_train, tree_valid = self._cv_split(
                tree=tree,
                train_indices=train_indices,
                cv_indices=cv_indices,
            )
            params.append((hyperparameters, tree_train, tree_valid))
        with multiprocessing.Pool(processes=processes) as pool:
            map_fn = pool.map if processes > 1 else map
            cv_metric_folds = list(map_fn(self._cv_metric, params))
        if verbose:
            print(f"cv_metric_folds = {cv_metric_folds}")
            print(f"mean log likelihood = {np.mean(cv_metric_folds)}")
        return sum(cv_metric_folds)

    @staticmethod
    def _cv_split(
        tree: CassiopeiaTree,
        train_indices: List[int],
        cv_indices: Optional[List[int]],
    ) -> Tuple[CassiopeiaTree, Optional[pd.DataFrame]]:
        """
        Cross-validation split.

        Args:
            tree: The CassiopeiaTree containing the whole training data.
            train_indices: Indices of the training characters.
            cv_indices: Indices of the CV characters, or None if no characters
                will be held out.

        Returns:
            The training and CV tree. The CV tree will be None if cv_indices is
            None (e.g. for optimizing training set likelihood with Ray as
            in Empirical Bayes).
        """

        def subset_tree_at_indices(
            a_tree: CassiopeiaTree, indices: List[int]
        ) -> CassiopeiaTree:
            """
            Return the CassiopeiaTree with states only at the given indices.
            """

            def subset_states(states: List[int]) -> List[str]:
                return [states[i] for i in indices]

            res_tree = deepcopy(a_tree)
            new_cm = a_tree.character_matrix.iloc[:, indices]
            new_states = {
                node: subset_states(a_tree.get_character_states(node))
                for node in tree.nodes
            }
            res_tree.character_matrix = new_cm
            res_tree.set_all_character_states(new_states)
            return res_tree

        tree_train = subset_tree_at_indices(tree, train_indices)
        if cv_indices:
            tree_valid = subset_tree_at_indices(tree, cv_indices)
        else:
            tree_valid = None

        return tree_train, tree_valid

    @abc.abstractmethod
    def _create_space(self, tree: CassiopeiaTree) -> Dict:
        """
        Creates the Ray tune hyperparameter search space.

        Args:
            tree: The CassiopeiaTree containing the whole training data.

        Returns:
            The Ray tune hyperparameter search space.
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _cv_metric(
        args: Tuple[Dict, CassiopeiaTree, Optional[pd.DataFrame]]
    ) -> float:
        """
        The CV metric.

        Args:
            args: A tuple containing the hyperparameter values, the training
                tree, and the held out character matrix (or None if no
                characters will be held out)

        Returns:
            The CV metric resulting from fitting the model with the given
            hyperparameters to the training data. Note that in some applications
            (such as Empirical Bayes), all characters are used for training.
            This is achieved by using n_folds=0.
        """
        raise NotImplementedError
