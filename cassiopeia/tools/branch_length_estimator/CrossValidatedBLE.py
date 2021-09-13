import abc
import multiprocessing
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from cassiopeia.data import CassiopeiaTree

from .BranchLengthEstimator import (
    BranchLengthEstimator,
    BranchLengthEstimatorError,
)
from .IIDExponentialBayesian import IIDExponentialBayesian
from .IIDExponentialMLE import IIDExponentialMLE, IIDExponentialMLEError


class CrossValidatedBLEError(BranchLengthEstimatorError):
    pass


class CrossValidatedBLE(BranchLengthEstimator, abc.ABC):
    """
    Uses character-level cross validation to tune the hyperparameters
    of the model.

    n_folds = 0 means that it is a training-based procedure only,
    such as in Empirical Bayes. This is a (somewhat hacky) way
    of leveraging ray tune to perform Empirical Bayes.

    TODO
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

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        """
        TODO
        """
        self._tree = tree
        random_char_indices = list(range(tree.n_character))
        np.random.shuffle(random_char_indices)
        self._random_char_indices = random_char_indices
        ray.init(num_cpus=self._n_parallel_hyperparams)
        try:
            analysis = tune.run(
                self._trainable,
                config=self._create_space(tree),
                num_samples=self._n_hyperparams,
                search_alg=HyperOptSearch(
                    metric="cv_metric",
                    mode="max",
                    random_state_seed=self._random_seed,
                ),
                metric="cv_metric",
                mode="max",
                verbose=0,
            )
        except:
            ray.shutdown()
            raise CrossValidatedBLEError("Ray tune failed")
        ray.shutdown()
        del (
            self._tree,
            self._random_char_indices,
        )
        best_config = analysis.best_config
        self.results_df = analysis.results_df
        self.best_cv_metric = analysis.best_result["cv_metric"]
        if self._verbose:
            print(f"Refitting full model with:\n" f"config={best_config}")
        model = self._create_model_from_config(best_config)
        model.estimate_branch_lengths(tree)
        # Copy over attributes of underlying model
        for attr in vars(model):
            setattr(self, attr, getattr(model, attr))

    def _trainable(self, config: Dict):
        tune.report(
            cv_metric=self._cv_metric_from_config(deepcopy(self._tree), config)
        )

    def _cv_metric_from_config(
        self,
        tree: CassiopeiaTree,
        config: Dict,
    ) -> float:
        verbose = self._verbose
        processes = self._n_parallel_folds
        n_folds = self._n_folds
        if n_folds == -1:
            n_folds = tree.n_character
        if verbose:
            print(f"Cross-validating hyperparameters:" f"\nconfig={config}")
        n_characters = tree.n_character
        params = []
        random_char_indices = self._random_char_indices
        for split_id in range(
            max(n_folds, 1)
        ):  # Want to loop exactly once if n_folds == 0
            if n_folds == 0:
                train_indices = random_char_indices[:]
                # CassiopeiaTree complains if we don't use at least
                # one character... Doesn't matter cause it won't be
                # used anyway...
                cv_indices = [0]
            else:
                split_size = int((n_characters + n_folds - 1) / n_folds)
                cv_indices = random_char_indices[
                    (split_id * split_size) : ((split_id + 1) * split_size)
                ]
                train_indices = [
                    i for i in range(n_characters) if i not in cv_indices
                ]
            train_tree, valid_tree = self._cv_split(
                tree=tree,
                train_indices=train_indices,
                cv_indices=cv_indices,
            )
            model = self._create_model_from_config(config)
            params.append((model, train_tree, valid_tree, self._cv_metric))
        with multiprocessing.Pool(processes=processes) as pool:
            map_fn = pool.map if processes > 1 else map
            cv_metric_folds = list(map_fn(_get_cv_metric_folds, params))
        if verbose:
            print(f"cv_metric_folds = {cv_metric_folds}")
            print(f"mean log likelihood = {np.mean(cv_metric_folds)}")
        return np.mean(np.array(cv_metric_folds))

    def _cv_split(
        self,
        tree: CassiopeiaTree,
        train_indices: List[int],
        cv_indices: List[int],
    ) -> Tuple[CassiopeiaTree, CassiopeiaTree]:
        verbose = self._verbose
        if verbose:
            print("CrossValidatedBLE train_indices " f"= {train_indices}")
            print("CrossValidatedBLE cv_indices " f"= {cv_indices}")
        tree_topology = tree.get_tree_topology()
        train_states = {}
        valid_states = {}
        for node in tree.nodes:
            state = tree.get_character_states(node)
            train_state = [state[i] for i in train_indices]
            valid_state = [state[i] for i in cv_indices]
            train_states[node] = train_state
            valid_states[node] = valid_state
        train_tree = CassiopeiaTree(tree=tree_topology)
        valid_tree = CassiopeiaTree(tree=tree_topology)
        train_tree.set_all_character_states(train_states)
        valid_tree.set_all_character_states(valid_states)
        return train_tree, valid_tree

    @staticmethod
    @abc.abstractmethod
    def _create_space(tree: CassiopeiaTree):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _create_model_from_config(config: Dict):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _cv_metric(fitted_model, valid_tree: CassiopeiaTree) -> float:
        raise NotImplementedError


def _get_cv_metric_folds(args: Tuple):
    """
    TODO
    """
    model, train_tree, valid_tree, _cv_metric = args
    try:
        model.estimate_branch_lengths(train_tree)
        valid_tree.set_times(train_tree.get_times())
        cv_metric = _cv_metric(model, valid_tree)
    except:
        cv_metric = -np.inf
    return cv_metric


def _create_space_iid_exponential_bayesian(tree: CassiopeiaTree) -> Dict:
    mle = IIDExponentialMLE()
    mle.estimate_branch_lengths(deepcopy(tree))
    space = {}
    space["mutation_rate"] = tune.loguniform(
        mle.mutation_rate / 2.0, mle.mutation_rate * 2.0
    )
    space["e_pop_size"] = tune.loguniform(
        tree.n_cell / 10.0, tree.n_cell * 10.0
    )
    space["sampling_probability"] = tune.loguniform(0.0000001, 1.0)
    return space


def _create_model_from_config_iid_exponential_bayesian(config: Dict) -> Dict:
    return IIDExponentialBayesian(
        mutation_rate=config["mutation_rate"],
        birth_rate=np.log(config["e_pop_size"])
        + np.log(1.0 / config["sampling_probability"]),
        sampling_probability=config["sampling_probability"],
        discretization_level=600,
    )


class IIDExponentialBayesianCrossValidated(CrossValidatedBLE):
    """
    TODO
    """

    @staticmethod
    def _create_space(tree: CassiopeiaTree):
        return _create_space_iid_exponential_bayesian(tree)

    @staticmethod
    def _create_model_from_config(config: Dict):
        return _create_model_from_config_iid_exponential_bayesian(config)

    @staticmethod
    def _cv_metric(fitted_model, valid_tree: CassiopeiaTree) -> float:
        return IIDExponentialMLE.model_log_likelihood(
            valid_tree,
            mutation_rate=fitted_model.mutation_rate,
        )


class IIDExponentialMLECrossValidated(CrossValidatedBLE):
    """
    TODO
    """

    @staticmethod
    def _create_space(tree: CassiopeiaTree):
        space = {}
        space["minimum_branch_length"] = tune.loguniform(
            0.0000001, 1.0 / (tree.get_edge_depth() + 1e-8)
        )
        return space

    @staticmethod
    def _create_model_from_config(config: Dict):
        return IIDExponentialMLE(**config)

    @staticmethod
    def _cv_metric(fitted_model, valid_tree: CassiopeiaTree) -> float:
        return IIDExponentialMLE.model_log_likelihood(
            valid_tree,
            mutation_rate=fitted_model.mutation_rate,
        )


class IIDExponentialBayesianEmpiricalBayes(CrossValidatedBLE):
    """
    TODO
    """

    def __init__(
        self,
        n_hyperparams: int = 60,
        n_parallel_hyperparams: int = 6,
        random_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            n_hyperparams=n_hyperparams,
            n_parallel_hyperparams=n_parallel_hyperparams,
            n_folds=0,
            n_parallel_folds=1,
            random_seed=random_seed,
            verbose=verbose,
        )

    @staticmethod
    def _create_space(tree: CassiopeiaTree):
        return _create_space_iid_exponential_bayesian(tree)

    @staticmethod
    def _create_model_from_config(config: Dict):
        return _create_model_from_config_iid_exponential_bayesian(config)

    @staticmethod
    def _cv_metric(fitted_model, valid_tree: CassiopeiaTree) -> float:
        return fitted_model.log_likelihood
