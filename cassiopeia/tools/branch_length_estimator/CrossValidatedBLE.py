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

    TODO
    """

    def __init__(
        self,
        n_hyperparams: int = 60,
        n_parallel_hyperparams: int = 6,
        n_folds: int = 6,
        n_parallel_folds: int = 6,
        verbose: bool = False,
    ) -> None:
        self._n_hyperparams = n_hyperparams
        self._n_parallel_hyperparams = n_parallel_hyperparams
        self._n_folds = n_folds
        self._n_parallel_folds = n_parallel_folds
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
                search_alg=HyperOptSearch(metric="log_likelihood", mode="max"),
                metric="log_likelihood",
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
        if self._verbose:
            print(f"Refitting full model with:\n" f"config={best_config}")
        model = self._create_model_from_config(best_config)
        model.estimate_branch_lengths(tree)
        # Copy over attributes of underlying model
        for attr in vars(model):
            setattr(self, attr, getattr(model, attr))

    def _trainable(self, config: Dict):
        tune.report(
            log_likelihood=self._cv_log_likelihood(deepcopy(self._tree), config)
        )

    def _cv_log_likelihood(
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
        split_size = int((n_characters + n_folds - 1) / n_folds)
        random_char_indices = self._random_char_indices
        for split_id in range(n_folds):
            held_out_character_idxs = random_char_indices[
                (split_id * split_size) : ((split_id + 1) * split_size)
            ]
            train_tree, valid_tree = self._cv_split(
                tree=tree, held_out_character_idxs=held_out_character_idxs
            )
            model = self._create_model_from_config(config)
            params.append((model, train_tree, valid_tree))
        with multiprocessing.Pool(processes=processes) as pool:
            map_fn = pool.map if processes > 1 else map
            log_likelihood_folds = list(map_fn(_fit_model, params))
        if verbose:
            print(f"log_likelihood_folds = {log_likelihood_folds}")
            print(f"mean log likelihood = {np.mean(log_likelihood_folds)}")
        return np.mean(np.array(log_likelihood_folds))

    def _cv_split(
        self, tree: CassiopeiaTree, held_out_character_idxs: List[int]
    ) -> Tuple[CassiopeiaTree, CassiopeiaTree]:
        verbose = self._verbose
        if verbose:
            print(
                "CrossValidatedBLE held_out_character_idxs "
                f"= {held_out_character_idxs}"
            )
        tree_topology = tree.get_tree_topology()
        train_states = {}
        valid_states = {}
        for node in tree.nodes:
            state = tree.get_character_states(node)
            train_state = [
                state[i]
                for i in range(len(state))
                if i not in held_out_character_idxs
            ]
            valid_state = [state[i] for i in held_out_character_idxs]
            train_states[node] = train_state
            valid_states[node] = valid_state
        train_tree = CassiopeiaTree(tree=tree_topology)
        valid_tree = CassiopeiaTree(tree=tree_topology)
        train_tree.set_all_character_states(train_states)
        valid_tree.set_all_character_states(valid_states)
        return train_tree, valid_tree

    @abc.abstractmethod
    def _create_space(self, tree: CassiopeiaTree):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_model_from_config(self, config: Dict):
        raise NotImplementedError


def _fit_model(args):
    """
    TODO
    """
    model, train_tree, valid_tree = args
    try:
        model.estimate_branch_lengths(train_tree)
        valid_tree.set_times(train_tree.get_times())
        held_out_log_likelihood = IIDExponentialMLE.model_log_likelihood(
            valid_tree, mutation_rate=model.mutation_rate
        )
    except (IIDExponentialMLEError, ValueError):
        held_out_log_likelihood = -np.inf
    return held_out_log_likelihood


class IIDExponentialBayesianCrossValidated(CrossValidatedBLE):
    def _create_space(self, tree: CassiopeiaTree):
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

    def _create_model_from_config(self, config: Dict):
        return IIDExponentialBayesian(
            mutation_rate=config["mutation_rate"],
            birth_rate=np.log(config["e_pop_size"])
            + np.log(1.0 / config["sampling_probability"]),
            sampling_probability=config["sampling_probability"],
            discretization_level=600,
        )


class IIDExponentialMLECrossValidated(CrossValidatedBLE):
    def _create_space(self, tree: CassiopeiaTree):
        space = {}
        space["minimum_branch_length"] = tune.loguniform(
            0.0000001, 1.0 / (tree.get_edge_depth() + 1e-8)
        )
        return space

    def _create_model_from_config(self, config: Dict):
        return IIDExponentialMLE(**config)
