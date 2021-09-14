import multiprocessing
import os
import subprocess
import tempfile
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate
from scipy.special import binom, logsumexp

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import ProgressReporter
import logging

from cassiopeia.data import CassiopeiaTree

from . import utils
from .BranchLengthEstimator import (
    BranchLengthEstimator,
    BranchLengthEstimatorError,
)
from .IIDExponentialMLE import IIDExponentialMLE, IIDExponentialMLEError
from .IIDExponentialBayesian import IIDExponentialBayesian


def _fit_model(model_and_tree):
    r"""
    This is used by IIDExponentialPosteriorMeanBLEGridSearchCV to
    parallelize the grid search. It must be defined here (at the top level of
    the module) for multiprocessing to be able to pickle it. (This is why
    coverage misses it)
    """
    model, tree = model_and_tree
    model.estimate_branch_lengths(tree)
    return model.log_likelihood


class IIDExponentialPosteriorMeanBLEGridSearchCV(BranchLengthEstimator):
    r"""
    Like IIDExponentialBayesian but with automatic tuning of
    hyperparameters.

    This class fits the hyperparameters of IIDExponentialBayesian based
    on data log-likelihood. I.e. is performs empirical Bayes.

    Args: TODO
    """

    def __init__(
        self,
        mutation_rates: Tuple[float] = (0,),
        birth_rates: Tuple[float] = (0,),
        sampling_probability: float = 1.0,
        discretization_level: int = 1000,
        processes: int = 6,
        verbose: bool = False,
    ):
        self.mutation_rates = mutation_rates
        self.birth_rates = birth_rates
        self.sampling_probability = sampling_probability
        self.discretization_level = discretization_level
        self.processes = processes
        self.verbose = verbose

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        mutation_rates = self.mutation_rates
        birth_rates = self.birth_rates
        sampling_probability = self.sampling_probability
        discretization_level = self.discretization_level
        processes = self.processes
        verbose = self.verbose

        lls = []
        grid = np.zeros(shape=(len(mutation_rates), len(birth_rates)))
        models = []
        mutation_and_birth_rates = []
        ijs = []
        for i, mutation_rate in enumerate(mutation_rates):
            for j, birth_rate in enumerate(birth_rates):
                if self.verbose:
                    print(
                        f"Fitting model with:\n"
                        f"mutation_rate={mutation_rate}\n"
                        f"birth_rate={birth_rate}"
                    )
                models.append(
                    IIDExponentialBayesian(
                        mutation_rate=mutation_rate,
                        birth_rate=birth_rate,
                        sampling_probability=sampling_probability,
                        discretization_level=discretization_level,
                    )
                )
                mutation_and_birth_rates.append((mutation_rate, birth_rate))
                ijs.append((i, j))
        with multiprocessing.Pool(processes=processes) as pool:
            map_fn = pool.map if processes > 1 else map
            lls = list(
                map_fn(
                    _fit_model,
                    zip(models, [deepcopy(tree) for _ in range(len(models))]),
                )
            )
        lls_and_rates = list(zip(lls, mutation_and_birth_rates))
        for ll, (i, j) in list(zip(lls, ijs)):
            grid[i, j] = ll
        lls_and_rates.sort(reverse=True)
        (best_mutation_rate, best_birth_rate,) = lls_and_rates[
            0
        ][1]
        if verbose:
            print(
                f"Refitting model with:\n"
                f"best_mutation_rate={best_mutation_rate}\n"
                f"best_birth_rate={best_birth_rate}"
            )
        final_model = IIDExponentialBayesian(
            mutation_rate=best_mutation_rate,
            birth_rate=best_birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
        )
        final_model.estimate_branch_lengths(tree)
        self.mutation_rate = best_mutation_rate
        self.birth_rate = best_birth_rate
        self.log_likelihood = final_model.log_likelihood
        self.grid = grid

    def plot_grid(
        self, figure_file: Optional[str] = None, show_plot: bool = True
    ):
        utils.plot_grid(
            grid=self.grid,
            yticklabels=self.mutation_rates,
            xticklabels=self.birth_rates,
            ylabel=r"Mutation Rate ($r$)",
            xlabel=r"Birth Rate ($\lambda$)",
            title=f"Sampling Probability = {self.sampling_probability}\nLL = {self.log_likelihood}",
            figure_file=figure_file,
            show_plot=show_plot,
        )


class EmptyReporter(ProgressReporter):
    """Never report"""
    def should_report(self, trials, done=False):
        return False

    def report(self, trials, *sys_info):
        print(f"Empty report")


class IIDExponentialPosteriorMeanBLEAutotune(BranchLengthEstimator):
    def __init__(
        self,
        discretization_level: int,
        processes: int = 6,
        num_samples: int = 100,
        space: Optional[Dict] = None,
        search_alg=None,
        verbose: int = 0,
    ) -> None:
        self.discretization_level = discretization_level
        self.processes = processes
        self.num_samples = num_samples
        self.verbose = verbose
        if space is None:
            space = {
                "mutation_rate": tune.loguniform(0.01, 5.0),
                "birth_rate": tune.loguniform(1.0, 30.0),
                "sampling_probability": tune.loguniform(0.0000001, 1.0),
            }
        self.space = space
        if search_alg is None:
            search_alg = HyperOptSearch(
                metric="log_likelihood", mode="max", random_state_seed=0
            )
        self.search_alg = search_alg

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        self.tree = tree
        ray.init(num_cpus=self.processes, logging_level=logging.FATAL, log_to_driver=False)
        try:
            analysis = tune.run(
                self._trainable,
                config=self.space,
                num_samples=self.num_samples,
                search_alg=self.search_alg,
                metric='log_likelihood',
                mode='max',
                progress_reporter=EmptyReporter(),  # Doesn't seem to work as I intend it to...
                verbose=self.verbose,
            )
        except:
            ray.shutdown()
            raise BranchLengthEstimatorError(f"Ray tune failed")
        ray.shutdown()
        self.analysis = analysis
        best_config = analysis.best_config
        self.model = self._create_model_from_config(best_config)
        self.model.estimate_branch_lengths(tree)
        # Copy over attributes associated with the bayesian estimator.
        self.mutation_rate = self.model.mutation_rate
        self.birth_rate = self.model.birth_rate
        self.sampling_probability = self.model.sampling_probability
        self.log_likelihood = self.model.log_likelihood
        del self.tree, self.search_alg, self.model, self.analysis, self.space

    def _trainable(self, config: Dict):
        model = self._create_model_from_config(config)
        model.estimate_branch_lengths(deepcopy(self.tree))
        tune.report(log_likelihood=model.log_likelihood)

    def _create_model_from_config(self, config):
        return IIDExponentialBayesian(
            mutation_rate=config['mutation_rate'],
            birth_rate=config['birth_rate'],
            discretization_level=self.discretization_level,
            sampling_probability=config['sampling_probability'],
        )


class IIDExponentialPosteriorMeanBLEAutotuneSmartMutRate(BranchLengthEstimator):
    """
    Like IIDExponentialPosteriorMeanBLEAutotune, but we use the MLE
    to get the mutation rate.
    """
    def __init__(
        self,
        discretization_level: int,
        processes: int = 6,
        num_samples: int = 100,
        space: Optional[Dict] = None,
        search_alg=None,
        verbose: int = 0,
    ) -> None:
        self.discretization_level = discretization_level
        self.processes = processes
        self.num_samples = num_samples
        self.verbose = verbose
        if space is None:
            space = {
                "birth_rate": tune.loguniform(1.0, 30.0),
                "sampling_probability": tune.loguniform(0.0000001, 1.0),
            }
        else:
            assert sorted(list(set(space.keys()))) == ["birth_rate", "sampling_probability"]
        self.space = space
        if search_alg is None:
            search_alg = HyperOptSearch(
                metric="log_likelihood", mode="max", random_state_seed=0
            )
        self.search_alg = search_alg

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        # Estimate mutation rate with MLE
        mle = IIDExponentialMLE()
        mle.estimate_branch_lengths(deepcopy(tree))
        self.space["mutation_rate"] = tune.loguniform(mle.mutation_rate / 2.0, mle.mutation_rate * 2.0)

        self.tree = tree
        ray.init(num_cpus=self.processes)
        try:
            analysis = tune.run(
                self._trainable,
                config=self.space,
                num_samples=self.num_samples,
                search_alg=self.search_alg,
                metric='log_likelihood',
                mode='max',
                progress_reporter=EmptyReporter(),  # Doesn't seem to work as I intend it to...
                verbose=self.verbose,
            )
        except:
            ray.shutdown()
            raise BranchLengthEstimatorError(f"Ray tune failed")
        ray.shutdown()
        self.analysis = analysis
        best_config = analysis.best_config
        self.model = self._create_model_from_config(best_config)
        self.model.estimate_branch_lengths(tree)
        # Copy over attributes associated with the bayesian estimator.
        self.mutation_rate = self.model.mutation_rate
        self.birth_rate = self.model.birth_rate
        self.sampling_probability = self.model.sampling_probability
        self.log_likelihood = self.model.log_likelihood
        del self.tree, self.search_alg, self.model, self.analysis, self.space

    def _trainable(self, config: Dict):
        model = self._create_model_from_config(config)
        model.estimate_branch_lengths(deepcopy(self.tree))
        tune.report(log_likelihood=model.log_likelihood)

    def _create_model_from_config(self, config):
        return IIDExponentialBayesian(
            mutation_rate=config['mutation_rate'],
            birth_rate=config['birth_rate'],
            discretization_level=self.discretization_level,
            sampling_probability=config['sampling_probability'],
        )


class IIDExponentialPosteriorMeanBLEAutotuneSmart(BranchLengthEstimator):
    """
    Like IIDExponentialPosteriorMeanBLEAutotune, but we use the MLE
    to get the mutation rate, and use moment matching to inform
    a reasonable grid for the birth rate and subsampling probability.
    """
    def __init__(
        self,
        discretization_level: int,
        processes: int = 6,
        num_samples: int = 100,
        search_alg=None,
        verbose: int = 0,
    ) -> None:
        self.discretization_level = discretization_level
        self.processes = processes
        self.num_samples = num_samples
        self.verbose = verbose
        if search_alg is None:
            search_alg = HyperOptSearch(
                metric="log_likelihood", mode="max", random_state_seed=0
            )
        self.search_alg = search_alg

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        # Estimate mutation rate with MLE
        mle = IIDExponentialMLE()
        mle.estimate_branch_lengths(deepcopy(tree))
        self.space = {}
        self.space["mutation_rate"] = tune.loguniform(mle.mutation_rate / 2.0, mle.mutation_rate * 2.0)
        self.space["e_pop_size"] = tune.loguniform(tree.n_cell / 10.0, tree.n_cell * 10.0)
        self.space["sampling_probability"] = tune.loguniform(0.0000001, 1.0)

        self.tree = tree
        ray.init(num_cpus=self.processes)
        try:
            analysis = tune.run(
                self._trainable,
                config=self.space,
                num_samples=self.num_samples,
                search_alg=self.search_alg,
                metric='log_likelihood',
                mode='max',
                progress_reporter=EmptyReporter(),  # Doesn't seem to work as I intend it to...
                verbose=self.verbose,
            )
        except:
            ray.shutdown()
            raise BranchLengthEstimatorError(f"Ray tune failed")
        ray.shutdown()
        self.analysis = analysis
        best_config = analysis.best_config
        self.model = self._create_model_from_config(best_config)
        self.model.estimate_branch_lengths(tree)
        # Copy over attributes associated with the bayesian estimator.
        self.mutation_rate = self.model.mutation_rate
        self.birth_rate = self.model.birth_rate
        self.sampling_probability = self.model.sampling_probability
        self.log_likelihood = self.model.log_likelihood
        del self.tree, self.search_alg, self.model, self.analysis, self.space

    def _trainable(self, config: Dict):
        model = self._create_model_from_config(config)
        model.estimate_branch_lengths(deepcopy(self.tree))
        tune.report(log_likelihood=model.log_likelihood)

    def _create_model_from_config(self, config):
        return IIDExponentialBayesian(
            mutation_rate=config['mutation_rate'],
            birth_rate=np.log(config['e_pop_size']) + np.log(1.0 / config['sampling_probability']),
            discretization_level=self.discretization_level,
            sampling_probability=config['sampling_probability'],
        )


class IIDExponentialPosteriorMeanBLEAutotuneSmartCV(BranchLengthEstimator):
    """
    Like IIDExponentialPosteriorMeanBLEAutotuneSmart, but we use
    held out character cross validation instead of Empirical Bayes.
    This makes the approach very similar to the CV MLE, but where the
    regularizer is the BP prior instead of the minimum_branch_length.
    """
    def __init__(
        self,
        discretization_level: int,
        processes: int = 1,
        num_samples: int = 100,
        search_alg=None,
        n_fold: int = 5,
        processes_cv: int = 5,
        verbose: int = 0,
        verbose_cv: bool = False,
    ) -> None:
        self.discretization_level = discretization_level
        self.processes = processes
        self.num_samples = num_samples
        self.verbose = verbose
        self.verbose_cv = verbose_cv
        if search_alg is None:
            search_alg = HyperOptSearch(
                metric="log_likelihood", mode="max", random_state_seed=0
            )
        self.search_alg = search_alg
        self.n_fold = n_fold
        self.processes_cv = processes_cv

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        # Estimate mutation rate with MLE
        mle = IIDExponentialMLE()
        mle.estimate_branch_lengths(deepcopy(tree))
        self.space = {}
        self.space["mutation_rate"] = tune.loguniform(mle.mutation_rate / 2.0, mle.mutation_rate * 2.0)
        self.space["e_pop_size"] = tune.loguniform(tree.n_cell / 10.0, tree.n_cell * 10.0)
        self.space["sampling_probability"] = tune.loguniform(0.0000001, 1.0)

        self.tree = tree
        random_char_indices = list(range(self.tree.n_character))
        np.random.shuffle(random_char_indices)
        self.random_char_indices = random_char_indices
        ray.init(num_cpus=self.processes)
        try:
            analysis = tune.run(
                self._trainable,
                config=self.space,
                num_samples=self.num_samples,
                search_alg=self.search_alg,
                metric='log_likelihood',
                mode='max',
                progress_reporter=EmptyReporter(),  # Doesn't seem to work as I intend it to...
                verbose=self.verbose,
            )
        except:
            ray.shutdown()
            raise BranchLengthEstimatorError(f"Ray tune failed")
        ray.shutdown()
        self.analysis = analysis
        best_config = analysis.best_config
        if self.verbose_cv:
            print(
                f"Refitting full model with:\n"
                f"config={best_config}"
            )
        self.model = self._create_model_from_config(best_config)
        self.model.estimate_branch_lengths(tree)
        # Copy over attributes associated with the bayesian estimator.
        self.mutation_rate = self.model.mutation_rate
        self.birth_rate = self.model.birth_rate
        self.sampling_probability = self.model.sampling_probability
        self.log_likelihood = self.model.log_likelihood
        del self.tree, self.search_alg, self.model, self.analysis, self.space

    def _trainable(self, config: Dict):
        tune.report(log_likelihood=self._cv_log_likelihood(deepcopy(self.tree), config))

    def _cv_log_likelihood(
        self,
        tree: CassiopeiaTree,
        config,
    ) -> float:
        verbose = self.verbose_cv
        processes = self.processes_cv
        n_fold = self.n_fold
        if n_fold == -1:
            n_fold = tree.n_character
        if verbose:
            print(
                f"Cross-validating hyperparameters:"
                f"\nconfig={config}"
            )
        n_characters = tree.n_character
        params = []
        split_size = int((n_characters + n_fold - 1) / n_fold)
        random_char_indices = self.random_char_indices
        for split_id in range(n_fold):
            held_out_character_idxs = random_char_indices[(split_id * split_size): ((split_id + 1) * split_size)]
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
        """
        TODO: Copy-pasta from MLE_CV
        """
        verbose = self.verbose_cv
        if verbose:
            print(f"IIDExponentialPosteriorMeanBLEAutotuneSmartCV held_out_character_idxs = {held_out_character_idxs}")
        tree_topology = tree.get_tree_topology()
        train_states = {}
        valid_states = {}
        for node in tree.nodes:
            state = tree.get_character_states(node)
            train_state = [state[i] for i in range(len(state)) if i not in held_out_character_idxs]
            valid_state = [state[i] for i in held_out_character_idxs]
            train_states[node] = train_state
            valid_states[node] = valid_state
        train_tree = CassiopeiaTree(tree=tree_topology)
        valid_tree = CassiopeiaTree(tree=tree_topology)
        train_tree.set_all_character_states(train_states)
        valid_tree.set_all_character_states(valid_states)
        return train_tree, valid_tree

    def _create_model_from_config(self, config):
        return IIDExponentialBayesian(
            mutation_rate=config['mutation_rate'],
            birth_rate=np.log(config['e_pop_size']) + np.log(1.0 / config['sampling_probability']),
            discretization_level=self.discretization_level,
            sampling_probability=config['sampling_probability'],
        )


def _fit_model(args):
    r"""
    This is used by IIDExponentialMLEGridSearchCV to
    parallelize the CV folds. It must be defined here (at the top level of
    the module) for multiprocessing to be able to pickle it. (This is why
    coverage misses it)
    """
    model, train_tree, valid_tree = args
    try:
        model.estimate_branch_lengths(train_tree)
        valid_tree.set_times(train_tree.get_times())
        held_out_log_likelihood = IIDExponentialMLE.model_log_likelihood(valid_tree, mutation_rate=model.mutation_rate)
    except (IIDExponentialMLEError, ValueError):
        held_out_log_likelihood = -np.inf
    return held_out_log_likelihood
