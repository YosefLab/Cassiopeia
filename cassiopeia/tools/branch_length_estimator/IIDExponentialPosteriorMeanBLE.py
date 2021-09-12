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

from cassiopeia.data import CassiopeiaTree

from . import utils
from .BranchLengthEstimator import (
    BranchLengthEstimator,
    BranchLengthEstimatorError,
)
from .IIDExponentialMLE import IIDExponentialMLE
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
        verbose: bool = False,
    ) -> None:
        self.discretization_level = discretization_level
        self.processes = processes
        self.num_samples = num_samples
        self.verbose = verbose
        if space is None:
            space = {
                "mutation_rate": tune.loguniform(0.01, 5.0),
                "birth_rate": tune.loguniform(0.01, 30.0),
                "sampling_probability": tune.loguniform(0.0000001, 1.0),
            }
        self.space = space
        if search_alg is None:
            search_alg = HyperOptSearch(
                metric="log_likelihood", mode="max"
            )
        self.search_alg = search_alg

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
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
        self.log_likelihood = self.model.log_likelihood

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
        verbose: bool = False,
    ) -> None:
        self.discretization_level = discretization_level
        self.processes = processes
        self.num_samples = num_samples
        self.verbose = verbose
        if space is None:
            space = {
                "birth_rate": tune.loguniform(0.01, 30.0),
                "sampling_probability": tune.loguniform(0.0000001, 1.0),
            }
        else:
            assert sorted(list(set(space.keys()))) == ["birth_rate", "sampling_probability"]
        self.space = space
        if search_alg is None:
            search_alg = HyperOptSearch(
                metric="log_likelihood", mode="max"
            )
        self.search_alg = search_alg

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        # Estimate mutation rate with MLE
        mle = IIDExponentialMLE()
        mle.estimate_branch_lengths(deepcopy(tree))
        self.space["mutation_rate"] = mle.mutation_rate

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
        self.log_likelihood = self.model.log_likelihood

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
