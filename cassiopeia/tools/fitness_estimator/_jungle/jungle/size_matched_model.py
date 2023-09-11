import ast
import json

import numpy as np
import scipy  # import scipy so its available in namespace for evaluating distributions

# TODO evaluate distribution in global namespace, so that import is not necessary here


class SizeMatchedModel:
    def __init__(self, bins, params, distribution, name=None):
        """Initialize SizeMatchedModel from a list of bins, a list of parameters, and a distribution"""
        assert (
            len(params) == len(bins) - 1
        ), "Length of params must be one less than length of bins"
        self.bins = bins
        self.params = params
        self.distribution = distribution
        self.name = name

    # @classmethod
    # def from_dict(cls, bin_to_params, distribution):
    #     """ Initialize SizeMatchedModel from a dictionary of bin-parameter mappings and a distribution """
    #     bins = bin_to_params.keys()
    #     params = bin_to_params.values()
    #     return SizeMatchedModel(bins, params, distribution)

    @classmethod
    def from_json(cls, filename):
        """Load SizeMatchedModel from JSON file"""
        with open(filename) as f:
            attributes_str = json.load(f)
        attributes = dict()
        attributes["bins"] = ast.literal_eval(attributes_str["bins"])
        attributes["params"] = ast.literal_eval(attributes_str["params"])
        attributes["name"] = ast.literal_eval(attributes_str["name"])
        distribution = eval(
            attributes_str["distribution"]
        )()  # evaluate class name and instantiate
        return SizeMatchedModel(
            attributes["bins"],
            attributes["params"],
            distribution,
            attributes["name"],
        )

    def to_json(self, outfile):
        """Write SizeMatchedModel to JSON file"""
        attributes = dict()
        attributes["bins"] = json.dumps(self.bins)
        attributes["params"] = json.dumps(self.params)
        attributes["name"] = json.dumps(self.name)

        # Get distribution class name
        # Distribution is a function, so we need to parse out the class name to save it in JSON format
        distribution_str = (
            self.distribution.__class__.__module__
            + "."
            + self.distribution.__class__.__name__
        )
        attributes["distribution"] = distribution_str

        with open(outfile, "w") as out:
            json.dump(attributes, out)

    def _params_for_size(self, size, strict_bounds=True):
        """Find parameters for bin that matches size"""

        # Find matching bin based on size
        bin_index = np.digitize(
            size, self.bins
        )  # digitize returns the index of the bin to which value belongs

        if (bin_index == 0 or bin_index == len(self.bins)) and strict_bounds:
            # if strict bounds are used, only allow values that fall strictly within the bins
            raise ValueError(
                "Size must be within bounds of bins (if strict_bounds=True)"
            )

        if bin_index == 0 and not strict_bounds:
            # if loose bounds are used, values less than the bounds of bins should be set to smallest bin
            bin_index = 1

        if bin_index == len(self.bins) and not strict_bounds:
            # if loose bounds are used, values greater than the bounds of binds should be set to largest bin
            bin_index = len(self.bins) - 1

        # Adjust bin index to match indexing of params
        # np.digitize returns a one-indexed value, whereas params is a zero-indexed value
        # This line shifts the index, so that it matches the indexing of params
        bin_index = bin_index - 1

        # Get parameters of matching bin
        params_match = self.params[bin_index]

        return params_match

    def pvalue(self, x, size, invert_cdf=False, strict_bounds=True):
        """Calculate P value of x under model"""

        # Find model parameters for matching bin based on size
        params = self._params_for_size(size, strict_bounds)

        # Calculate probability of finding the observed x, or more extreme, under model
        p = self.distribution.cdf(x, *params)

        if invert_cdf:
            p = 1 - p

        return p

    def model_mean(self, size, strict_bounds=True):
        """Find mean of model for given size"""

        # Find model parameters for matching bin based on size
        params = self._params_for_size(size, strict_bounds)

        # Calculate mean of model
        mean = self.distribution.mean(*params)

        return mean
