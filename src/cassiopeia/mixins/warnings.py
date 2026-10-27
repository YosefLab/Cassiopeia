class CassiopeiaTreeWarning(UserWarning):
    """A Warning for the CassiopeiaTree class."""

    pass


class PreprocessWarning(UserWarning):
    """Warning emitted during preprocessing workflows.

    Returns:
        PreprocessWarning - Specialized warning for preprocessing alerts.
    """

    pass


class ParameterEstimateWarning(UserWarning):
    """An warning class for the estimation and retrieval of tree parameters."""

    pass


class LeafSubsamplerWarning(UserWarning):
    """Warning specific to leaf subsampling helpers.

    Returns:
        LeafSubsamplerWarning - Specialized warning for subsampling notices.
    """

    pass
