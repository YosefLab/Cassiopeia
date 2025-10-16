# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unrealeased]

### Added

### Changed

- Removed plotting module. Use [Pycea](https://pycea.readthedocs.io/en/latest/index.html) for all plotting needs.
- Removed `IIDExponentialBayesian` and `IIDExponentialBayesian` branch length estimation classes. The latest version of these are part of the [ConvexML](https://github.com/songlab-cal/ConvexML) package.
- Removed  `FitnessEstimator` fitness estimation class. This is now implemented by [pycea.tl.fitness](https://pycea.readthedocs.io/en/latest/generated/pycea.tl.fitness.html#pycea.tl.fitness)
- Reimplemented Robinson-Foulds calculation, allowing functionality for passing in CassiopeiaTree, TreeData, or NetworkX DiGraph objects.
New implementation achieves ~250x faster RF distance computation.

### Fixed

- .c files for cython extensions are now included in the source distribution.
