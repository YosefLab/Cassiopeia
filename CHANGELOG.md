# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unrealeased]

### Added

- Added a ``cassiopeia.utils`` module with helpers for retrieving leaves and roots
  and collapsing unifurcations across multiple tree representations.
- Added pytest coverage for the new utility helpers.

### Changed

- Removed plotting module. Use [Pycea](https://pycea.readthedocs.io/en/latest/index.html) for all plotting needs.
- Removed `IIDExponentialBayesian` and `IIDExponentialBayesian` branch length estimation classes. The latest version of these are part of the [ConvexML](https://github.com/songlab-cal/ConvexML) package.
- Removed  `FitnessEstimator` fitness estimation class. This is now implemented by [pycea.tl.fitness](https://pycea.readthedocs.io/en/latest/generated/pycea.tl.fitness.html#pycea.tl.fitness)
- Updated the Robinson-Foulds implementation to rely on the shared utilities
  while continuing to accept mixed tree types and support string keys via
  ``TreeData`` without changing the public API.
- Updated :meth:`cassiopeia.data.CassiopeiaTree.collapse_unifurcations` to use
  the shared utility backend for consistent topology updates.

### Fixed

- .c files for cython extensions are now included in the source distribution.
