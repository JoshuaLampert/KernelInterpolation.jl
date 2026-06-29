# Changelog

KernelInterpolation.jl follows the interpretation of
[semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file
for human readability.

## Changes in the v0.3 lifecycle

#### Added

- Added `fill_distance` function ([#187]).
- Added support for RBF-FD ([#182]).
- Added `differentiation_matrix` to assemble the matrix of a differential operator (sparse for
  `RBFFDBasis`, dense for kernel collocation), and added polynomial augmentation to the collocation
  PDE assembly (`solve_stationary`, `operator_matrix`, `pde_boundary_matrix`) so that conditionally
  positive definite kernels are augmented automatically ([#182]).
- Add multiscale interpolation functionality with the function `multiscale_interpolate` and the type `MultiscaleInterpolation` ([#180]).
- Allow applying differential operators to an `Interpolation` to get a callable object that evaluates the operator at any point ([#179]).
- Added support for methods from `LinearSolve.jl` in `solve_stationary` ([#178]).
- Added support for methods from `LinearSolve.jl` in `interpolate` ([#176]).
- Added a keyword argument `factorization_method` to `interpolate`, `interpolation_matrix`,
  and `least_squares_matrix` to allow for different factorization methods ([#130]).

#### Changed

- Several performance improvements reducing allocations, using threaded loops, using `KDTree`
  for nearest-neighbor searches, and computing the separation distance lazily ([#191]).
- Speed up `LagrangeBasis` construction (and hence also the RBF-FD `RBFFDLagrangeBasis` cache) by
  assembling and factorizing the augmented interpolation matrix once per node set and solving
  for all cardinal functions with a single multiple-right-hand-side solve, instead of
  re-assembling and re-factorizing it for each cardinal function ([#190]).
- Speed up RBF-FD stencil selection (`KNearestNeighbors`, `RadiusSearch`) by using a `KDTree`
  from NearestNeighbors.jl instead of a brute-force `O(N^2)` distance scan, reducing the overall
  neighborhood search to roughly `O(N log N)` ([#189]).
- Define slicing for `NodeSet`s and deprecate `values_along_dim` ([#139]).
- Fix order of `PolyharmonicSplineKernel` to return an integer ([#127]).

## Changes when updating to v0.3 from v0.2.x

#### Added

- General floating point support ([#121]).

#### Changed

- The functions `random_hypersphere` and `random_hypersphere_boundary` not require a `Tuple` for
  the argument `center`. Before, e.g., a `Vector` was allowed ([#121]).
- The element type of `NodeSet`s will now always be converted to a floating point type, i.e., also when
  integer values are passed. This is more consistent for an interpolation framework makes many things easier.
  A similar approach is also used in the Meshes.jl/CoordRefSystems.jl ecosystem ([#121]).

## Changes in the v0.2 lifecycle

#### Added

- Added support for general RNG in `random_*` functions ([#106]).
- Added `LagrangeBasis` ([#103]).

#### Changed

- Use OrdinaryDiffEqRosenbrock.jl instead of OrdinaryDiffEq.jl in the examples and documentation ([#108]).
- Fix seriestype for 1D plots ([#101]).

## Changes when updating to v0.2 from v0.1.x

#### Added

- Added tutorial on noisy data ([#95]).
- Added L2 regularization ([#94]).
- Added least squares approximation ([#93], [#97]).

#### Changed

- Add interface for general bases and add `StandardBasis`. This is breaking for least squares approximations because
  the order of `centers` and `nodeset` needs to be swapped in the `interpolate` function. Alternatively, use the new
  `StandardBasis` ([#100]).

## Changes in the v0.1 lifecycle

#### Added

- Added tutorial to documentation ([#66]).
- Added `PartialDerivativeOperator` ([#65]).
- Added compactly supported Wu kernels ([#64]).
- Added compatibility for `PointSet`s  from Meshes.jl ([#63]).
