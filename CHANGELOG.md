# Changelog

KernelInterpolation.jl follows the interpretation of
[semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file
for human readability.

## Changes when updating to v0.4 from v0.3.x

#### Changed

- `MultiquadricKernel` and `PolyharmonicSplineKernel` now include the sign `(-1)^m`, where `m` is
  the order of the kernel, such that they are conditionally positive definite of order `m` (see
  Wendland (2004), Corollary 8.18, and Fasshauer (2007), Chapter 8). This changes the sign of
  `phi` and of the kernel coefficients of the `MultiquadricKernel` with odd `ceil(beta)` (e.g.,
  the default `beta = 0.5`) and of the `PolyharmonicSplineKernel` with `k = 1, 4, 5, 8, 9, ...`.
  Interpolants and least squares approximations are the same as before. However, `kernel_norm`
  now returns the native space semi-norm for these kernels instead of `0`, and interpolants with
  an `L2Regularization` change, since the regularization only acts as intended with the sign
  ([#220]).
- `MultiquadricKernel` throws an `ArgumentError` for `beta <= 0` (use `InverseMultiquadricKernel`
  with `-beta` for negative `beta`) and for integer `beta` (the kernel is then a polynomial and
  not conditionally positive definite). `PolyharmonicSplineKernel` throws an `ArgumentError` if
  `k` is not a positive integer ([#220]).
- `kernel_norm` and `kernel_inner_product` project the kernel coefficients onto the vectors
  satisfying the moment conditions of the polynomials (which hold for interpolations up to
  round-off errors and for least squares approximations only approximately). `kernel_norm`
  throws a `DomainError` if the squared native space norm is then negative beyond the level of
  round-off errors, instead of silently returning `0` ([#220]).

- `RieszKernel` now uses its `shape_parameter`, i.e., `phi(r) = -(shape_parameter * r)^beta` as
  documented, instead of ignoring it. Interpolants do not change, since the kernel is only scaled
  ([#221]).
- The order of a `SumKernel` is the maximum instead of the minimum of the orders of its
  components, since a sum of conditionally positive definite kernels is in general only
  conditionally positive definite of the largest order. Hence, interpolations with such a kernel
  are augmented by polynomials of higher order ([#221]).
- `ProductKernel` throws an `ArgumentError` if one of its components is not positive definite
  (i.e., has an order larger than 0), since the product of conditionally positive definite
  kernels is in general not conditionally positive definite ([#221]).
- `RadialCharacteristicKernel` throws an `ArgumentError` for `beta < (Dim + 1) / 2`, where it is
  not guaranteed to be positive definite. Before, its `order` returned `Inf` in this case, and
  `interpolate` failed with an `InexactError` ([#221]).

- `SumKernel` and `ProductKernel` store their component kernels in a `Tuple` instead of a
  `Vector{AbstractKernel}` and have the type of this tuple as an additional type parameter, e.g.,
  `SumKernel{2, Tuple{GaussKernel{2, Float64}, ThinPlateSplineKernel{2}}}`. They can still be
  constructed from a vector. Nested sums (products) are flattened into a single sum (product).
  Evaluating them is now type-stable and does not allocate, which
  makes, e.g., assembling kernel matrices several times faster ([#222]).

- Invalid arguments throw an `ArgumentError`, `DimensionMismatch`, or `BoundsError` instead of an
  `AssertionError`, e.g., invalid kernel parameters, points of different lengths, or a `dim` keyword
  argument that does not match the other arguments. The checks of the kernel parameters are in
  the inner constructors, such that they also apply if the constructor with all type parameters
  is called directly, e.g., `MultiquadricKernel{2, Float64}(1.0, 1.0)` ([#223]).

#### Fixed

- `RadialCharacteristicKernel` was treated as not positive definite for `beta = (Dim + 1) / 2`,
  e.g., for the default `beta = 2` in three dimensions, although it is positive definite there
  ([#221]).
- `kernel_norm` and `kernel_inner_product` returned wrong values for interpolations in the
  `LagrangeBasis`, since they used the coefficients with respect to the cardinal functions as
  kernel coefficients. They now compute the native space (semi-)norm and inner product, which
  agree with the ones of the same interpolation in the standard basis. For interpolations in an
  `RBFFDBasis`, which have no global native space representation, they throw an
  `ArgumentError` instead of returning a meaningless value ([#220]).

#### Removed

- The deprecated function `values_along_dim` is removed. Use `nodeset[:, i]` instead ([#220]).

## Changes in the v0.3 lifecycle

#### Added

- The whole interpolation process is now differentiable with forward-mode automatic differentiation,
  e.g. with ForwardDiff.jl, i.e. not only the evaluation of an `Interpolation`, but also the assembly
  of the system matrix and the linear solve. The interpolated values, the nodes, and the parameters of
  the kernel may have different element types, which allows differentiating `interpolate` and
  `solve_stationary` with respect to any of them ([#200]).
- Added `smoothness(kernel)`, returning the largest `k` such that the multivariate function
  `Phi(x) = phi(||x||)` is `k` times continuously differentiable. It is used to decide whether a
  differential operator of order `m` may be evaluated at the center of a kernel, which requires
  `smoothness(kernel) >= m`. The fallback for user-defined kernels is the conservative value `0`.
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

#### Fixed

- `homogeneous_hypercube` and `homogeneous_hypercube_boundary` did not terminate if the bounds
  were of different types, e.g., `homogeneous_hypercube(3, 0, 1.0; dim = 2)`. The functions
  creating `NodeSet`s (`random_hypercube`, `random_hypercube_boundary`, `homogeneous_hypercube`,
  `homogeneous_hypercube_boundary`, `random_hypersphere`, `random_hypersphere_boundary`) as well
  as `AdvectionEquation` and `AdvectionDiffusionEquation` now accept arguments of different
  types, e.g., integer and floating point bounds, radii, centers, or advection velocities, and
  convert them to a common type ([#218]).
- `multiscale_interpolate` failed if the kernels of the different scales are of different
  types. Constructing a `MultiquadricKernel`, `InverseMultiquadricKernel`,
  `RadialCharacteristicKernel`, `MaternKernel`, or `RieszKernel` with an integer
  `shape_parameter` failed with an `InexactError`; it is now converted to a floating point
  number ([#212]).
- `random_hypersphere_boundary` ignored the `center` in one dimension. `homogeneous_hypercube`
  produced `NaN` coordinates if only one node is used in a dimension; this node is now placed at
  the midpoint between the bounds. The heatmap of a two-dimensional kernel was transposed, which
  is visible for kernels that are not symmetric with respect to swapping the coordinates
  ([#213]).
- Setting a node of a `NodeSet` by an `MVector` did not invalidate the cached separation
  distance, so `separation_distance` could return an outdated value. Setting a node of a
  one-dimensional `NodeSet` by a `Vector` or an `MVector` failed with a method ambiguity ([#211]).
- Second derivatives at the kernel center of the general `MaternKernel` (with `nu > 1`) and of
  `PolyharmonicSplineKernel` with even `k >= 4` returned `NaN`, e.g., for `Laplacian` and
  `EllipticOperator`. Hence, collocation matrices of second-order PDEs with these kernels had
  `NaN` on the diagonal. In addition, `phi(MaternKernel{d}(1.0), 0.0)` returned `NaN` instead
  of `1.0` ([#210]).
- Derivatives of radial-symmetric kernels are now evaluated through the chain rule on the
  scalar radial profile `phi` instead of by differentiating `x -> Phi(kernel, x)` directly.
  Previously, the `save_call` workaround perturbed the argument by `eps` at the kernel center
  to keep automatic differentiation from producing `NaN` at the singularity of `norm`. That
  had three consequences, all of which are fixed:
  - Second-order operators were badly wrong at the center. The radial formula
    `Delta Phi = phi'' + (d - 1) phi' / r` suffers catastrophic cancellation at `r = eps`, so
    every diagonal entry of a `Laplacian` or `EllipticOperator` collocation matrix carried an
    `O(10%)` error. For example, `Laplacian` of `WendlandKernel{2}(3, shape_parameter = 0.4)`
    at the center returned `-5.52` instead of the correct `-7.04`. Solutions of PDE examples
    change accordingly, and are generally more accurate.
  - For kernels that are not differentiable at the origin (`smoothness == 0`, e.g.
    `WendlandKernel` with `k = 0`, `Matern12Kernel`, `RieszKernel` with `beta <= 1`), a
    non-existent derivative was silently returned as a value pointing along the first
    coordinate direction. Such calls now throw an `ArgumentError`.
  - `save_call` mutated its argument, so derivatives at the center errored for immutable
    input vectors such as `SVector`. These now work ([#204]).

#### Changed

- The right-hand side `rhs!` of a `Semidiscretization` no longer allocates. The
  `ThinPlateSplineKernel`, the `SumKernel`, and `kernel_inner_product` keep the floating point
  type of their input, e.g., `Float32`. `vtk_read` no longer silently ignores errors when
  reading the point data ([#218]).
- `fill_distance` uses a `KDTree` instead of the full distance matrix, which makes it much faster
  and avoids the memory of size `length(nodeset) * length(reference)`. Evaluating an RBF-FD
  interpolation, or applying a differential operator to it, finds the nearest center with the
  `KDTree` stored in the `RBFFDBasis` instead of a linear search over all centers. In addition,
  the assembly of sparse RBF-FD matrices (e.g., `kernel_matrix`, `pde_matrix`, and
  `differentiation_matrix` with an `RBFFDBasis`) is type-stable now, which makes it about twice as
  fast ([#217]).
- Applying differential operators and equations to a radial-symmetric kernel at static vectors
  no longer allocates, which speeds up the assembly of collocation matrices. Evaluating an
  interpolation with polynomial augmentation, or applying a differential operator to it, at a
  plain `Vector` allocates less and is faster ([#216]).
- Evaluating a radial-symmetric kernel at static vectors, e.g., at the nodes of a `NodeSet`, no
  longer allocates, which speeds up the evaluation of interpolants and the assembly of kernel
  matrices ([#215]).
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

- The functions `random_hypersphere` and `random_hypersphere_boundary` now require a `Tuple` for
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
