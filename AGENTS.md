# AGENTS.md

This file provides guidance to coding agents (e.g. Claude Code, which reads it via the
`@AGENTS.md` import in `CLAUDE.md`) when working with code in this repository.

## What this is

KernelInterpolation.jl is a Julia package for multivariate interpolation in arbitrary dimension using symmetric
(conditionally) positive-definite kernels (focus on radial basis functions). It supports classical scattered-data
interpolation, generalized (Hermite-Birkhoff) meshfree collocation for solving stationary and time-dependent PDEs,
and local RBF finite differences (RBF-FD) for sparse PDE discretization. Minimum Julia version is 1.10.

## Common commands

Tests use [TestItemRunner](https://github.com/julia-actions/TestItemRunner). Each test is an isolated `@testitem`
with `setup=[...]` snippets (`Setup`, `AdditionalImports`, `PDEExamples`, ...) defined in `test/runtests.jl` and
`test/test_util.jl`.

Run individual test items via the persistent `mcp__julia__julia_eval` session (`env_path` = the package root), not `julia --project=test`. Use `TestEnv.activate()` to build the same merged env
(package + test deps) that `Pkg.test` uses, then filter with TestItemRunner. The session stays warm, so later items
skip recompilation:

```julia
using TestEnv
TestEnv.activate() # merges test/Project.toml deps with the package; activates a temp env
using TestItemRunner
cd("test")
@run_package_tests filter = ti -> occursin("rbf_fd_advection_2d_basic.jl", ti.name)
```

The `filter` matches against `ti.name` (the string in `@testitem "..."`), `ti.tags`, and `ti.filename`. Do `TestEnv.activate()`
once per session; afterwards just re-run the `@run_package_tests` line with a new filter. Do not run the whole suite.

Format code (CI enforces SciML style via `.JuliaFormatter.toml` — `yas_style_nesting`, `align_struct_field`,
`always_use_return`):

```sh
julia -e 'using JuliaFormatter; format(".")'
```

### Running examples or development scripts (the `run/` project)

Examples live in `examples/interpolation/` and `examples/PDEs/`. They depend on packages **not** in the main
`Project.toml` (Plots, OrdinaryDiffEq*, Meshes, QuasiMonteCarlo, WriteVTK). Use the dedicated `run/` project,
which `dev`s the local package and has those extras installed, e.g., to run an example already in the `examples/` folder:

```sh
julia --project=run examples/PDEs/rbf_fd_advection_2d_basic.jl
```

Run via the MCP Julia tools with `env_path` pointing at `run/` for an interactive and persistent session.

## Architecture

`src/KernelInterpolation.jl` is the module entry point; the `include` order there reflects dependencies. Key layers:

- **Kernels** (`src/kernels/`): `AbstractKernel` hierarchy. Radial-symmetric kernels (Gauss, Multiquadric,
  Wendland, PolyharmonicSpline, Matern, ...) in `radialsymmetric_kernel.jl`; composite/transformed kernels
  (Product, Sum, Transformation) in `special_kernel.jl`. Each kernel has an `order` used for polynomial
  augmentation of conditionally positive-definite kernels.

- **Nodes** (`src/nodes.jl`): `NodeSet{Dim,RealT}` is the core point-cloud type. Constructors like
  `homogeneous_hypercube`, `random_hypersphere_boundary`, `merge`, `empty_nodeset`. Most APIs related to PDEs take inner +
  boundary `NodeSet`s separately.

- **Bases** (`src/basis.jl`, `src/rbf_fd/`): `AbstractBasis` abstracts how a function is represented over nodes.
  `StandardBasis`/`LagrangeBasis` for global collocation; `RBFFDBasis` (in `src/rbf_fd/`) for local RBF-FD with
  pluggable stencil selection (`KNearestNeighbors`, `RadiusSearch`) and local basis (`RBFFDLagrangeBasis` default,
  `RBFFDStandardBasis`).

- **Differential operators** (`src/differential_operators.jl`): `PartialDerivative`, `Gradient`, `Laplacian`,
  `EllipticOperator` — applied symbolically to kernels (via ForwardDiff / TypedPolynomials) when assembling PDE
  matrices.

- **Equations** (`src/equations.jl`): `PoissonEquation`, `EllipticEquation`, `AdvectionEquation`, `HeatEquation`,
  `AdvectionDiffusionEquation`. Split into `AbstractStationaryEquation` and `AbstractTimeDependentEquation`;
  this distinction drives whether you call `solve_stationary` or `semidiscretize` + an ODE solver, and how tests
  compare results.

- **Discretization** (`src/discretization.jl`): the central PDE workflow.
  - `SpatialDiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary, method, basis)` where
    `method` is `Collocation()` (Kansa global method) or `RBFFD()` (sparse local). Many convenience constructors
    select the method/basis from argument types.
  - `solve_stationary(sd)` → an `Interpolation`.
  - `Semidiscretization(sd, initial_condition)` → `semidiscretize(semi, tspan)` produces an `ODEProblem` with a
    **singular mass matrix** (a DAE), so it needs DAE-capable solvers — examples use `Rodas5P` from
    OrdinaryDiffEqRosenbrock. The `rhs!` is `M c' = -A_LB c + b`. For `RBFFD` the kernel matrix is `I` and the
    mass matrix is a sparse diagonal (1 on inner nodes, 0 on boundary, enforcing BCs algebraically); for
    collocation the mass/kernel matrices are dense basis matrices.

- **Interpolation** (`src/interpolation.jl`): `interpolate`, the callable `Interpolation` result, and
  `TemporalInterpolation` (wraps an `ODESolution`; `titp(t)` gives the `Interpolation` at time `t`).

- **Callbacks** (`src/callbacks_step/`): `AliveCallback`, `SaveSolutionCallback`, `SummaryCallback` — SciML-style
  step callbacks reused from the Trixi ecosystem (`@trixi_timeit`, `TrixiBase`).

- **I/O & visualization** (`src/io.jl`, `src/visualization.jl`): VTK read/write (ParaView) and RecipesBase plot
  recipes for `NodeSet`/`Interpolation`. Meshes.jl integration is a weak-dep extension in `ext/`.

## Examples ↔ tests convention

Tests in `test/test_examples_*.jl` run each example file via `trixi_include` and compare against hard-coded
reference `l2`/`linf` norms. Two consequences when adding or editing an example:

1. **Examples must be overridable.** `@test_include_example(path; kernel=..., local_basis=..., tspan=...)`
   re-runs the example with those top-level assignments replaced (TrixiBase `trixi_include` semantics). Write
   examples as a flat script of `name = value` assignments so any can be overridden from a test. Tests also assume
   variables like `pde`, `nodeset_inner`, `nodeset_boundary`, `g`, `many_nodes`, `tspan`, and (for analytical
   comparison) `u` exist in the example scope — see `test/test_util.jl`.

2. **Reference values must be regenerated when behavior changes.** Run the example in the `run/` project, compute
   the `l2`/`linf` the test expects (for time-dependent PDEs: at `last(tspan)`, comparing `titp(t).(many_nodes)`
   against `u.(Ref(t), many_nodes, Ref(pde))`), and paste the new numbers. `test/test_util.jl` branches on
   `pde_test`, `least_square_test`, etc.
   CI seeds `Random.seed!(1)` (via the `PDEExamples` snippet) for reproducibility.

## CI checks to keep green

Beyond tests on Linux/macOS/Windows (Julia 1.10–1.12) and a Downgrade run: JuliaFormatter (SciML style),
Aqua + ExplicitImports (`test/test_aqua.jl`), JET, SpellCheck, and Documenter. Match existing import style
(explicit `using Pkg: name` imports at the top of `KernelInterpolation.jl`).
