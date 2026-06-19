using BenchmarkTools
using Random
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using LinearAlgebra
using LinearSolve
using KernelInterpolation
Random.seed!(1234)

const SUITE = BenchmarkGroup()

# Interpolation benchmarks
f(x) = sin(sum(x))

nodeset = NodeSet(LinRange(0.0, 2 * pi, 8))
values = f.(nodeset)
kernel = GaussKernel{dim(nodeset)}(shape_parameter = 0.5)

SUITE["interpolation 1D"] = @benchmarkable interpolate($nodeset, $values, $kernel)

nodeset = random_hypercube(50; dim = 2)
values = f.(nodeset)
kernel = ThinPlateSplineKernel{dim(nodeset)}()

SUITE["interpolation 2D"] = @benchmarkable interpolate($nodeset, $values, $kernel)
itp = interpolate(nodeset, values, kernel)
x = [0.5, 0.5]
SUITE["interpolation evaluation 2D"] = @benchmarkable $itp($x)
SUITE["computing Lagrange basis"] = @benchmarkable LagrangeBasis($nodeset, $kernel)
basis = LagrangeBasis(nodeset, kernel)
SUITE["interpolation 2D Lagrange basis"] = @benchmarkable interpolate($basis, $values)

nodeset = random_hypercube(100; dim = 5)
values = f.(nodeset)

kernel = WendlandKernel{dim(nodeset)}(3, shape_parameter = 0.1)
SUITE["interpolation 5D"] = @benchmarkable interpolate($nodeset, $values, $kernel)
SUITE["interpolation 5D cholesky"] = @benchmarkable interpolate($nodeset, $values, $kernel;
                                                                factorization_method = cholesky)
SUITE["interpolation 5D KrylovJL_GMRES"] = @benchmarkable interpolate($nodeset, $values,
                                                                      $kernel;
                                                                      linsolve = KrylovJL_GMRES())

# Least squares benchmarks
centers = random_hypercube(81; dim = 2)
nodeset = random_hypercube(1089; dim = 2)
values = f.(nodeset)

kernel = ThinPlateSplineKernel{dim(nodeset)}()
basis = StandardBasis(centers, kernel)
SUITE["least squares 2D"] = @benchmarkable interpolate($basis, $values, $nodeset)

nodeset1 = NodeSet(LinRange(0.0, 1.0, 5))
nodeset2 = NodeSet(LinRange(0.0, 1.0, 9))
nodesets = [nodeset1, nodeset2]
valuesets = [f.(nodeset1), f.(nodeset2)]
kernels = [WendlandKernel{dim(nodeset1)}(3, shape_parameter = 0.4),
    WendlandKernel{dim(nodeset2)}(3, shape_parameter = 0.8)]
SUITE["multiscale interpolation 1D"] = @benchmarkable multiscale_interpolate($nodesets,
                                                                             $valuesets,
                                                                             $kernels)
mitp = multiscale_interpolate(nodesets, valuesets, kernels)
x = [0.5]
SUITE["multiscale evaluation 1D"] = @benchmarkable $mitp($x)

# stationary PDE benchmarks
f(x, equations) = 5 / 4 * pi^2 * sinpi(x[1]) * cospi(x[2] / 2)
pde = PoissonEquation(f)
u(x, equations) = sinpi(x[1]) * cospi(x[2] / 2)
nodeset_inner = homogeneous_hypercube(10, (0.1, 0.1), (0.9, 0.9); dim = 2)
nodeset_boundary = homogeneous_hypercube_boundary(3; dim = 2)
g(x) = u(x, pde)

kernel = WendlandKernel{2}(3, shape_parameter = 0.3)
sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, kernel)
SUITE["Poisson 2D"] = @benchmarkable solve_stationary($sd)

# time-dependent PDE benchmarks
f(t, x, equations) = 0.0
pde = AdvectionEquation((0.5,), f)
u(t, x, equations) = exp(-100.0 * (x[1] - equations.advection_velocity[1] * t - 0.3)^2)
nodeset_inner = homogeneous_hypercube(20, 0.01, 1.0)
nodeset_boundary = NodeSet([0.0])
g(t, x) = u(t, x, pde)
kernel = WendlandKernel{1}(3, shape_parameter = 1.0)
sd = Semidiscretization(pde, nodeset_inner, g, nodeset_boundary, u, kernel)
tspan = (0.0, 0.01)
ode = semidiscretize(sd, tspan)
sol = solve(ode, Rodas5P())
SUITE["Advection 1D"] = @benchmarkable KernelInterpolation.rhs!($(similar(sol.u[end])),
                                                                $(copy(sol.u[end])), $(sd),
                                                                $(first(tspan)))

f(t, x, equations) = 1.0
pde = HeatEquation(1.0, f)
u(t, x, equations) = 0.0
nodeset_inner = homogeneous_hypercube(10, (0.1, 0.1), (0.9, 0.9); dim = 2)
nodeset_boundary = homogeneous_hypercube_boundary(3; dim = 2)
g(t, x) = 0.0
kernel = WendlandKernel{2}(3, shape_parameter = 0.3)
sd = Semidiscretization(pde, nodeset_inner, g, nodeset_boundary, u, kernel)
tspan = (0.0, 0.01)
ode = semidiscretize(sd, tspan)
sol = solve(ode, Rodas5P())
SUITE["Heat 2D"] = @benchmarkable KernelInterpolation.rhs!($(similar(sol.u[end])),
                                                           $(copy(sol.u[end])), $(sd),
                                                           $(first(tspan)))

# RBF-FD benchmarks
# Local RBF-FD discretization: stencils are precomputed at basis-construction time, so the
# dominant cost is split between building the basis (stencil selection + local weights) and
# assembling/solving the resulting sparse system.
f(x, equations) = 5 / 4 * pi^2 * sinpi(x[1]) * cospi(x[2] / 2)
pde = PoissonEquation(f)
u(x, equations) = sinpi(x[1]) * cospi(x[2] / 2)
nodeset_inner = homogeneous_hypercube(20, (0.1, 0.1), (0.9, 0.9); dim = 2)
nodeset_boundary = homogeneous_hypercube_boundary(10; dim = 2)
g(x) = u(x, pde)
kernel = PolyharmonicSplineKernel{2}(3)
nodeset = merge(nodeset_inner, nodeset_boundary)
stencil_selection = KNearestNeighbors(25)

# Building the RBF-FD basis (stencil selection + per-stencil local weight precomputation).
SUITE["RBF-FD basis Lagrange 2D"] = @benchmarkable RBFFDBasis($nodeset, $kernel,
                                                              $stencil_selection;
                                                              m = order($kernel),
                                                              local_basis = RBFFDLagrangeBasis())
SUITE["RBF-FD basis standard 2D"] = @benchmarkable RBFFDBasis($nodeset, $kernel,
                                                              $stencil_selection;
                                                              m = order($kernel),
                                                              local_basis = RBFFDStandardBasis())

# Assembling and solving the sparse stationary system (square RBF-FD). The two local-basis
# policies produce the same weights but differ in the numerical route, so we benchmark the
# full solve with both: the default Lagrange cardinal functions and the standard basis
# (solving each precomputed local kernel/polynomial system).
sd_lagrange = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, RBFFD(),
                                    kernel; stencil_selection, m = order(kernel),
                                    local_basis = RBFFDLagrangeBasis())
SUITE["RBF-FD Poisson 2D with Lagrange basis"] = @benchmarkable solve_stationary($sd_lagrange)
sd_standard = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, RBFFD(),
                                    kernel; stencil_selection, m = order(kernel),
                                    local_basis = RBFFDStandardBasis())
SUITE["RBF-FD Poisson 2D standard basis"] = @benchmarkable solve_stationary($sd_standard)

itp_lagrange = solve_stationary(sd_lagrange)
itp_standard = solve_stationary(sd_standard)
x = [0.5, 0.5]
SUITE["RBF-FD evaluation Lagrange basis 2D"] = @benchmarkable $itp_lagrange($x)
SUITE["RBF-FD evaluation standard basis 2D"] = @benchmarkable $itp_standard($x)

# Overdetermined (least-squares) RBF-FD: fewer centers than evaluation nodes.
inner_centers = homogeneous_hypercube(10, (0.1, 0.1), (0.9, 0.9); dim = 2)
centers = merge(inner_centers, nodeset_boundary)
basis = RBFFDBasis(centers, kernel, stencil_selection; m = order(kernel))
sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, basis)
SUITE["RBF-FD Poisson 2D least squares"] = @benchmarkable solve_stationary($sd)

# Time-dependent RBF-FD: right-hand side evaluation of the sparse semidiscretization.
f(t, x, equations) = 0.0
pde = AdvectionEquation((0.5, 0.5), f)
u(t, x, equations) = exp(-20.0 * sum((x .- equations.advection_velocity .* t .- 0.3) .^ 2))
nodeset_inner = homogeneous_hypercube(20, 0.01, 1.0; dim = 2)
nodeset_boundary = NodeSet(union([[0.0, y] for y in LinRange(0.0, 1.0, 20)],
                                 [[x, 0.0] for x in LinRange(0.0, 1.0, 20)]))
g(t, x) = u(t, x, pde)
kernel = PolyharmonicSplineKernel{2}(3)
sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, RBFFD(), kernel;
                           stencil_selection = KNearestNeighbors(25), m = order(kernel))
semi = Semidiscretization(sd, u)
tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)
sol = solve(ode, Rodas5P())
SUITE["RBF-FD Advection 2D"] = @benchmarkable KernelInterpolation.rhs!($(similar(sol.u[end])),
                                                                       $(copy(sol.u[end])),
                                                                       $(semi),
                                                                       $(first(tspan)))
