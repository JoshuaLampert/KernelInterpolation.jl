using BenchmarkTools
using Random
using OrdinaryDiffEqNonlinearSolve, OrdinaryDiffEqNonlinearSolve
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
SUITE["computing Lagrange basis"] = @benchmarkable LagrangeBasis($nodeset, $kernel)
basis = LagrangeBasis(nodeset, kernel)
SUITE["interpolation 2D Lagrange basis"] = @benchmarkable interpolate($basis, $values)

nodeset = random_hypercube(100; dim = 5)
values = f.(nodeset)

kernel = WendlandKernel{dim(nodeset)}(3, shape_parameter = 0.1)
SUITE["interpolation 5D"] = @benchmarkable interpolate($nodeset, $values, $kernel)

# Least squares benchmarks
centers = random_hypercube(81; dim = 2)
nodeset = random_hypercube(1089; dim = 2)
values = f.(nodeset)

kernel = ThinPlateSplineKernel{dim(nodeset)}()
basis = StandardBasis(centers, kernel)
SUITE["least squares 2D"] = @benchmarkable interpolate($basis, $values, $nodeset)

# stationary PDE benchmarks
f(x, equations) = 5 / 4 * pi^2 * sinpi(x[1]) * cospi(x[2] / 2)
pde = PoissonEquation(f)
u(x, equations) = sinpi(x[1]) * cospi(x[2] / 2)
nodeset_inner = homogeneous_hypercube(10, (0.1, 0.1), (0.9, 0.9); dim = 2)
nodeset_boundary = homogeneous_hypercube_boundary(3; dim = 2)
# Dirichlet boundary condition (here taken from analytical solution)
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
