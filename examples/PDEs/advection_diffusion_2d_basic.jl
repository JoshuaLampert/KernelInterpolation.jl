using KernelInterpolation
using OrdinaryDiffEq
using LinearAlgebra: norm
using Plots

# right-hand-side of advection diffusion equation
f(t, x, equations) = 0.0
pde = AdvectionDiffusionEquation(0.01, (0.2, 0.7), f)

# initial condition
u(t, x, equations) = exp(-100.0 * norm(x - [0.4, 0.5])^2)

n = 15
nodeset_inner = homogeneous_hypercube(n, (0.1, 0.1), (0.9, 1.9); dim = 2)
n_boundary = 5
nodeset_boundary = homogeneous_hypercube_boundary(n_boundary, (0.0, 0.0), (1.0, 2.0); dim = 2)
# Dirichlet boundary condition
g(t, x) = 0.0

kernel = WendlandKernel{2}(3, shape_parameter = 0.3)
sd = Semidiscretization(pde, nodeset_inner, g, nodeset_boundary, u, kernel)
tspan = (0.0, 1.0)
ode = semidiscretize(sd, tspan)

callback = SaveSolutionCallback(dt = 0.01)
sol = solve(ode, Rosenbrock23(), saveat = 0.01, callback = callback)
titp = TemporalInterpolation(sol)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, titp(last(tspan)))
