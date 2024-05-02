using KernelInterpolation
using OrdinaryDiffEq
using Plots

# right-hand-side of heat equation
f(t, x, equations) = exp(t) * (x[1] * (x[1] - 1) * x[2] * (x[2] - 1) - 2 * equations.diffusivity * (x[1] * (x[1] - 1) + x[2] * (x[2] - 1)))
pde = HeatEquation(2.0, f)

# analytical solution
u(t, x, equations) = exp(t) * x[1] * (x[1] - 1) * x[2] * (x[2] - 1)

n = 10
nodeset_inner = homogeneous_hypercube(n, (0.1, 0.1), (0.9, 0.9); dim = 2)
n_boundary = 3
nodeset_boundary = homogeneous_hypercube_boundary(n_boundary; dim = 2)
# Dirichlet boundary condition
g(t, x) = 0.0

kernel = WendlandKernel{2}(3, shape_parameter = 0.3)
sd = Semidiscretization(pde, nodeset_inner, g, nodeset_boundary, u, kernel)
tspan = (0.0, 1.0)
ode = semidiscretize(sd, tspan)
sol = solve(ode, Rosenbrock23(), saveat = 0.01)
titp = TemporalInterpolation(sol)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, titp(last(tspan)))
plot!(many_nodes, x -> u(last(tspan), x, pde))
