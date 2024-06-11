using KernelInterpolation
using OrdinaryDiffEq
using Plots

# right-hand-side of Heat equation
f(t, x, equations) = 1.0
pde = HeatEquation(1.0, f)

# initial condition
u(t, x, equations) = 0.0

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

callback = SaveSolutionCallback(interval = 10)
sol = solve(ode, Rodas5P(), saveat = 0.01, callback = callback)
titp = TemporalInterpolation(sol)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, titp(last(tspan)))
