using KernelInterpolation
using Plots

# right-hand-side of Poisson equation
f(x, equations) = 5 / 4 * pi^2 * sinpi(x[1]) * cospi(x[2] / 2)
pde = PoissonEquation(f)

# analytical solution of equation
u(x, equations) = sinpi(x[1]) * cospi(x[2] / 2)

n = 10
nodeset_inner = homogeneous_hypercube(n, (0.1, 0.1), (0.9, 0.9); dim = 2)
n_boundary = 3
nodeset_boundary = homogeneous_hypercube_boundary(n_boundary; dim = 2)
# Dirichlet boundary condition (here taken from analytical solution)
g(x) = u(x, pde)

kernel = WendlandKernel{2}(3, shape_parameter = 0.3)
sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, kernel)
itp = solve_stationary(sd)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, itp)
plot!(many_nodes, x -> u(x, pde))
