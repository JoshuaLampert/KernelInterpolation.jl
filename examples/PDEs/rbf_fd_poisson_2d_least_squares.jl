using KernelInterpolation
using Plots

# Least-squares RBF-FD: the basis is built on fewer centers than collocation nodes
# (inner + boundary), so the linear system is overdetermined and solved in the
# least-squares sense. This is particularly effective with the Gauss kernel, which
# becomes ill-conditioned in the standard (square) RBF-FD system for large stencils.
# With a coarser center grid (overdetermination ratio ≈ 3), the global overdetermination
# stabilizes the solve and allows using a large stencil for better local accuracy.

# Right-hand side of Poisson equation.
f(x, equations) = 5 / 4 * pi^2 * sinpi(x[1]) * cospi(x[2] / 2)
pde = PoissonEquation(f)

# Analytical solution of equation.
u(x, equations) = sinpi(x[1]) * cospi(x[2] / 2)

# Evaluation nodes: 10×10 inner + 8 boundary = 108 total.
n_inner = 10
nodeset_inner = homogeneous_hypercube(n_inner, (0.1, 0.1), (0.9, 0.9); dim = 2)
n_boundary = 3
nodeset_boundary = homogeneous_hypercube_boundary(n_boundary; dim = 2)
# Dirichlet boundary condition (here taken from analytical solution).
g(x) = u(x, pde)

# Coarser center grid: 5×5 inner centers merged with the same 8 boundary nodes = 33 total.
# Overdetermination ratio: 108 / 33 ≈ 3.3.
n_inner_centers = 5
inner_centers = homogeneous_hypercube(n_inner_centers, (0.1, 0.1), (0.9, 0.9); dim = 2)
centers = merge(inner_centers, nodeset_boundary)

kernel = GaussKernel{2}(shape_parameter = 0.5)
basis = RBFFDBasis(centers, kernel, KNearestNeighbors(25); m = order(kernel))

sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, basis)
itp = solve_stationary(sd)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, itp)
plot!(many_nodes, x -> u(x, pde))
