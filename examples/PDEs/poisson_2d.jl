using KernelInterpolation
using Plots

# right-hand-side of Poisson equation
f(x) = 5/4 * pi^2 * sin(pi * x[1]) * cos(pi * x[2] / 2)
pde = PoissonEquation(f)

# analytical solution of equation
u(x) = sin(pi * x[1]) * cos(pi * x[2] / 2)

n = 10
nodeset_inner = homogeneous_hypercube(n, (0.1, 0.1), (0.9, 0.9); dim = 2)
n_boundary = 3
nodeset_boundary = homogeneous_hypercube_boundary(n_boundary; dim = 2)
values = f.(nodeset_inner)
# Dirichlet boundary condition (here taken from analytical solution)
g(x) = u(x)
values_boundary = g.(nodeset_boundary)

kernel = GaussKernel{2}()
itp = solve(pde, nodeset_inner, nodeset_boundary, values_boundary, kernel)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, itp)
plot!(many_nodes, u)
