using KernelInterpolation
using Plots

# analytical solution of equation (manufactured solution)
u(x, equations) = exp(x[1] * x[2])

# coefficients of elliptic equation
A(x) = [x[1] * x[2]+1 sin(x[2])
        sin(x[2]) 2]
b(x) = [x[1]^2, 1]
c(x) = 3 * x[2]^2 * x[1] + 4 * x[2]
# right-hand-side of elliptic equation (computed from analytical solution)
function f(x, equations)
    AA = equations.op.A(x)
    bb = equations.op.b(x)
    cc = equations.op.c(x)
    return (-AA[1, 1] * x[2]^2 - (AA[1, 2] + AA[2, 1]) * (1 + x[1] * x[2]) -
            AA[2, 2] * x[1]^2 + bb[1] * x[2] + bb[2] * x[1] + cc) * u(x, equations)
end
pde = EllipticEquation(A, b, c, f)

n = 10
nodeset_inner = homogeneous_hypercube(n, (0.1, 0.1), (0.9, 0.9); dim = 2)
n_boundary = 3
nodeset_boundary = homogeneous_hypercube_boundary(n_boundary; dim = 2)
# Dirichlet boundary condition (here taken from analytical solution)
g(x) = u(x, pde)

kernel = WendlandKernel{2}(3, shape_parameter = 0.8)
sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, kernel)
itp = solve_stationary(sd)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, itp)
plot!(many_nodes, x -> u(x, pde), label = "analytical solution")
