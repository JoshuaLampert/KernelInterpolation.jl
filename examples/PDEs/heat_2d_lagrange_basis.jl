using KernelInterpolation
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using Plots

# right-hand-side of heat equation
function f(t, x, equations)
    return exp(t) * (x[1] * (x[1] - 1) * x[2] * (x[2] - 1) -
            2 * equations.diffusivity * (x[1] * (x[1] - 1) + x[2] * (x[2] - 1)))
end
pde = HeatEquation(2.0, f)

# analytical solution
u(t, x, equations) = exp(t) * x[1] * (x[1] - 1) * x[2] * (x[2] - 1)

n = 10
nodeset_inner = homogeneous_hypercube(n, (0.1, 0.1), (0.9, 0.9); dim = 2)
n_boundary = 3
nodeset_boundary = homogeneous_hypercube_boundary(n_boundary; dim = 2)
# Dirichlet boundary condition
g(t, x) = 0.0

centers = merge(nodeset_inner, nodeset_boundary)
kernel = MultiquadricKernel{2}(shape_parameter = 1.5)
basis = LagrangeBasis(centers, kernel)
spatial_discretization = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary,
                                               basis)
semi = Semidiscretization(spatial_discretization, u)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)
sol = solve(ode, Rodas5P(), saveat = 0.01)
titp = TemporalInterpolation(sol)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, titp(last(tspan)))
plot!(many_nodes, x -> u(last(tspan), x, pde), label = "analytical solution")
