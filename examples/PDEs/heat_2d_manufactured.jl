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

kernel = WendlandKernel{2}(3, shape_parameter = 0.3)
sd = Semidiscretization(pde, nodeset_inner, g, nodeset_boundary, u, kernel)
tspan = (0.0, 1.0)
ode = semidiscretize(sd, tspan)
callback = SaveSolutionCallback(dt = 0.01, extra_functions = ((t, x) -> u(t, x, pde),),
                                keys = ["itp", "u"])
sol = solve(ode, Rodas5P(), saveat = 0.01, callback = callback)
titp = TemporalInterpolation(sol)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, titp(last(tspan)))
plot!(many_nodes, x -> u(last(tspan), x, pde))
