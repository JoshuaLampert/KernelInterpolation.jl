using KernelInterpolation
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using Plots

# source term of advection equation
f(t, x, equations) = 0.0
pde = AdvectionEquation((0.5, 0.5), f)

# initial condition (also used as analytical solution). Written component-wise (instead of
# with `norm` and a vector literal) so it is allocation-free in the hot boundary-condition
# evaluation `g`.
function u(t, x, equations)
    v = equations.advection_velocity
    return exp(-20.0 * ((x[1] - v[1] * t - 0.3)^2 + (x[2] - v[2] * t - 0.3)^2))
end

n = 15
nodeset_inner = homogeneous_hypercube(n, 0.01, 1.0; dim = 2)
# only provide boundary condition on the inflow boundaries (left and bottom)
nodeset_boundary = NodeSet(union([[0.0, y] for y in LinRange(0.0, 1.0, n)],
                                 [[x, 0.0] for x in LinRange(0.0, 1.0, n)]))
# Capture `pde` in the closure instead of referencing the non-const global, so the boundary
# condition is type-stable and allocation-free when evaluated in the time-stepping loop.
g = let pde = pde
    (t, x) -> u(t, x, pde)
end

kernel = PolyharmonicSplineKernel{2}(3)
local_basis = RBFFDLagrangeBasis()
sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary,
                           RBFFD(), kernel;
                           stencil_selection = KNearestNeighbors(25),
                           m = order(kernel),
                           local_basis = local_basis)
semi = Semidiscretization(sd, u)
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

save_solution = SaveSolutionCallback(dt = 0.01)
alive = AliveCallback(interval = 10)
summary = SummaryCallback()
callbacks = CallbackSet(alive, summary, save_solution)
sol = solve(ode, Rodas5P(), saveat = 0.01, callback = callbacks)
titp = TemporalInterpolation(sol)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, titp(last(tspan)))
plot!(many_nodes, x -> u(last(tspan), x, pde))
