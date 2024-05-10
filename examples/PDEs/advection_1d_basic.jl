using KernelInterpolation
using OrdinaryDiffEq
using Plots

# source term of advection equation
f(t, x, equations) = 0.0
pde = AdvectionEquation((0.5,), f)

# initial condition
u(t, x, equations) = exp(-100.0 * (x[1] - equations.advection_velocity[1] * t - 0.3)^2)

n = 20
nodeset_inner = homogeneous_hypercube(n, 0.01, 1.0)
# only provide boundary condition at left boundary
nodeset_boundary = NodeSet([0.0])
g(t, x) = u(t, x, pde)

kernel = WendlandKernel{1}(3, shape_parameter = 1.0)
sd = Semidiscretization(pde, nodeset_inner, g, nodeset_boundary, u, kernel)
tspan = (0.0, 1.0)
ode = semidiscretize(sd, tspan)

sol = solve(ode, Rosenbrock23(), saveat = 0.01)
titp = TemporalInterpolation(sol)

many_nodes = homogeneous_hypercube(200; dim = 1)

plot()
for t in 0.0:0.2:1.0
    plot!(many_nodes, titp(t), training_nodes = false, linewidth = 2, color = :blue,
          label = t == 0.0 ? "numerical" : "")
    scatter!(many_nodes, u.(Ref(t), many_nodes, Ref(pde)), linewidth = 2,
             linestyle = :dot, color = :red, markersize = 2,
             label = t == 0.0 ? "analytical" : "")
end
plot!()
