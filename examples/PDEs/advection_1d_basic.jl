using KernelInterpolation
using OrdinaryDiffEq
using Plots

# source term of advection equation
f(t, x, equations) = 0.0
pde = AdvectionEquation([0.5,], f)

# initial condition
u(t, x, equations) = exp(-100.0 * (x[1] - equations.advection_velocity[1] * t - 0.5) ^ 2)

n = 20
nodeset_inner = homogeneous_hypercube(n, 0.01, 1.0)
# only provide boundary condition at left boundary
nodeset_boundary = NodeSet([0.0])
g(t, x) = 0.0

kernel = WendlandKernel{1}(3, shape_parameter = 1.0)
sd = Semidiscretization(pde, nodeset_inner, g, nodeset_boundary, u, kernel)
tspan = (0.0, 1.0)
ode = semidiscretize(sd, tspan)

sol = solve(ode, Rosenbrock23(), saveat = 0.01)
titp = TemporalInterpolation(sol)

many_nodes = homogeneous_hypercube(1000; dim = 1)

anim = @animate for t in sol.t
    plot(many_nodes, titp(t), label = "numerical", training_nodes = false, linewidth = 2)
    plot!(many_nodes, u.(Ref(t), many_nodes, Ref(pde)), label = "analytical", linewidth = 2, linestyle = :dot)
    plot!(plot_title = "t = $t", ylims = (-0.1, 1.1))
end
OUT = "out"
ispath(OUT) || mkpath(OUT)
gif(anim, joinpath(OUT, "advection_1d_basic.gif"), fps = 20)
