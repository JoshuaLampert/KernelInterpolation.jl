using KernelInterpolation
using OrdinaryDiffEq
using LinearAlgebra: norm
using WriteVTK: WriteVTK, paraview_collection

# source term of advection equation
f(t, x, equations) = 0.0
pde = AdvectionEquation([0.5, 0.5, 0.5], f)

# initial condition
u(t, x, equations) = exp(-20.0 * norm(x .- equations.advection_velocity .* t .- [0.3, 0.3, 0.3]) ^ 2)

n = 10
nodeset_inner = homogeneous_hypercube(n, 0.01, 1.0; dim = 3)
# don't provide any boundary condition
nodeset_boundary = empty_nodeset(3, Float64)
g(t, x) = 0.0

kernel = WendlandKernel{3}(3, shape_parameter = 1.0)
sd = Semidiscretization(pde, nodeset_inner, g, nodeset_boundary, u, kernel)
tspan = (0.0, 1.0)
ode = semidiscretize(sd, tspan)

sol = solve(ode, Rosenbrock23(), saveat = 0.01)
titp = TemporalInterpolation(sol)

many_nodes = homogeneous_hypercube(15; dim = 3)
OUT = "out"
ispath(OUT) || mkpath(OUT)
pvd = paraview_collection(joinpath(OUT, "solution"))
for t in sol.t
    KernelInterpolation.add_to_pvd(joinpath(OUT, "advection_3d_basic_$(lpad(round(Int, t * 100), 4, '0'))"), pvd, t, many_nodes,
               titp(t).(many_nodes), u.(Ref(t), many_nodes, Ref(pde)), keys = ["numerical", "analytical"])
end
WriteVTK.vtk_save(pvd)
