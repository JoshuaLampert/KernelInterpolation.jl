using KernelInterpolation
using Plots

# function to interpolate
f(x) = (x[1] + x[2]^2 < 0.0 ? 1.0 : 0.1)

x_min = -1.0
x_max = 1.0
n = 80
nodeset = random_hypercube(n, x_min, x_max; dim = 2)
values = f.(nodeset)

kernel = Matern12Kernel{dim(nodeset)}()
trafo(x) = [x[1] + x[2]^2, 0.0]
trafo_kernel = TransformationKernel{2}(kernel, trafo)
itp = interpolate(nodeset, values, trafo_kernel)
itp_base = interpolate(nodeset, values, kernel)

many_nodes = homogeneous_hypercube(20, x_min, x_max; dim = 2)

abs_diff_trafo = abs.(itp.(many_nodes) .- f.(many_nodes))
abs_diff = abs.(itp_base.(many_nodes) .- f.(many_nodes))

l1_error_trafo = sum(abs_diff_trafo)
l1_error = sum(abs_diff)
linf_error_trafo = maximum(abs_diff_trafo)
linf_error = maximum(abs_diff)

@show l1_error_trafo
@show l1_error
@show linf_error_trafo
@show linf_error

plot(layout = (1, 3))
plot!(many_nodes, trafo_kernel, subplot = 1, st = :surface, cbar = false,
      c = cgrad(:grays, rev = true), camera = (0, 90), xguide = "x", yguide = "y")

plot!(many_nodes, itp, subplot = 2)
plot!(many_nodes, f, subplot = 2)

plot!(many_nodes, itp_base, subplot = 3)
plot!(many_nodes, f, subplot = 3)
