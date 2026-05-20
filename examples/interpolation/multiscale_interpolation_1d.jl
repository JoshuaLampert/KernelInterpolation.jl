using KernelInterpolation
using Plots

f(x) = sinpi(2 * x[1]) + 0.25 * cospi(12 * x[1])

nodeset1 = homogeneous_hypercube(5; dim = 1)
nodeset2 = homogeneous_hypercube(9; dim = 1)
nodeset = homogeneous_hypercube(17; dim = 1)
nodesets = [nodeset1, nodeset2, nodeset]
values = f.(nodeset)
valuesets = [f.(nodeset1), f.(nodeset2), values]

kernels = [WendlandKernel{1}(3; shape_parameter = 0.2),
           WendlandKernel{1}(3; shape_parameter = 0.6),
           WendlandKernel{1}(3; shape_parameter = 1.0)]
itp = multiscale_interpolate(nodesets, valuesets, kernels)

many_nodes = homogeneous_hypercube(200; dim = 1)

p1 = plot(many_nodes, itp[1]; training_nodes = false, label = "coarse", color = 1, title = "Partial sums")
plot!(p1, many_nodes, itp[1].(many_nodes) + itp[2].(many_nodes); label = "medium", color = 2)
plot!(p1, many_nodes, itp; label = "fine", color = 3)
plot!(p1, many_nodes, f; label = "true", color = 4)

p2 = plot(many_nodes, itp[1]; label = "coarse", color = 1, title = "Individual scales")
plot!(p2, many_nodes, itp[2]; label = "medium", color = 2)
plot!(p2, many_nodes, itp[3]; label = "fine", color = 3)

plot(p1, p2)
