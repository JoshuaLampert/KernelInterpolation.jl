using KernelInterpolation
using Plots

# simple 1D example demonstrating multiscale interpolation with growing nodesets
f(x) = sinpi(x[1])

x_min = -1.0
x_max = 1.0
coarse_nodeset = random_hypercube(24, x_min, x_max; dim = 1)
fine_nodeset = random_hypercube(80, x_min, x_max; dim = 1)
nodesets = [coarse_nodeset, fine_nodeset]
valuesets = [f.(coarse_nodeset), f.(fine_nodeset)]

kernel1 = GaussKernel{1}(shape_parameter = 0.5)
kernel2 = GaussKernel{1}(shape_parameter = 0.1)

mitp = multiscale_interpolate(nodesets, valuesets, [kernel1, kernel2])

many_nodes = homogeneous_hypercube(200, x_min, x_max; dim = 1)
plot(many_nodes, mitp, label = "multiscale")
plot!(many_nodes, f, label = "true")
