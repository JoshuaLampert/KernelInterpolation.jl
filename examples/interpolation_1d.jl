using KernelInterpolation
using Plots

# function to interpolate
f(x) = sin(x[1])

x_min = 0.0
x_max = 2 * pi
n = 8
nodeset = NodeSet(LinRange(x_min, x_max, n))
values = f.(nodeset)

kernel = GaussKernel{dim(nodeset)}(shape_parameter = 0.5)
itp = interpolate(nodeset, values, kernel)

N = 1000
many_nodes = NodeSet(LinRange(x_min, x_max, N))

plot(many_nodes, itp)
plot!(many_nodes, f)
