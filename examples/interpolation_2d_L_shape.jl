using KernelInterpolation
using Plots

# function to interpolate
f(x) = cos(x[1]) * x[2]

x_min1 = (0.0, 0.0)
x_max1 = (2 * pi, 1.0)
x_min2 = (0.0, 1.0)
x_max2 = (1 * pi, 2.0)
nodeset1 = homogeneous_hypercube(5, 2, x_min1, x_max1)
nodeset2 = homogeneous_hypercube(5, 2, x_min2, x_max2)
nodeset = merge(nodeset1, nodeset2)
unique!(nodeset)
values = f.(nodeset)

kernel = RadialCharacteristicKernel{dim(nodeset)}(2.0, shape_parameter = 0.5)
itp = interpolate(nodeset, values, kernel)

N = 500
many_nodes1 = random_hypercube(N, 2, x_min1, x_max1)
many_nodes2 = random_hypercube(N, 2, x_min2, x_max2)
many_nodes = merge(many_nodes1, many_nodes2)

plot(many_nodes, itp)
plot!(many_nodes, f)
