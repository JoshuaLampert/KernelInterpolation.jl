using KernelInterpolation
using Plots

# function to interpolate
f(x) = sinpi(2 * x[1])

n = 50
nodeset = random_hypercube(n; dim = 2)
values = f.(nodeset)

kernel = GaussKernel{dim(nodeset)}(shape_parameter = 3.0)
itp = interpolate(nodeset, values, kernel)

many_nodes = homogeneous_hypercube(20; dim = 2)

plot(many_nodes, itp)
plot!(many_nodes, f)
