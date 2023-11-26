using KernelInterpolation
using Plots

# function to interpolate
f(x) = sin(2 * pi * x[1])

n = 30
nodeset = random_hypercube(n, 2)
values = f.(nodeset)

itp = interpolate(nodeset, values)

many_nodes = homogeneous_hypercube(20, 2)

plot(many_nodes, itp)
plot!(many_nodes, f)
