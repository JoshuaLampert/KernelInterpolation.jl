using KernelInterpolation
using Plots

# function to interpolate
f(x) = sinpi(2 * x[1])

n = 30
nodeset = random_hypercube(n, 2)
values = f.(nodeset)

itp = interpolate(nodeset, values)

many_nodes = homogeneous_hypercube(20, 2)

plot(many_nodes, itp)
plot!(many_nodes, f)
