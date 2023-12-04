using KernelInterpolation
using LinearAlgebra: norm

# function to interpolate
f(x) = sum(x)

n = 100
d = 5
nodeset = random_hypercube(n, d)
values = f.(nodeset)

kernel = WendlandKernel{d}(3, shape_parameter = 0.1)
itp = interpolate(nodeset, values, kernel)

many_nodes = homogeneous_hypercube(5, d)
println(norm(f.(many_nodes) - itp.(many_nodes), Inf))
