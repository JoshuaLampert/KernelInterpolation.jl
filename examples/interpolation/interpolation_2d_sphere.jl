using KernelInterpolation
using Plots

# function to interpolate
f(x) = x[1] * x[2]

r = 2.0
center = [-1.0, 2.0]
n = 40
nodeset = random_hypersphere(n, r, center)
nodeset_boundary = random_hypersphere_boundary(20, r, center)
merge!(nodeset, nodeset_boundary)
values = f.(nodeset)

kernel = InverseMultiquadricKernel{dim(nodeset)}()
basis = StandardBasis(nodeset, kernel)
itp = interpolate(basis, values)

N = 500
many_nodes = random_hypersphere(N, r, center)
many_nodes_boundary = random_hypersphere_boundary(100, r, center)
merge!(many_nodes, many_nodes_boundary)

plot(many_nodes, zcolor = itp.(many_nodes))
