using KernelInterpolation
using Plots

# function to interpolate
f(x) = x[1] * x[2]

r = 2.0
center = [-1.0, 2.0]
n = 40
nodeset = random_hypersphere(n, 2, r, center)
values = f.(nodeset)

itp = interpolate(nodeset, values, InverseMultiquadricKernel())

N = 500
many_nodes = random_hypersphere(N, 2, r, center)

plot(many_nodes, itp)
plot!(many_nodes, f, st = :surface)
