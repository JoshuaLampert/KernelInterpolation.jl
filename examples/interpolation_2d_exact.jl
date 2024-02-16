using KernelInterpolation
using Plots
using LinearAlgebra

x1 = rand(2)
x2 = rand(2)
# function to interpolate
# here linear combination of kernel to verify it's reconstructed exactly
f(x) = -3*exp(-norm(x .- x1)^2) + 4*exp(-norm(x .- x2)^2)

nodeset = NodeSet([x1, x2])
values = f.(nodeset)

kernel = GaussKernel{dim(nodeset)}()
itp = interpolate(nodeset, values, kernel)

many_nodes = homogeneous_hypercube(20; dim = 2)

# Test if reconstruction is exact and coefficients match
@assert isapprox(itp.c, [-3, 4], atol = 1e-14)
@assert isapprox(norm(itp.(many_nodes) .- f.(many_nodes)), 0.0, atol = 1e-14)

plot(many_nodes, itp)
plot!(many_nodes, f)

