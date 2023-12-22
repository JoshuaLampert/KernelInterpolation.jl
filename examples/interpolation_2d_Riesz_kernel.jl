using KernelInterpolation
using Plots

# interpolate Franke function
function f(x)
    0.75 * exp(-0.25 * ((9 * x[1] - 2)^2 + (9 * x[2] - 2)^2)) +
    0.75 * exp(-(9 * x[1] + 1)^2 / 49 - (9 * x[2] + 1) / 10) +
    0.5 * exp(-0.25 * ((9 * x[1] - 7)^2 + (9 * x[2] - 3)^2)) -
    0.2 * exp(-(9 * x[1] - 4)^2 - (9 * x[2] - 7)^2)
end

n = 300
nodeset = random_hypercube(n; dim = 2)
values = f.(nodeset)

kernel = RieszKernel{dim(nodeset)}(1.0)
itp = interpolate(nodeset, values, kernel)

N = 20
many_nodes = homogeneous_hypercube(N; dim = 2)

p1 = plot(many_nodes, itp)
plot!(p1, many_nodes, f, st = :surface)

x = unique(values_along_dim(many_nodes, 1))
y = unique(values_along_dim(many_nodes, 2))
z = reshape(itp.(many_nodes), (N, N))
p2 = plot(x, y, z, linetype = :contourf)
plot(p1, p2, layout = (2, 1))
