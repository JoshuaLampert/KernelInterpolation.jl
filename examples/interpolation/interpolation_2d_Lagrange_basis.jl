using KernelInterpolation
using Plots

# interpolate Franke function
function f(x)
    0.75 * exp(-0.25 * ((9 * x[1] - 2)^2 + (9 * x[2] - 2)^2)) +
    0.75 * exp(-(9 * x[1] + 1)^2 / 49 - (9 * x[2] + 1) / 10) +
    0.5 * exp(-0.25 * ((9 * x[1] - 7)^2 + (9 * x[2] - 3)^2)) -
    0.2 * exp(-(9 * x[1] - 4)^2 - (9 * x[2] - 7)^2)
end

n = 50
nodeset = random_hypercube(n; dim = 2)
values = f.(nodeset)

kernel = ThinPlateSplineKernel{dim(nodeset)}()
# Computing the Lagrange basis is expensive, but interpolation with it is cheap
basis = LagrangeBasis(nodeset, kernel)
itp = interpolate(basis, values)

N = 20
many_nodes = homogeneous_hypercube(N; dim = 2)

p1 = plot(many_nodes, itp)
plot!(p1, many_nodes, f, st = :surface)

p2 = plot(itp, st = :heatmap)
plot(p1, p2, layout = (2, 1))
