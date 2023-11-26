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
nodeset = random_hypercube(n, 2)
values = f.(nodeset)

kernel = ThinPlateSplineKernel{dim(nodeset)}()
itp = interpolate(nodeset, values, kernel)

many_nodes = homogeneous_hypercube(20, 2)

plot(many_nodes, itp)
plot!(many_nodes, f, st =:surface)
