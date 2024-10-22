using KernelInterpolation
using Plots

# interpolate Franke function
function f(x)
    0.75 * exp(-0.25 * ((9 * x[1] - 2)^2 + (9 * x[2] - 2)^2)) +
    0.75 * exp(-(9 * x[1] + 1)^2 / 49 - (9 * x[2] + 1) / 10) +
    0.5 * exp(-0.25 * ((9 * x[1] - 7)^2 + (9 * x[2] - 3)^2)) -
    0.2 * exp(-(9 * x[1] - 4)^2 - (9 * x[2] - 7)^2)
end

n = 1089
nodeset = random_hypercube(n; dim = 2)
values = f.(nodeset) .+ 0.03 * randn(n)

kernel = ThinPlateSplineKernel{dim(nodeset)}()
itp_reg = interpolate(nodeset, values, kernel, reg = L2Regularization(1e-2))
itp = interpolate(nodeset, values, kernel)

N = 40
many_nodes = homogeneous_hypercube(N; dim = 2)

p1 = plot(many_nodes, itp_reg, st = :surface, training_nodes = false)
p2 = plot(many_nodes, itp, st = :surface, training_nodes = false)

plot(p1, p2, layout = (2, 1))
