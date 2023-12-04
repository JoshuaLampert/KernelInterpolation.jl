using KernelInterpolation
using LinearAlgebra: cond
using Plots

# function to interpolate
f(x) = sin(x[1]) * sin(x[2])

d = 2
x_min = 0.0
x_max = 2 * pi
ns = 5:30

many_nodes = homogeneous_hypercube(40, d, x_min, x_max)
f_many = f.(many_nodes)

p = plot(xguide = "separation distance", yguide = "condition number", xscale = :log10,
         yscale = :log10, legend = :topright)

kernels = [GaussKernel{d}(), MultiquadricKernel{d}(), InverseMultiquadricKernel{d}(),
    ThinPlateSplineKernel{d}(), PolyharmonicSplineKernel{d}(3), WendlandKernel{d}(0),
    WendlandKernel{d}(1), WendlandKernel{d}(2), WendlandKernel{d}(3), Matern12Kernel{d}(),
    Matern32Kernel{d}(), Matern52Kernel{d}(), Matern72Kernel{d}()]
for kernel in kernels
    local separation_distances = []
    local conds = []

    for n in ns
        local nodeset = homogeneous_hypercube(n, d, x_min, x_max)
        push!(separation_distances, separation_distance(nodeset))
        local values = f.(nodeset)

        local itp = interpolate(nodeset, values, kernel)
        push!(conds, cond(system_matrix(itp)))
    end

    plot!(p, separation_distances, conds, label = get_name(kernel))
end
p
