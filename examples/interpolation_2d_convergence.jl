using KernelInterpolation
using LinearAlgebra: norm
using Plots

# function to interpolate
f(x) = sin(x[1]) * sin(x[2])

d = 2
x_min = 0.0
x_max = 2 * pi
ns = 5:39

many_nodes = homogeneous_hypercube(40, d, x_min, x_max)
f_many = f.(many_nodes)

p = plot(xguide = "fill distance", yguide = "max error", xscale = :log10, yscale = :log10,
         legend = :bottomleft, xaxis = :flip)

kernels = [GaussKernel{d}(), MultiquadricKernel{d}(), InverseMultiquadricKernel{d}(),
    ThinPlateSplineKernel{d}(), PolyharmonicSplineKernel{d}(3), WendlandKernel{d}(0),
    WendlandKernel{d}(1), WendlandKernel{d}(2), WendlandKernel{d}(3), Matern12Kernel{d}(),
    Matern32Kernel{d}(), Matern52Kernel{d}(), Matern72Kernel{d}(), RieszKernel{d}(1.0)]
for kernel in kernels
    local fill_distances = []
    local errors = []

    for n in ns
        local nodeset = homogeneous_hypercube(n, d, x_min, x_max)
        push!(fill_distances, sqrt(d) / (2 * (n - 1)))
        local values = f.(nodeset)

        local itp = interpolate(nodeset, values, kernel)
        error = norm(itp.(many_nodes) - f_many, Inf)
        push!(errors, error)
    end

    plot!(p, fill_distances, errors, label = get_name(kernel), linewidth = 2)
end
p
