using KernelInterpolation
using Plots

# See S. De Marchi, F. Marchetti & E. Perracchione
# Jumping with variably scaled discontinuous kernels (VSDKs) (2019)
function f(x)
    if -1 <= x[1] <= -0.5
        return exp(-x[1])
    elseif -0.5 <= x[1] <= 0.5
        return x[1]^3
    else
        return 1.0
    end
end

x_min = -1.0
x_max = 1.0
n = 79
nodeset = NodeSet(LinRange(x_min, x_max, n))
values = f.(nodeset)

kernel = GaussKernel{dim(nodeset)}()
itp = interpolate(nodeset, values, kernel)

N = 1000
many_nodes = NodeSet(LinRange(x_min, x_max, N))

plot(many_nodes, itp)
plot!(many_nodes, f)
