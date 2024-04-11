using KernelInterpolation
using Plots

# function to interpolate
f(x) = sinpi(x[1]) + sinpi(x[2])

x_min = -1.0
x_max = 1.0
n = 100
nodeset = random_hypercube(n, x_min, x_max; dim = 2)
values = f.(nodeset)

proj1(x) = [x[1]]
kernel1 = TransformationKernel{2}(Matern12Kernel{1}(), proj1)
proj2(x) = [x[2]]
kernel2 = TransformationKernel{2}(RadialCharacteristicKernel{1}(shape_parameter = 2.0),
                                  proj2)
sum_kernel = SumKernel{2}([kernel1, kernel2])
itp = interpolate(nodeset, values, sum_kernel)

many_nodes = homogeneous_hypercube(20, x_min, x_max; dim = 2)

plot(layout = (1, 2))
plot!(sum_kernel, subplot = 1)

plot!(many_nodes, itp, subplot = 2)
plot!(many_nodes, f, subplot = 2)
