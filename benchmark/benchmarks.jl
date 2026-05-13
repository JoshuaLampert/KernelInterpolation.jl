using BenchmarkTools
using KernelInterpolation

const SUITE = BenchmarkGroup()

f(x) = sin(sum(x))

nodeset = NodeSet(LinRange(0.0, 2 * pi, 8))
values = f.(nodeset)
kernel = GaussKernel{dim(nodeset)}(shape_parameter = 0.5)

SUITE["interpolation 1D"] = @benchmarkable interpolate($nodeset, $values, $kernel)
