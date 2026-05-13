using BenchmarkTools
using Random
using KernelInterpolation
Random.seed!(1234)

const SUITE = BenchmarkGroup()

# Interpolation benchmarks
f(x) = sin(sum(x))

nodeset = NodeSet(LinRange(0.0, 2 * pi, 8))
values = f.(nodeset)
kernel = GaussKernel{dim(nodeset)}(shape_parameter = 0.5)

SUITE["interpolation 1D"] = @benchmarkable interpolate($nodeset, $values, $kernel)

nodeset = random_hypercube(50; dim = 2)
values = f.(nodeset)
kernel = ThinPlateSplineKernel{dim(nodeset)}()

SUITE["interpolation 2D"] = @benchmarkable interpolate($nodeset, $values, $kernel)
SUITE["computing Lagrange basis"] = @benchmarkable LagrangeBasis($nodeset, $kernel)
basis = LagrangeBasis(nodeset, kernel)
SUITE["interpolation 2D Lagrange basis"] = @benchmarkable interpolate($basis, $values)

nodeset = random_hypercube(100; dim = 5)
values = f.(nodeset)

kernel = WendlandKernel{dim(nodeset)}(3, shape_parameter = 0.1)
SUITE["interpolation 5D"] = @benchmarkable interpolate($nodeset, $values, $kernel)

# PDE benchmarks
