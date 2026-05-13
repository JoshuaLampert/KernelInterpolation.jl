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

# Least squares benchmarks
centers = random_hypercube(81; dim = 2)
nodeset = random_hypercube(1089; dim = 2)
values = f.(nodeset)

kernel = ThinPlateSplineKernel{dim(nodeset)}()
basis = StandardBasis(centers, kernel)
SUITE["least squares 2D"] = @benchmarkable interpolate($basis, $values, $nodeset)

# stationary PDE benchmarks
EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples", "PDEs")
examples = [joinpath(EXAMPLES_DIR, "poisson_2d_basic.jl"),
    joinpath(EXAMPLES_DIR, "anisotropic_elliptic_2d_basic.jl"),
    joinpath(EXAMPLES_DIR, "poisson_3d_ball.jl")]

for example in examples
    benchname = joinpath(basename(dirname(example)), basename(example)) * " - rhs!:"
    println("Running $benchname...")
    redirect_stdout(devnull) do
        trixi_include(@__MODULE__, example)
        return nothing
    end
    local sd = @invokelatest Main.sd
    SUITE[benchname] = @benchmarkable solve_stationary($sd)
end

# time-dependent PDE benchmarks
examples = []

for example in examples
    benchname = joinpath(basename(dirname(example)), basename(example)) * " - rhs!:"
    println("Running $benchname...")
    redirect_stdout(devnull) do
        trixi_include(@__MODULE__, example, tspan = (0.0, 1e-11))
        return nothing
    end
    local sol = @invokelatest Main.sol
    local ode = @invokelatest Main.ode
    local tspan = @invokelatest Main.tspan
    SUITE[benchname] = @benchmarkable KernelInterpolation.rhs!($(similar(sol.u[end])),
                                                               $(copy(sol.u[end])),
                                                               $(ode), $(first(tspan)))
end
