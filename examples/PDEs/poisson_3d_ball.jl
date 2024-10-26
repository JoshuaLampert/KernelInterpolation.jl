using KernelInterpolation
using Meshes: Meshes, Ball, MinDistanceSampling, RegularSampling, sample, boundary

# right-hand-side of Poisson equation
f(x, equations) = 1 / 8 * pi^2 * sinpi(x[1]) * cospi(x[2] / 2) * (5 + 21 * cospi(2 * x[3]))
pde = PoissonEquation(f)

# analytical solution of equation
u(x, equations) = sinpi(x[1]) * cospi(x[2] / 2) * cospi(x[3])^2

geometry = Ball((0.0, 0.0, 0.0), 2.0)
points_inner = sample(geometry, MinDistanceSampling(0.2))
nodeset_inner = NodeSet(collect(points_inner))
points_boundary = sample(boundary(geometry), RegularSampling(15))
nodeset_boundary = NodeSet(collect(points_boundary))
# Dirichlet boundary condition (here taken from analytical solution)
g(x) = u(x, pde)

kernel = WendlandKernel{3}(3, shape_parameter = 0.3)
sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, kernel)
itp = solve_stationary(sd)

many_points = sample(geometry, MinDistanceSampling(0.15))
many_nodes = NodeSet(collect(many_points))
OUT = "out"
ispath(OUT) || mkpath(OUT)
vtk_save(joinpath(OUT, "poisson_3d_ball"), many_nodes,
         itp, x -> u(x, pde), x -> itp(x) - u(x, pde),
         keys = ["solution", "analytical", "error"])
