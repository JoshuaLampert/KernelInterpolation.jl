using KernelInterpolation
using QuasiMonteCarlo: sample, HaltonSample
using LinearAlgebra: norm
using Plots

# right-hand-side of Poisson equation is zero -> Laplace equation
f(x, equations) = 0.0
pde = PoissonEquation(f)

# Create inner nodes of annulus-like domain
function create_annulus_inner(n, inner_radius = 1.0, outer_radius = 2.0)
    nodes_matrix = sample(n, [-outer_radius, -outer_radius], [outer_radius, outer_radius],
                          HaltonSample())
    nodes_halton = NodeSet(nodes_matrix')
    # Remove nodes inside unit circle
    to_delete = []
    for i in eachindex(nodes_halton)
        if norm(nodes_halton[i]) > outer_radius || norm(nodes_halton[i]) < inner_radius
            push!(to_delete, i)
        end
    end
    deleteat!(nodes_halton, to_delete)
    return nodes_halton
end

inner_radius = 1.0
outer_radius = 2.0
nodeset_inner = create_annulus_inner(300, inner_radius, outer_radius)
# boundary nodes are the nodes on circles with radius 1.0 and 2.0
nodes_boundary_outer = random_hypersphere_boundary(80, outer_radius; dim = 2)
nodes_boundary_inner = random_hypersphere_boundary(20, inner_radius; dim = 2)
nodeset_boundary = merge(nodes_boundary_outer, nodes_boundary_inner)
# Dirichlet boundary condition (here 0.0 at inner boundary and sin(x[1]) at outer boundary)
g(x) = isapprox(norm(x), outer_radius) ? cos(5.0 * acos(0.5 * x[1])) : 0.0

kernel = WendlandKernel{2}(3, shape_parameter = 0.3)
sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, kernel)
itp = solve_stationary(sd)

many_nodes_inner = create_annulus_inner(800)
many_nodes_boundary_outer = random_hypersphere_boundary(160, outer_radius; dim = 2)
many_nodes_boundary_inner = random_hypersphere_boundary(40, inner_radius; dim = 2)
many_nodes = merge(many_nodes_inner, many_nodes_boundary_outer, many_nodes_boundary_inner)

OUT = "out"
ispath(OUT) || mkpath(OUT)
vtk_save(joinpath(OUT, "laplace_2d_annulus"), many_nodes, itp)
