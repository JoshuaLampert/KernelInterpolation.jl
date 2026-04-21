"""
    RBFFiniteDifferenceDiscretization(equations, nodeset_inner, boundary_condition,
                                      nodeset_boundary, basis)
    RBFFiniteDifferenceDiscretization(equations, nodeset_inner, boundary_condition,
                                      nodeset_boundary, kernel = GaussKernel{dim(nodeset_inner)}();
                                      stencil_selection = KNearestNeighbors(),
                                      m = order(kernel))

RBF-FD spatial discretization of PDEs with Dirichlet boundary conditions.
"""
struct RBFFiniteDifferenceDiscretization{Dim, RealT, Equations, BoundaryCondition,
                                         Basis <: RBFFDBasis}
    equations::Equations
    nodeset_inner::NodeSet{Dim, RealT}
    boundary_condition::BoundaryCondition
    nodeset_boundary::NodeSet{Dim, RealT}
    basis::Basis

    function RBFFiniteDifferenceDiscretization(equations,
                                               nodeset_inner::NodeSet{Dim, RealT},
                                               boundary_condition,
                                               nodeset_boundary::NodeSet{Dim, RealT},
                                               basis::RBFFDBasis{Dim, RealT}) where {
                                                                                    Dim,
                                                                                    RealT
                                                                                    }
        return new{Dim, RealT, typeof(equations), typeof(boundary_condition),
                   typeof(basis)}(equations, nodeset_inner, boundary_condition,
                                  nodeset_boundary, basis)
    end
end

function RBFFiniteDifferenceDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                                           boundary_condition,
                                           nodeset_boundary::NodeSet{Dim, RealT},
                                           kernel = GaussKernel{Dim}();
                                           stencil_selection::AbstractStencilSelection = KNearestNeighbors(),
                                           m::Int = order(kernel)) where {
                                                                                                       Dim,
                                                                                                       RealT
                                                                                                       }
    nodeset = merge(nodeset_inner, nodeset_boundary)
    basis = RBFFDBasis(nodeset, kernel, stencil_selection;
                       m)
    return RBFFiniteDifferenceDiscretization(equations, nodeset_inner,
                                             boundary_condition, nodeset_boundary,
                                             basis)
end

function Base.show(io::IO, sd::RBFFiniteDifferenceDiscretization)
    N_i = length(sd.nodeset_inner)
    N_b = length(sd.nodeset_boundary)
    k = interpolation_kernel(sd.basis)
    print(io,
          "RBFFiniteDifferenceDiscretization with $(dim(sd)) dimensions, $N_i inner nodes, $N_b boundary nodes, and kernel $k")
    return nothing
end

dim(::RBFFiniteDifferenceDiscretization{Dim}) where {Dim} = Dim
Base.eltype(::RBFFiniteDifferenceDiscretization{Dim, RealT}) where {Dim, RealT} = RealT

"""
    solve_stationary(rbf_fd_discretization)

Solve a stationary PDE using RBF-FD and return an interpolation of the nodal solution.
"""
function solve_stationary(sd::RBFFiniteDifferenceDiscretization)
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary, basis = sd

    A = rbf_fd_pde_boundary_matrix(equations, nodeset_inner, nodeset_boundary, basis)
    b = [rhs(nodeset_inner, equations); boundary_condition.(nodeset_boundary)]
    u = A \ b

    return interpolate(merge(nodeset_inner, nodeset_boundary), u,
                       interpolation_kernel(basis))
end

"""
    RBFFDSemidiscretization(spatial_discretization, initial_condition)

Semidiscretization wrapper for time-dependent PDEs using RBF-FD in space.
"""
struct RBFFDSemidiscretization{InitialCondition, Cache}
    spatial_discretization::RBFFiniteDifferenceDiscretization
    initial_condition::InitialCondition
    cache::Cache
end

# Alias the standard API entry point to the RBF-FD semidiscretization type.
Semidiscretization(spatial_discretization::RBFFiniteDifferenceDiscretization,
                   initial_condition) =
    RBFFDSemidiscretization(spatial_discretization, initial_condition)

function RBFFDSemidiscretization(spatial_discretization::RBFFiniteDifferenceDiscretization,
                                 initial_condition)
    @unpack equations, nodeset_inner, nodeset_boundary, basis = spatial_discretization

    pdeb_matrix = rbf_fd_pde_boundary_matrix(equations, nodeset_inner, nodeset_boundary, basis)
    n_inner = length(nodeset_inner)
    n_total = length(merge(nodeset_inner, nodeset_boundary))
    mass_diag = vcat(ones(eltype(nodeset_inner), n_inner),
                     zeros(eltype(nodeset_inner), n_total - n_inner))
    mass_matrix = sparse(1:n_total, 1:n_total, mass_diag, n_total, n_total)

    cache = (; mass_matrix = mass_matrix, pde_boundary_matrix = pdeb_matrix)
    return RBFFDSemidiscretization{typeof(initial_condition), typeof(cache)}(spatial_discretization,
                                                                              initial_condition,
                                                                              cache)
end

function semidiscretize(semi::RBFFDSemidiscretization, tspan)
    nodeset = merge(semi.spatial_discretization.nodeset_inner,
                    semi.spatial_discretization.nodeset_boundary)
    u0 = semi.initial_condition.(Ref(first(tspan)), nodeset,
                                 Ref(semi.spatial_discretization.equations))

    iip = true
    f = ODEFunction{iip}(rhs!, mass_matrix = semi.cache.mass_matrix)
    return ODEProblem(f, u0, tspan, semi)
end

function rhs!(du, u, semi::RBFFDSemidiscretization, t)
    @unpack pde_boundary_matrix = semi.cache
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary = semi.spatial_discretization
    @trixi_timeit timer() "rhs!" begin
        @trixi_timeit timer() "rhs vector" begin
            rhs_vector = [rhs(t, nodeset_inner, equations);
                          boundary_condition.(Ref(t), nodeset_boundary)]
        end
        @trixi_timeit timer() "muladd" du[:]=Base.muladd(pde_boundary_matrix, -u,
                                                          rhs_vector)
    end
    return nothing
end
