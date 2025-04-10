"""
    SpatialDiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary, basis)
    SpatialDiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary,
                          [centers,] kernel = GaussKernel{dim(nodeset_inner)}())

Spatial discretization of a partial differential equation with Dirichlet boundary conditions.
The `nodeset_inner` are the nodes in the domain and `nodeset_boundary` are the nodes on the boundary. The `boundary_condition`
is a function describing the Dirichlet boundary conditions. The `centers` are the centers of the kernel functions. By default,
`centers` is set to `merge(nodeset_inner, nodeset_boundary)`. Otherwise, a least squares problem is solved.

See also [`Semidiscretization`](@ref), [`solve_stationary`](@ref).
"""
struct SpatialDiscretization{Dim, RealT, Equations, BoundaryCondition,
                             Basis <: AbstractBasis}
    equations::Equations
    nodeset_inner::NodeSet{Dim, RealT}
    boundary_condition::BoundaryCondition
    nodeset_boundary::NodeSet{Dim, RealT}
    basis::Basis

    function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                                   boundary_condition,
                                   nodeset_boundary::NodeSet{Dim, RealT},
                                   basis::AbstractBasis) where {Dim,
                                                                RealT}
        new{Dim, RealT, typeof(equations), typeof(boundary_condition),
            typeof(basis)}(equations, nodeset_inner,
                           boundary_condition, nodeset_boundary,
                           basis)
    end
end

function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                               boundary_condition,
                               nodeset_boundary::NodeSet{Dim, RealT},
                               centers::NodeSet{Dim, RealT},
                               kernel = GaussKernel{Dim}()) where {Dim,
                                                                   RealT}
    SpatialDiscretization(equations, nodeset_inner, boundary_condition,
                          nodeset_boundary, StandardBasis(centers, kernel))
end

function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                               boundary_condition,
                               nodeset_boundary::NodeSet{Dim, RealT},
                               kernel = GaussKernel{Dim}()) where {Dim, RealT
                                                                   }
    return SpatialDiscretization(equations, nodeset_inner, boundary_condition,
                                 nodeset_boundary,
                                 merge(nodeset_inner, nodeset_boundary), kernel)
end

function Base.show(io::IO, sd::SpatialDiscretization)
    N_i = length(sd.nodeset_inner)
    N_b = length(sd.nodeset_boundary)
    k = interpolation_kernel(sd.basis)
    print(io,
          "SpatialDiscretization with $(dim(sd)) dimensions, $N_i inner nodes, $N_b boundary nodes, and kernel $k")
end

dim(::SpatialDiscretization{Dim}) where {Dim} = Dim
Base.eltype(::SpatialDiscretization{Dim, RealT}) where {Dim, RealT} = RealT

"""
    solve_stationary(spatial_discretization)

Solve a stationary partial differential equation discretized as `spatial_discretization` with Dirichlet boundary
conditions by non-symmetric collocation (Kansa method).
Returns an [`Interpolation`](@ref) object.
"""
function solve_stationary(spatial_discretization::SpatialDiscretization{Dim, RealT}) where {
                                                                                            Dim,
                                                                                            RealT
                                                                                            }
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary, basis = spatial_discretization
    @unpack centers, kernel = basis

    system_matrix = pde_boundary_matrix(equations, nodeset_inner, nodeset_boundary, centers,
                                        kernel)
    b = [rhs(nodeset_inner, equations); boundary_condition.(nodeset_boundary)]
    c = system_matrix \ b

    # Do not support additional polynomial basis for now
    xx = polyvars(Dim)
    ps = monomials(xx, 0:-1)
    nodeset = merge(nodeset_inner, nodeset_boundary)
    return Interpolation(basis, nodeset, c, system_matrix,
                         ps, xx)
end

"""
    Semidiscretization(spatial_discretization, initial_condition)
    Semidiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary, [centers,]
                       initial_condition, kernel = GaussKernel{dim(nodeset_inner)}())

Semidiscretization of a partial differential equation with Dirichlet boundary conditions and initial condition `initial_condition`. The `boundary_condition` function
can be time- and space-dependent. The `initial_condition` function is time- and space-dependent to be able to reuse it as analytical solution if available. If no
analytical solution is available, the time variable can be ignored in the `initial_condition` function. The `centers` are the centers of the kernel functions. By default,
`centers` is set to `merge(nodeset_inner, nodeset_boundary)`. Note that `centers` needs to have the center number of nodes as the number of nodes in the domain and on the boundary
because OrdinaryDiffEq.jl does not support DAEs with rectangular mass matrices.

See also [`SpatialDiscretization`](@ref), [`semidiscretize`](@ref).
"""
struct Semidiscretization{InitialCondition, Cache}
    spatial_discretization::SpatialDiscretization
    initial_condition::InitialCondition
    cache::Cache
end

function Semidiscretization(spatial_discretization::SpatialDiscretization,
                            initial_condition)
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary, basis = spatial_discretization
    @unpack centers, kernel = basis
    @assert length(centers)==length(nodeset_inner) + length(nodeset_boundary) "The number of centers must be equal to the number of inner and boundary nodes."
    k_matrix_inner = kernel_matrix(centers, nodeset_inner, kernel)
    k_matrix_boundary = kernel_matrix(centers, nodeset_boundary, kernel)
    # whole kernel matrix is not needed for rhs, but for initial condition
    k_matrix = [k_matrix_inner
                k_matrix_boundary]
    pdeb_matrix = pde_boundary_matrix(equations, nodeset_inner, nodeset_boundary, centers,
                                      kernel)
    m_matrix = [k_matrix_inner
                zeros(eltype(k_matrix_inner), size(k_matrix_boundary)...)]
    cache = (; kernel_matrix = k_matrix, mass_matrix = m_matrix,
             pde_boundary_matrix = pdeb_matrix)
    return Semidiscretization{typeof(initial_condition), typeof(cache)}(spatial_discretization,
                                                                        initial_condition,
                                                                        cache)
end

function Semidiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                            boundary_condition, nodeset_boundary::NodeSet{Dim, RealT},
                            centers::NodeSet{Dim, RealT}, initial_condition,
                            kernel = GaussKernel{Dim}()) where {Dim, RealT}
    return Semidiscretization(SpatialDiscretization(equations, nodeset_inner,
                                                    boundary_condition, nodeset_boundary,
                                                    centers, kernel), initial_condition)
end

function Semidiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                            boundary_condition, nodeset_boundary::NodeSet{Dim, RealT},
                            initial_condition,
                            kernel = GaussKernel{Dim}()) where {Dim, RealT}
    return Semidiscretization(equations, nodeset_inner, boundary_condition,
                              nodeset_boundary, merge(nodeset_inner, nodeset_boundary),
                              initial_condition, kernel)
end

function Base.show(io::IO, semi::Semidiscretization)
    N_i = length(semi.spatial_discretization.nodeset_inner)
    N_b = length(semi.spatial_discretization.nodeset_boundary)
    k = interpolation_kernel(semi.spatial_discretization.basis)
    print(io,
          "Semidiscretization with $(dim(semi)) dimensions, $N_i inner nodes, $N_b boundary nodes, and kernel $k")
end

dim(semi::Semidiscretization) = dim(semi.spatial_discretization)
Base.eltype(semi::Semidiscretization) = eltype(semi.spatial_discretization)

# right-hand side of the ODE (N_I: number of inner nodes, N_B: number of boundary nodes, N = N_I + N_B)
# M c' = -A_{LB} c + b
# where M is the (singular) mass matrix
# M = (A_I; 0)∈R^{N x N}, A_I∈R^{N_I x N}
# A_{LB} = (A_L; A_B)∈R^{N x N}, A_L∈R^{N_I x N}, A_B∈R^{N_B x N}
# b = (f; g)∈R^N, f∈R^{N_I}, g∈R^{N_B}

# We can get u from c by
# u = A c
# A = (A_I; A_B)∈R^{N x N}
function rhs!(dc, c, semi, t)
    @unpack pde_boundary_matrix = semi.cache
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary = semi.spatial_discretization
    @trixi_timeit timer() "rhs!" begin
        @trixi_timeit timer() "rhs vector" begin
            rhs_vector = [rhs(t, nodeset_inner, equations);
                          boundary_condition.(Ref(t), nodeset_boundary)]
        end
        # dc = -pde_boundary_matrix * c + rhs_vector
        @trixi_timeit timer() "muladd" dc[:]=Base.muladd(pde_boundary_matrix, -c,
                                                         rhs_vector)
    end
    return nothing
end

"""
    semidiscetize(semi::Semidiscretization, tspan)

Wrap a [`Semidiscretization`](@ref) object into an `ODEProblem` object with time span `tspan`.
"""
function semidiscretize(semi::Semidiscretization, tspan)
    nodeset = merge(semi.spatial_discretization.nodeset_inner,
                    semi.spatial_discretization.nodeset_boundary)
    u0 = semi.initial_condition.(Ref(first(tspan)), nodeset,
                                 Ref(semi.spatial_discretization.equations))
    c0 = semi.cache.kernel_matrix \ u0
    iip = true # is-inplace, i.e., we modify a vector when calling rhs!
    # TODO: This defines an ODEProblem with a mass matrix, which is singular, i.e. the problem is a DAE.
    # Many ODE solvers do not support DAEs.
    f = ODEFunction{iip}(rhs!, mass_matrix = semi.cache.mass_matrix)
    return ODEProblem(f, c0, tspan, semi)
end
