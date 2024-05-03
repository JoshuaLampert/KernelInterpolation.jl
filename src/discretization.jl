"""
    SpatialDiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary)

Spatial discretization of a partial differential equation with Dirichlet boundary conditions.
The `nodeset_inner` are the nodes in the domain and `nodeset_boundary` are the nodes on the boundary. The `boundary_condition`
is a function describing the Dirichlet boundary conditions.

See also [`Semidiscretization`](@ref), [`solve_stationary`](@ref).
"""
struct SpatialDiscretization{Dim, RealT, Equations, BoundaryCondition,
                             Kernel <: AbstractKernel{Dim}}
    equations::Equations
    nodeset_inner::NodeSet{Dim, RealT}
    boundary_condition::BoundaryCondition
    nodeset_boundary::NodeSet{Dim, RealT}
    kernel::Kernel

    function SpatialDiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary,
                                   kernel = GaussKernel{dim(nodeset_inner)}())
        new{dim(nodeset_inner), eltype(nodeset_inner), typeof(equations), typeof(boundary_condition), typeof(kernel)}(equations,
            nodeset_inner, boundary_condition, nodeset_boundary, kernel)
    end
end

function Base.show(io::IO, sd::SpatialDiscretization)
    print(io,
          "SpatialDiscretization with $(dim(sd)) dimensions, $(length(sd.nodeset_inner)) inner nodes, $(length(sd.nodeset_boundary)) boundary nodes, and kernel $(sd.kernel)")
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
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary, kernel = spatial_discretization
    nodeset = merge(nodeset_inner, nodeset_boundary)

    pd_matrix = pde_matrix(equations, nodeset_inner, nodeset, kernel)
    b_matrix = kernel_matrix(nodeset_boundary, nodeset, kernel)
    system_matrix = [pd_matrix
                     b_matrix]
    b = [rhs(nodeset_inner, equations); boundary_condition.(nodeset_boundary)]
    c = system_matrix \ b

    # Do not support additional polynomial basis for now
    xx = polyvars(Dim)
    ps = monomials(xx, 0:-1)
    return Interpolation(kernel, nodeset, c, system_matrix, ps, xx)
end

"""
    Semidiscretization(spatial_discretization, initial_condition)
    Semidiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary, initial_condition, kernel = GaussKernel{dim(nodeset_inner)}())

Semidiscretization of a partial differential equation with Dirichlet boundary conditions and initial condition `initial_condition`. The `boundary_condition` function
can be time- and space-dependent. The `initial_condition` function is time- and space-dependent to be able to reuse it as analytical solution if available. If no
analytical solution is available, the time variable can be ignored in the `initial_condition` function.

See also [`SpatialDiscretization`](@ref), [`semidiscretize`](@ref).
"""
struct Semidiscretization{InitialCondition, Cache}
    spatial_discretization::SpatialDiscretization
    initial_condition::InitialCondition
    cache::Cache
end

function Semidiscretization(spatial_discretization, initial_condition)
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary, kernel = spatial_discretization
    nodeset = merge(nodeset_inner, nodeset_boundary)
    k_matrix_inner = kernel_matrix(nodeset_inner, nodeset, kernel)
    k_matrix_boundary = kernel_matrix(nodeset_boundary, nodeset, kernel)
    # whole kernel matrix is not needed for rhs, but for initial condition
    k_matrix = [k_matrix_inner
                k_matrix_boundary]
    pd_matrix = pde_matrix(equations, nodeset_inner, nodeset, kernel)
    b_matrix = kernel_matrix(nodeset_boundary, nodeset, kernel)
    pde_boundary_matrix = [pd_matrix
                           b_matrix]
    m_matrix = [k_matrix_inner
                zeros(eltype(k_matrix_inner), size(k_matrix_boundary)...)]
    cache = (; kernel_matrix = k_matrix, mass_matrix = m_matrix,
             pde_boundary_matrix = pde_boundary_matrix)
    return Semidiscretization{typeof(initial_condition), typeof(cache)}(spatial_discretization,
                                                                        initial_condition,
                                                                        cache)
end

function Semidiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary,
                            initial_condition, kernel = GaussKernel{dim(nodeset_inner)}())
    return Semidiscretization(SpatialDiscretization(equations, nodeset_inner,
                                                    boundary_condition, nodeset_boundary,
                                                    kernel), initial_condition)
end

function Base.show(io::IO, semi::Semidiscretization)
    print(io,
          "Semidiscretization with $(dim(semi)) dimensions, $(length(semi.spatial_discretization.nodeset_inner)) inner nodes, $(length(semi.spatial_discretization.nodeset_boundary)) boundary nodes, and kernel $(semi.spatial_discretization.kernel)")
end

dim(semi::Semidiscretization) = dim(semi.spatial_discretization)
Base.eltype(semi::Semidiscretization) = eltype(semi.spatial_discretization)

# right-hand side of the ODE
# M c' = -A c + b
# where M is the (singular) mass matrix
# M = (A_I; 0)∈R^{N x N}, A_I∈R^{N_I x N}
# A_{LB} = (A_L; A_B)∈R^{N x N}, A_L∈R^{N_I x N}, A_B∈R^{N_B x N}
# b = (g; g)∈R^N, f∈R^{N_I}, g∈R^{N_B}

# We can get u from c by
# u = A c
# A = (A_I; A_B)∈R^{N x N}
function rhs!(dc, c, semi, t)
    @unpack pde_boundary_matrix = semi.cache
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary = semi.spatial_discretization
    rhs_vector = [rhs(t, nodeset_inner, equations);
                  boundary_condition.(Ref(t), nodeset_boundary)]
    # dc = -pde_boundary_matrix * c + rhs_vector
    dc[:] = muladd(pde_boundary_matrix, -c, rhs_vector)
    return nothing
end

"""
    semidiscetize(semi::Semidiscretization, tspan)

Wrap a `Semidiscretization` object into an `ODEProblem` object with time span `tspan`.
"""
function semidiscretize(semi::Semidiscretization, tspan)
    nodeset = merge(semi.spatial_discretization.nodeset_inner,
                    semi.spatial_discretization.nodeset_boundary)
    u0 = semi.initial_condition.(Ref(first(tspan)), nodeset,
                                 Ref(semi.spatial_discretization.equations))
    c0 = semi.cache.kernel_matrix \ u0
    iip = true # is-inplace, i.e., we modify a vector when calling rhs!
    f = ODEFunction{iip}(rhs!, mass_matrix = semi.cache.mass_matrix)
    return ODEProblem(f, c0, tspan, semi)
end
