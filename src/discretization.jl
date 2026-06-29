"""
    AbstractSpatialMethod

Abstract type tagging the spatial discretization strategy.
"""
abstract type AbstractSpatialMethod end

"""
    Collocation()

Global collocation strategy (Kansa method).

See also [`AbstractSpatialMethod`](@ref) and [`RBFFD`](@ref).
"""
struct Collocation <: AbstractSpatialMethod end

"""
    RBFFD()

Local radial basis function finite difference strategy.

The `local_basis` keyword of [`SpatialDiscretization`](@ref) (or the `local_basis` field of
[`RBFFDBasis`](@ref)) selects the algorithm used consistently for weight computation, matrix
assembly, and interpolant evaluation:
- `RBFFDLagrangeBasis()` (default): precompute the local cardinal functions and evaluate
  `w_k = 𝓛ℓ_k(x_i)`.
- `RBFFDStandardBasis()`: cache the factorization of each local kernel/polynomial system and
  solve `M w = rhs`, `rhs_k = 𝓛K(x_i, x_k)` (plus polynomial rows).

Both give the same weights up to numerical precision.

See also [`AbstractSpatialMethod`](@ref) and [`Collocation`](@ref).
"""
struct RBFFD <: AbstractSpatialMethod end

"""
    SpatialDiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary, basis)
    SpatialDiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary, method, basis)
    SpatialDiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary,
                          [centers,] kernel = GaussKernel{dim(nodeset_inner)}())
    SpatialDiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary,
                          RBFFD(), kernel = GaussKernel{dim(nodeset_inner)}();
                          stencil_selection, m = order(kernel),
                          local_basis = RBFFDLagrangeBasis())

Spatial discretization of a partial differential equation with Dirichlet boundary conditions.
The `nodeset_inner` are the nodes in the domain and `nodeset_boundary` are the nodes on the boundary. The `boundary_condition`
is a function describing the Dirichlet boundary conditions. The `centers` are the centers of the kernel functions. By default,
`centers` is set to `merge(nodeset_inner, nodeset_boundary)`. Otherwise, a least squares problem is solved.

Uses `method` to select the spatial discretization strategy. If `method` is not given, it is selected based on the type of `basis`.
If `basis` is an [`RBFFDBasis`](@ref), `method` is set to [`RBFFD`](@ref). Otherwise, `method` is set to [`Collocation`](@ref).

See also [`Semidiscretization`](@ref), [`solve_stationary`](@ref).
"""
struct SpatialDiscretization{Dim, RealT, Equations, BoundaryCondition,
                             Method <: AbstractSpatialMethod, Basis}
    equations::Equations
    nodeset_inner::NodeSet{Dim, RealT}
    boundary_condition::BoundaryCondition
    nodeset_boundary::NodeSet{Dim, RealT}
    method::Method
    basis::Basis

    function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                                   boundary_condition,
                                   nodeset_boundary::NodeSet{Dim, RealT},
                                   method::AbstractSpatialMethod,
                                   basis) where {Dim, RealT}
        return new{Dim, RealT, typeof(equations), typeof(boundary_condition),
                   typeof(method), typeof(basis)}(equations, nodeset_inner,
                                                  boundary_condition, nodeset_boundary,
                                                  method, basis)
    end
end

function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                               boundary_condition,
                               nodeset_boundary::NodeSet{Dim, RealT},
                               basis::AbstractBasis) where {Dim, RealT}
    return SpatialDiscretization(equations, nodeset_inner, boundary_condition,
                                 nodeset_boundary, Collocation(), basis)
end

function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                               boundary_condition,
                               nodeset_boundary::NodeSet{Dim, RealT},
                               basis::RBFFDBasis) where {Dim, RealT}
    return SpatialDiscretization(equations, nodeset_inner, boundary_condition,
                                 nodeset_boundary, RBFFD(), basis)
end

function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                               boundary_condition,
                               nodeset_boundary::NodeSet{Dim, RealT},
                               centers::NodeSet{Dim, RealT},
                               kernel::AbstractKernel{Dim} = GaussKernel{Dim}()) where {
                                                                                        Dim,
                                                                                        RealT
                                                                                        }
    return SpatialDiscretization(equations, nodeset_inner, boundary_condition,
                                 nodeset_boundary, Collocation(),
                                 StandardBasis(centers, kernel))
end

function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                               boundary_condition,
                               nodeset_boundary::NodeSet{Dim, RealT},
                               kernel::AbstractKernel{Dim} = GaussKernel{Dim}()) where {
                                                                                        Dim,
                                                                                        RealT
                                                                                        }
    return SpatialDiscretization(equations, nodeset_inner, boundary_condition,
                                 nodeset_boundary,
                                 merge(nodeset_inner, nodeset_boundary), kernel)
end

function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                               boundary_condition,
                               nodeset_boundary::NodeSet{Dim, RealT},
                               ::Collocation,
                               centers::NodeSet{Dim, RealT},
                               kernel::AbstractKernel{Dim} = GaussKernel{Dim}()) where {
                                                                                        Dim,
                                                                                        RealT
                                                                                        }
    return SpatialDiscretization(equations, nodeset_inner, boundary_condition,
                                 nodeset_boundary, Collocation(),
                                 StandardBasis(centers, kernel))
end

function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                               boundary_condition,
                               nodeset_boundary::NodeSet{Dim, RealT},
                               ::Collocation,
                               kernel::AbstractKernel{Dim} = GaussKernel{Dim}()) where {
                                                                                        Dim,
                                                                                        RealT
                                                                                        }
    return SpatialDiscretization(equations, nodeset_inner, boundary_condition,
                                 nodeset_boundary, Collocation(),
                                 merge(nodeset_inner, nodeset_boundary), kernel)
end

function SpatialDiscretization(equations, nodeset_inner::NodeSet{Dim, RealT},
                               boundary_condition,
                               nodeset_boundary::NodeSet{Dim, RealT},
                               ::RBFFD,
                               kernel::AbstractKernel{Dim} = GaussKernel{Dim}();
                               stencil_selection::AbstractStencilSelection,
                               m::Int = order(kernel),
                               local_basis::AbstractRBFFDLocalBasis = RBFFDLagrangeBasis()) where {
                                                                                                   Dim,
                                                                                                   RealT
                                                                                                   }
    nodeset = merge(nodeset_inner, nodeset_boundary)
    basis = RBFFDBasis(nodeset, kernel, stencil_selection; m, local_basis)
    return SpatialDiscretization(equations, nodeset_inner, boundary_condition,
                                 nodeset_boundary, RBFFD(), basis)
end

function Base.show(io::IO, sd::SpatialDiscretization)
    N_i = length(sd.nodeset_inner)
    N_b = length(sd.nodeset_boundary)
    k = interpolation_kernel(sd.basis)
    print(io,
          "SpatialDiscretization ($(nameof(typeof(sd.method)))) with $(dim(sd)) dimensions, $N_i inner nodes, $N_b boundary nodes, and kernel $k")
    return nothing
end

dim(::SpatialDiscretization{Dim}) where {Dim} = Dim
Base.eltype(::SpatialDiscretization{Dim, RealT}) where {Dim, RealT} = RealT

# Polynomials used for the collocation system. Conditionally positive definite kernels are
# augmented up to `order(basis)`; RBF-FD handles polynomials per stencil, so it uses none.
polynomials(basis::AbstractBasis, xx) = monomials(xx, 0:(order(basis) - 1))
polynomials(::RBFFDBasis, xx) = monomials(xx, 0:-1)

"""
    solve_stationary(spatial_discretization; linsolve = nothing)

Solve a stationary partial differential equation discretized as `spatial_discretization` with Dirichlet boundary
conditions by non-symmetric collocation (Kansa method).
Returns an [`Interpolation`](@ref) object.
The `linsolve` keyword argument can be used to specify a linear solver from `LinearSolve.jl` for the linear system.
If `linsolve = nothing`, the default backslash operator is used.

For kernel collocation, the kernel space is augmented with polynomials up to `order(basis)`, so that
conditionally positive definite kernels are augmented automatically, while
strictly positive definite kernels (`order == 0`) use no polynomials. RBF-FD discretizations handle
polynomial augmentation locally via the stencils.

See also [`SpatialDiscretization`](@ref).
"""
function solve_stationary(spatial_discretization::SpatialDiscretization{Dim, RealT};
                          linsolve = nothing) where {Dim, RealT}
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary, basis = spatial_discretization

    xx = polyvars(Dim)
    ps = polynomials(basis, xx)
    q = length(ps)
    system_matrix = pde_boundary_matrix(equations, nodeset_inner, nodeset_boundary, basis,
                                        ps)
    b = [rhs(nodeset_inner, equations); boundary_condition.(nodeset_boundary);
         zeros(RealT, q)]
    c = solve_linear_system(system_matrix, b, linsolve)
    nodeset = merge(nodeset_inner, nodeset_boundary)
    return Interpolation(basis, nodeset, c, system_matrix, ps, xx)
end

"""
    Semidiscretization(spatial_discretization, initial_condition)
    Semidiscretization(equations, nodeset_inner, boundary_condition, nodeset_boundary, [centers,]
                       initial_condition, kernel = GaussKernel{dim(nodeset_inner)}())

Semidiscretization of a partial differential equation with Dirichlet boundary conditions and initial condition `initial_condition`. The `boundary_condition` function
can be time- and space-dependent. The `initial_condition` function is time- and space-dependent to be able to reuse it as analytical solution if available. If no
analytical solution is available, the time variable can be ignored in the `initial_condition` function.
If a basis is passed via [`SpatialDiscretization`](@ref), this basis is used consistently in the semidiscretization.
For convenience constructors with `centers` and `kernel`, a [`StandardBasis`](@ref) is used and by default
`centers` is set to `merge(nodeset_inner, nodeset_boundary)`.
The number of basis functions must be equal to the number of inner and boundary nodes because OrdinaryDiffEq.jl
does not support DAEs with rectangular mass matrices.

If `method` in `spatial_discretization` is [`Collocation`](@ref), the semidiscretization is built by collocating the PDE at the inner nodes and the boundary conditions
at the boundary nodes, leading to a system of ODEs for the coefficients of the kernel interpolant. If `method` is [`RBFFD`](@ref), the semidiscretization is built by
applying the RBF-FD operator matrix to the vector of nodal values, leading to a system of ODEs for the nodal values themselves.

See also [`SpatialDiscretization`](@ref), [`semidiscretize`](@ref).
"""
struct Semidiscretization{InitialCondition, Cache}
    spatial_discretization::SpatialDiscretization
    initial_condition::InitialCondition
    cache::Cache
end

function Semidiscretization(spatial_discretization::SpatialDiscretization{Dim, RealT},
                            initial_condition) where {Dim, RealT}
    @unpack equations, nodeset_inner, boundary_condition, nodeset_boundary, method, basis = spatial_discretization
    nodeset = merge(nodeset_inner, nodeset_boundary)
    pdeb_matrix = pde_boundary_matrix(equations, nodeset_inner, nodeset_boundary, basis)

    cache = create_cache(method, nodeset_inner, nodeset_boundary, nodeset,
                         basis,
                         pdeb_matrix, RealT)

    return Semidiscretization{typeof(initial_condition), typeof(cache)}(spatial_discretization,
                                                                        initial_condition,
                                                                        cache)
end

# Build the method-specific cache (kernel/mass matrices) for the semidiscretization.
# Dispatch on the spatial method so new methods can be supported by adding a method here.
function create_cache(::RBFFD, nodeset_inner, nodeset_boundary, nodeset, basis,
                      pdeb_matrix, ::Type{RealT}) where {RealT}
    n_inner = length(nodeset_inner)
    n_total = length(nodeset)
    mass_diag = zeros(RealT, n_total)
    mass_diag[1:n_inner] .= one(RealT)
    m_matrix = sparse(Diagonal(mass_diag))
    return (; kernel_matrix = I, mass_matrix = m_matrix,
            pde_boundary_matrix = pdeb_matrix)
end

function create_cache(::Collocation, nodeset_inner, nodeset_boundary, nodeset,
                      basis, pdeb_matrix, ::Type{RealT}) where {RealT}
    @assert length(basis)==length(nodeset) "The basis must have the same number of functions as the number of inner and boundary nodes."
    basis_matrix_inner = kernel_matrix(basis, nodeset_inner)
    basis_matrix_boundary = kernel_matrix(basis, nodeset_boundary)
    # whole basis matrix is not needed for rhs, but for initial condition
    basis_matrix = [basis_matrix_inner
                    basis_matrix_boundary]
    m_matrix = [basis_matrix_inner
                zeros(eltype(basis_matrix_inner), size(basis_matrix_boundary)...)]
    return (; kernel_matrix = basis_matrix, mass_matrix = m_matrix,
            pde_boundary_matrix = pdeb_matrix)
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
    return nothing
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
        # Seed `dc` with the right-hand side b = (f; g) (source term at the inner nodes,
        # boundary values at the boundary nodes). Writing into `dc` directly keeps `rhs!`
        # allocation-free and works when a solver evaluates it with `ForwardDiff.Dual` numbers
        # since `dc` already has the matching element type.
        @trixi_timeit timer() "rhs vector" begin
            fill_rhs_vector!(dc, t, nodeset_inner, boundary_condition, nodeset_boundary,
                             equations)
        end
        # dc = -pde_boundary_matrix * c + dc, computed in place via a 5-argument `mul!`.
        @trixi_timeit timer() "mul!" mul!(dc, pde_boundary_matrix, c, -1, true)
    end
    return nothing
end

# In-place assembly of the ODE right-hand side `b = (f; g)` (source term `f` at the inner
# nodes, boundary values `g` at the boundary nodes) into `out`. This is the allocation-free
# counterpart of `[rhs(t, nodeset_inner, equations); g.(...)]` used in the hot `rhs!`
# time-stepping loop; `out` (the solver's `dc`) carries the appropriate element type.
function fill_rhs_vector!(out, t, nodeset_inner, boundary_condition, nodeset_boundary,
                          equations)
    n_inner = length(nodeset_inner)
    f = equations.f
    if f isa AbstractVector
        @views out[1:n_inner] .= f
    else
        for i in 1:n_inner
            out[i] = f(t, nodeset_inner[i], equations)
        end
    end
    for j in eachindex(nodeset_boundary)
        out[n_inner + j] = boundary_condition(t, nodeset_boundary[j])
    end
    return out
end

"""
    semidiscetize(semi::Semidiscretization, tspan)

Wrap a [`Semidiscretization`](@ref) object into an `ODEProblem` object from SciMLBase.jl with time span `tspan`.
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
