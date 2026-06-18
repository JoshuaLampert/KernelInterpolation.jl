@doc raw"""
    Interpolation

Interpolation object that can be evaluated at a node and represents a kernel interpolation of the form
```math
    s(x) = \sum_{j = 1}^N c_jb_j(x) + \sum_{k = 1}^Q d_kp_k(x),
```
where ``b_j`` are the basis functions and ``p_k`` is a basis of the `Q`-dimensional space of multivariate
polynomials of order [`order`](@ref). The additional conditions
```math
    \sum_{j = 1}^N c_jp_k(x_j) = 0, \quad k = 1,\ldots, Q
```
are enforced.

See also [`interpolate`](@ref).
"""
struct Interpolation{Basis, Dim, RealT, A, Monomials, PolyVars} <:
       AbstractInterpolation{Basis, Dim, RealT}
    basis::Basis
    nodeset::NodeSet{Dim, RealT}
    c::Vector{RealT}
    system_matrix::A
    ps::Monomials
    xx::PolyVars
end

const LagrangeInterpolation = Interpolation{<:LagrangeBasis}

@doc raw"""
    RBFFDInterpolation

Type alias for [`Interpolation{<:RBFFDBasis}`](@ref).

The coefficient vector `c` holds **nodal values** ``u(x_i)`` for all nodes
``x_1,\ldots,x_N`` (inner nodes first, then boundary nodes), ordered consistently
with `merge(nodeset_inner, nodeset_boundary)`.

Evaluation at a point `x` uses the local stencil ``S(j)`` of the nearest center
``x_j``:
```math
u_h(x) = \sum_{k \in S(j)} c_k \, w_k(x;\, S(j)),
```
where ``w_k(x)`` are the local cardinal weights mapping the nodal values to the value of
the local interpolant at `x` (i.e. [`local_weights(@ref)(basis, j, x, Identity())`]). This formula
holds regardless of the weight computation algorithm: for [`RBFFDLagrangeBasis`](@ref) the
``w_k(x) = \ell_k(x)`` are the precomputed cardinal functions, while for
[`RBFFDStandardBasis`](@ref) they are obtained by solving the cached local system with the
evaluation right-hand side. Both give the same ``w_k`` mathematically.
"""
const RBFFDInterpolation = Interpolation{<:RBFFDBasis}

function Base.show(io::IO, itp::Interpolation)
    return print(io,
                 "Interpolation with $(length(nodeset(itp))) nodes, kernel $(interpolation_kernel(itp)) and polynomial of order $(order(itp)).")
end

"""
    dim(itp::Interpolation)

Return the dimension of the input variables of the [`Interpolation`](@ref).
"""
dim(::AbstractInterpolation{Basis, Dim}) where {Basis, Dim} = Dim

"""
    basis(itp)

Return the basis from an [`Interpolation`](@ref) object.
"""
basis(itp::AbstractInterpolation) = itp.basis

"""
    interpolation_kernel(itp)

Return the kernel from an [`Interpolation`](@ref) object.
"""
interpolation_kernel(itp::AbstractInterpolation) = interpolation_kernel(basis(itp))

"""
    nodeset(itp)

Return the node set from an [`Interpolation`](@ref) object.
"""
nodeset(itp::AbstractInterpolation) = itp.nodeset

"""
    centers(itp)

Return the centers from the basis of an [`Interpolation`](@ref) object.
"""
centers(itp::AbstractInterpolation) = centers(basis(itp))

"""
    coefficients(itp::Interpolation)

Obtain all the coefficients of the linear combination for the [`Interpolation`](@ref), i.e., both
the coefficients for the kernel part and for the polynomial part.

See also [`kernel_coefficients`](@ref) and [`polynomial_coefficients`](@ref).
"""
coefficients(itp::Interpolation) = itp.c

"""
    kernel_coefficients(itp::Interpolation)

Obtain the coefficients of the kernel part of the linear combination for the
[`Interpolation`](@ref).

See also [`coefficients`](@ref) and [`polynomial_coefficients`](@ref).
"""
@views kernel_coefficients(itp::Interpolation) = itp.c[eachindex(centers(itp))]

"""
    polynomial_coefficients(itp::Interpolation)

Obtain the coefficients of the polynomial part of the linear combination for the
[`Interpolation`](@ref).

See also [`coefficients`](@ref) and [`kernel_coefficients`](@ref).
"""
@views polynomial_coefficients(itp::Interpolation) = itp.c[(length(centers(itp)) + 1):end]

"""
    polynomial_basis(itp::Interpolation)

Return a vector of the polynomial basis functions used for the [`Interpolation`](@ref).

See also [`polyvars`](@ref).
"""
polynomial_basis(itp::Interpolation) = itp.ps

"""
    polyvars(itp::Interpolation)

Return a vector of the polynomial variables for the [`Interpolation`](@ref).

See also [`polynomial_basis`](@ref).
"""
polyvars(itp::Interpolation) = itp.xx

"""
    order(itp)

Return the order ``m`` of the polynomial used for the [`Interpolation`](@ref), i.e.,
the polynomial degree plus 1. If ``m = 0``, no polynomial is added.
"""
order(itp::AbstractInterpolation) = order(basis(itp))
order(itp::Interpolation) = maximum(degree.(itp.ps), init = -1) + 1

@doc raw"""
    system_matrix(itp::Interpolation)

Return the system matrix, i.e., the matrix ``A`` in the linear system
```math
    Ac = f,
```
where ``c`` are the coefficients of the [`Interpolation`](@ref) and ``f`` the vector
of known values. The exact form of ``A`` differs depending on which method is used.
"""
system_matrix(itp::Interpolation) = itp.system_matrix

@doc raw"""
    interpolate(basis, values, nodeset = centers(basis); m = order(basis),
                regularization = NoRegularization(), factorization_method = nothing,
                linsolve = nothing)
    interpolate(centers, [nodeset,] values, kernel = GaussKernel{dim(nodeset)}();
                m = order(kernel), regularization = NoRegularization(),
                factorization_method = nothing, linsolve = nothing)

Interpolate the `values` evaluated at the nodes in the `nodeset` to a function using the kernel `kernel`
and polynomials up to a order `m` (i.e. degree - 1), i.e., determine the coefficients ``c_j`` and ``d_k`` in the expansion
```math
    s(x) = \sum_{j = 1}^N c_jb_j(x) + \sum_{k = 1}^Q d_kp_k(x),
```
where ``b_j`` are the basis functions in the `basis` and ``s(x)`` the interpolant ``s(x_j) = f(x_j)``, where ``f(x_j)``
are given by `values`, ``x_j`` are the nodes in the `nodeset`, and ``p_k`` is a basis of the ``Q``-dimensional
space of multivariate polynomials with maximum degree of `m - 1`. If `m = 0`, no polynomial is added.
The additional conditions
```math
    \sum_{j = 1}^N c_jp_k(x_j) = 0, \quad k = 1,\ldots, Q = \begin{pmatrix}m - 1 + d\\d\end{pmatrix}
```
are enforced. Returns an [`Interpolation`](@ref) object.

If `nodeset` is provided, the interpolant is a least squares approximation with a different set of nodes as the centers
used for the basis.
Otherwise, `nodeset` is set to `centers(basis)` or `centers`.

A regularization can be applied to the kernel matrix using the `regularization` argument, cf. [`regularize!`](@ref).
In addition, the `factorization_method` can be specified to determine how the system matrix is factorized. By default,
the system matrix is just wrapped as a `Symmetric` matrix for interpolation and no factorization is applied
for a least squares solution, but you can, e.g., also explicitly use `cholesky`, `lu`, or `qr` factorization.
If `linsolve` is provided, the linear system is solved with LinearSolve.jl and any LinearSolve.jl algorithm can
be passed there. If `linsolve` is not provided, the linear system is solved with the backslash operator, which will
automatically use the factorization if `factorization_method` is provided.
"""
function interpolate(basis::AbstractBasis, values::Vector{RealT},
                     nodeset::NodeSet{Dim, RealT} = centers(basis);
                     m = order(basis),
                     regularization = NoRegularization(),
                     factorization_method = nothing,
                     linsolve = nothing) where {Dim, RealT}
    @assert dim(basis) == Dim
    n = length(nodeset)
    @assert length(values) == n
    xx = polyvars(Val(Dim))
    ps = monomials(xx, 0:(m - 1))
    q = length(ps)

    if nodeset == centers(basis)
        if isnothing(linsolve)
            factorization_method = isnothing(factorization_method) ? Symmetric :
                                   factorization_method
            system_matrix = interpolation_matrix(basis, ps, regularization;
                                                 factorization_method)
        else
            system_matrix = interpolation_matrix(basis, ps, regularization;
                                                 factorization_method = Matrix)
        end
    else
        if isnothing(linsolve)
            factorization_method = isnothing(factorization_method) ? Matrix :
                                   factorization_method
            system_matrix = least_squares_matrix(basis, nodeset, ps, regularization;
                                                 factorization_method)
        else
            system_matrix = least_squares_matrix(basis, nodeset, ps, regularization;
                                                 factorization_method = Matrix)
        end
    end
    b = [values; zeros(RealT, q)]
    c = solve_linear_system(system_matrix, b, linsolve)
    return Interpolation{typeof(basis), dim(basis), eltype(nodeset), typeof(system_matrix),
                         typeof(ps), typeof(xx)}(basis, nodeset, c, system_matrix, ps, xx)
end

solve_linear_system(system_matrix, b, ::Nothing) = system_matrix \ b

function solve_linear_system(system_matrix, b, linsolve)
    linear_problem = SciMLBase.LinearProblem(system_matrix, b)
    return SciMLBase.solve(linear_problem, linsolve).u
end
function interpolate(centers::NodeSet{Dim, RealT}, nodeset::NodeSet{Dim, RealT},
                     values::AbstractVector{RealT}, kernel; kwargs...) where {Dim, RealT}
    return interpolate(StandardBasis(centers, kernel), values, nodeset; kwargs...)
end

function interpolate(centers::NodeSet{Dim, RealT},
                     values::AbstractVector{RealT},
                     kernel = GaussKernel{Dim}(; shape_parameter = RealT(1.0));
                     kwargs...) where {Dim, RealT}
    return interpolate(StandardBasis(centers, kernel), values; kwargs...)
end

# Evaluate interpolant
function (itp::Interpolation)(x)
    bas = basis(itp)
    c = kernel_coefficients(itp)
    s = zero(eltype(x))
    for j in eachindex(c)
        s += c[j] * bas[j](x)
    end

    # This works also for the `LagrangeBasis` because it does not have additional polynomials, i.e., `d` is empty
    d = polynomial_coefficients(itp)
    ps = polynomial_basis(itp)
    xx = polyvars(itp)
    for k in eachindex(d)
        s += d[k] * ps[k](xx => x)
    end
    return s
end

# Allow scalar input if interpolant is one-dimensional
function (itp::Interpolation)(x::RealT) where {RealT <: Real}
    @assert dim(itp) == 1
    return itp([x])
end

# RBF-FD evaluation: use the local stencil at center j
function (itp::RBFFDInterpolation)(x, j::Integer)
    bas = basis(itp)
    c = coefficients(itp)
    indices = bas.stencil_indices[j]
    w = local_weights(bas, j, x, Identity())
    s = zero(eltype(c))
    for k in eachindex(indices)
        s += c[indices[k]] * w[k]
    end
    return s
end

function (itp::RBFFDInterpolation)(x::RealT, j::Integer) where {RealT <: Real}
    @assert dim(itp) == 1
    return itp([x], j)
end

# Default 1-arg evaluation: use the stencil of the nearest center
function (itp::RBFFDInterpolation)(x)
    return itp(x, nearest_node_index(x, centers(itp)))
end

# To fix a method ambiguity
function (itp::RBFFDInterpolation)(x::RealT) where {RealT <: Real}
    @assert dim(itp) == 1
    return itp([x])
end

function (diff_op_or_pde::DifferentialOperatorOrEquation)(s, itp::Interpolation, x)
    kernel = interpolation_kernel(itp)
    xis = centers(itp)
    c = kernel_coefficients(itp)
    for j in eachindex(c)
        s += c[j] * diff_op_or_pde(kernel, x, xis[j])
    end

    d = polynomial_coefficients(itp)
    ps = polynomial_basis(itp)
    for k in eachindex(d)
        s += d[k] * diff_op_or_pde(ps[k], x)
    end
    return s
end

function (diff_op_or_pde::DifferentialOperatorOrEquation)(s, itp::LagrangeInterpolation, x)
    c = kernel_coefficients(itp)
    bas = basis(itp)
    for j in eachindex(c)
        s += c[j] * diff_op_or_pde(bas[j], x)
    end
    return s
end

# RBF-FD: the coefficients are nodal values and the local interpolant on the stencil of the
# nearest center represents the solution (see the evaluation above). Applying the operator
# therefore reduces to the local weights `local_weights(basis, j, x, op)` for that operator,
# dotted with the stencil's nodal values, not to a global kernel expansion.
function (diff_op_or_pde::DifferentialOperatorOrEquation)(s, itp::RBFFDInterpolation, x, j)
    bas = basis(itp)
    c = coefficients(itp)
    indices = bas.stencil_indices[j]
    w = local_weights(bas, j, x, diff_op_or_pde)
    if w isa AbstractVector
        for k in eachindex(indices)
            s += c[indices[k]] * w[k]
        end
    else
        for k in eachindex(indices)
            s = s .+ c[indices[k]] .* @view(w[k, :])
        end
    end
    return s
end

function (diff_op_or_pde::DifferentialOperatorOrEquation)(s, itp::RBFFDInterpolation, x)
    return diff_op_or_pde(s, itp, x, nearest_node_index(x, centers(itp)))
end

function (diff_op_or_pde::DifferentialOperatorOrEquation)(itp::Interpolation, x)
    return diff_op_or_pde(zero(eltype(x)), itp, x)
end

function (g::Gradient)(itp::Interpolation, x)
    return g(zero(x), itp, x)
end

function (diff_op_or_pde::DifferentialOperatorOrEquation)(itp::RBFFDInterpolation, x, j)
    return diff_op_or_pde(zero(eltype(x)), itp, x, j)
end

function (g::Gradient)(itp::RBFFDInterpolation, x, j)
    return g(zero(x), itp, x, j)
end

function (diff_op_or_pde::DifferentialOperatorOrEquation)(itp::Interpolation)
    return x -> diff_op_or_pde(itp, x)
end

# TODO: Does this also make sense for conditionally positive definite kernels?
@doc raw"""
    kernel_inner_product(itp1, itp2)

Inner product of the native space for two [`Interpolation`](@ref) objects `itp1` and `itp2`
with the same kernel. The inner product is defined as
```math
    \langle f, g\rangle_K = \sum_{i = 1}^N\sum_{j = 1}^Mc_i^fc_j^gK(x_i, \xi_j)
```
for the interpolants ``f(x) = \sum_{i = 1}^Nc_i^fK(x, x_i)`` and
``g(x) = \sum_{j = 1}^Mc_j^gK(x, \xi_j)``.

See also [`kernel_norm`](@ref).
"""
function kernel_inner_product(itp1::Interpolation, itp2::Interpolation)
    kernel = interpolation_kernel(itp1)
    @assert kernel == interpolation_kernel(itp2)
    c_f = kernel_coefficients(itp1)
    c_g = kernel_coefficients(itp2)
    xs = centers(itp1)
    xis = centers(itp2)
    s = 0
    for i in eachindex(c_f)
        for j in eachindex(c_g)
            s += c_f[i] * c_g[j] * kernel(xs[i], xis[j])
        end
    end
    return s
end

@doc raw"""
    kernel_norm(itp)

Norm of the native space defined by the kernel of the [`Interpolation`](@ref) `itp`.
The norm is defined as
```math
    \|f\|_K^2 = \sum_{i,j=1}^Nc_ic_jK(x_i, x_j)
```
for the interpolant ``f(x) = \sum_{j = 1}^nc_jK(x, x_j)``.

See also [`kernel_inner_product`](@ref).
"""
kernel_norm(itp::Interpolation) = sqrt(max(0, kernel_inner_product(itp, itp))) # Use max to avoid numerical issues with negative values due to round-off errors

"""
    TemporalInterpolation(ode_sol::ODESolution)

Temporal interpolation of an ODE solution. The result can be evaluated at a time `t` and a spatial point `x`.
Evaluating the interpolation at a time `t` returns an [`Interpolation`](@ref) object that can
be evaluated at a spatial point `x`.
"""
struct TemporalInterpolation
    ode_sol::ODESolution
end

function Base.show(io::IO, titp::TemporalInterpolation)
    sd = titp.ode_sol.prob.p.spatial_discretization
    tspan = titp.ode_sol.prob.tspan
    N_i = length(sd.nodeset_inner)
    N_b = length(sd.nodeset_boundary)
    k = interpolation_kernel(sd.basis)
    return print(io,
                 "Temporal interpolation with $N_i inner nodes, $N_b boundary nodes, kernel $k, and time span $tspan")
end

function (titp::TemporalInterpolation)(t)
    ode_sol = titp.ode_sol
    semi = ode_sol.prob.p
    @unpack nodeset_inner, boundary_condition, nodeset_boundary, basis = semi.spatial_discretization
    c = ode_sol(t)
    # Do not support additional polynomial basis for now
    xx = polyvars(dim(semi))
    ps = monomials(xx, 0:-1)
    nodeset = merge(nodeset_inner, nodeset_boundary)
    itp = Interpolation(basis, nodeset, c, semi.cache.mass_matrix, ps, xx)
    return itp
end

# This should give the same result as
# c = ode_sol(t)
# A = semi.cache.kernel_matrix
# return A * c
function (titp::TemporalInterpolation)(t, x)
    itp = titp(t)
    return itp(x)
end

"""
    MultiscaleInterpolation

Container for a multiscale interpolation composed of several single-scale
[`Interpolation`](@ref) objects. Evaluation is the sum of the evaluations of the
individual scales.

See also [`multiscale_interpolate`](@ref).
"""
struct MultiscaleInterpolation{Basis, Dim, RealT, Itp} <:
       AbstractInterpolation{Basis, Dim, RealT}
    basis::Basis
    nodeset::NodeSet{Dim, RealT}
    itps::Vector{Itp}
end

function Base.show(io::IO, mitp::MultiscaleInterpolation)
    print(io, "Multiscale interpolation with $(length(mitp.itps)) scales")
    return nothing
end

function (mitp::MultiscaleInterpolation)(x)
    s = zero(eltype(x))
    for itp in mitp.itps
        s += itp(x)
    end
    return s
end

Base.lastindex(mitp::MultiscaleInterpolation) = length(mitp.itps)
Base.getindex(mitp::MultiscaleInterpolation, i) = mitp.itps[i]

"""
    multiscale_interpolate(nodesets, valuesets, kernels; kwargs...)

Construct a multiscale interpolation by fitting successive interpolants with
the given `kernels` to the residual. The `nodesets` and `valuesets` must be provided
as vectors of the same length as `kernels`, allowing grids and data to grow
between scales. Each scale is constructed by calling [`interpolate`](@ref)
on the corresponding `nodeset` and residual values.

- Armin Iske (2018)
  Multiresolution Methods in Scattered Data Modelling
  Lecture Notes in Computational Science and Engineering (Springer)
  [DOI: 10.1007/978-3-642-18754-4](https://doi.org/10.1007/978-3-642-18754-4)
"""
function multiscale_interpolate(nodesets::AbstractVector{<:NodeSet{Dim, RealT}},
                                valuesets::AbstractVector{<:AbstractVector{RealT}},
                                kernels::AbstractVector;
                                kwargs...) where {Dim, RealT}
    @assert length(nodesets) == length(valuesets) == length(kernels)
    isempty(kernels) &&
        throw(ArgumentError("At least one kernel is required for multiscale interpolation"))

    nlevels = length(kernels)
    itps_any = Vector{Any}(undef, nlevels)
    for (i, (nodeset, values, kernel)) in enumerate(zip(nodesets, valuesets, kernels))
        @assert length(values) == length(nodeset)
        residual = copy(values)
        for j in 1:(i - 1)
            prev_itp = itps_any[j]
            residual .-= prev_itp.(nodeset)
        end
        itp = interpolate(StandardBasis(nodeset, kernel), residual, nodeset; kwargs...)
        itps_any[i] = itp
    end

    # Convert to a concretely-typed vector of interpolants to preserve type information
    Itp = typeof(itps_any[end])
    itps = Vector{Itp}(itps_any)
    final_basis = basis(itps[end])
    return MultiscaleInterpolation{typeof(final_basis), Dim, RealT, Itp}(final_basis,
                                                                         nodesets[end],
                                                                         itps)
end
