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

function Base.show(io::IO, itp::Interpolation)
    return print(io,
                 "Interpolation with $(length(nodeset(itp))) nodes, kernel $(interpolation_kernel(itp)) and polynomial of order $(order(itp)).")
end

"""
    dim(itp::Interpolation)

Return the dimension of the input variables of the interpolation.
"""
dim(::Interpolation{Basis, Dim}) where {Basis, Dim} = Dim

"""
    basis(itp)

Return the basis from an interpolation object.
"""
basis(itp::Interpolation) = itp.basis

"""
    interpolation_kernel(itp)

Return the kernel from an interpolation object.
"""
interpolation_kernel(itp::AbstractInterpolation) = interpolation_kernel(basis(itp))

"""
	nodeset(itp)

Return the node set from an interpolation object.
"""
nodeset(itp::AbstractInterpolation) = itp.nodeset

"""
    centers(itp::Interpolation)

Return the centers from the basis of an interpolation object.
"""
centers(itp::Interpolation) = centers(basis(itp))

"""
    coefficients(itp::Interpolation)

Obtain all the coefficients of the linear combination for the interpolant, i.e., both
the coefficients for the kernel part and for the polynomial part.

See also [`kernel_coefficients`](@ref) and [`polynomial_coefficients`](@ref).
"""
coefficients(itp::Interpolation) = itp.c

"""
    kernel_coefficients(itp::Interpolation)

Obtain the coefficients of the kernel part of the linear combination for the
interpolant.

See also [`coefficients`](@ref) and [`polynomial_coefficients`](@ref).
"""
@views kernel_coefficients(itp::Interpolation) = itp.c[eachindex(centers(itp))]

"""
    polynomial_coefficients(itp::Interpolation)

Obtain the coefficients of the polynomial part of the linear combination for the
interpolant.

See also [`coefficients`](@ref) and [`kernel_coefficients`](@ref).
"""
@views polynomial_coefficients(itp::Interpolation) = itp.c[(length(centers(itp)) + 1):end]

"""
    polynomial_basis(itp::Interpolation)

Return a vector of the polynomial basis functions used for the interpolation.

See also [`polyvars`](@ref).
"""
polynomial_basis(itp::Interpolation) = itp.ps

"""
    polyvars(itp::Interpolation)

Return a vector of the polynomial variables.

See also [`polynomial_basis`](@ref).
"""
polyvars(itp::Interpolation) = itp.xx

"""
    order(itp)

Return the order ``m`` of the polynomial used for the interpolation, i.e.,
the polynomial degree plus 1. If ``m = 0``, no polynomial is added.
"""
order(itp::Interpolation) = maximum(degree.(itp.ps), init = -1) + 1

@doc raw"""
    system_matrix(itp::Interpolation)

Return the system matrix, i.e., the matrix ``A`` in the linear system
```math
    Ac = f,
```
where ``c`` are the coefficients of the kernel interpolant and ``f`` the vector
of known values. The exact form of ``A`` differs depending on which method is used.
"""
system_matrix(itp::Interpolation) = itp.system_matrix

@doc raw"""
    interpolate(basis, values, nodeset = centers(basis); m = order(basis),
                regularization = NoRegularization(), factorization_method = nothing)
    interpolate(centers, [nodeset,] values, kernel = GaussKernel{dim(nodeset)}();
                m = order(kernel), regularization = NoRegularization(),
                factorization_method = nothing)

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
the system matrix is just wrapped as a Symmetric matrix for interpolation and no factorization is applied
for a least squares solution, but you can, e.g., also explicitly use `cholesky`, `lu`, or `qr` factorization.
"""
function interpolate(basis::AbstractBasis, values::Vector{RealT},
                     nodeset::NodeSet{Dim, RealT} = centers(basis);
                     m = order(basis),
                     regularization = NoRegularization(),
                     factorization_method = nothing) where {Dim, RealT}
    @assert dim(basis) == Dim
    n = length(nodeset)
    @assert length(values) == n
    xx = polyvars(Val(Dim))
    ps = monomials(xx, 0:(m - 1))
    q = length(ps)

    if nodeset == centers(basis)
        factorization_method = isnothing(factorization_method) ? Symmetric :
                               factorization_method
        system_matrix = interpolation_matrix(basis, ps, regularization;
                                             factorization_method)
    else
        factorization_method = isnothing(factorization_method) ? Matrix :
                               factorization_method
        system_matrix = least_squares_matrix(basis, nodeset, ps, regularization;
                                             factorization_method)
    end
    b = [values; zeros(RealT, q)]
    c = system_matrix \ b
    return Interpolation{typeof(basis), dim(basis), eltype(nodeset), typeof(system_matrix),
                         typeof(ps), typeof(xx)}(basis, nodeset, c, system_matrix, ps, xx)
end
function interpolate(centers::NodeSet{Dim, RealT}, nodeset::NodeSet{Dim, RealT},
                     values::AbstractVector{RealT}, kernel = GaussKernel{Dim, RealT}();
                     kwargs...) where {Dim, RealT}
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

function (diff_op_or_pde::Union{AbstractDifferentialOperator, AbstractStationaryEquation})(itp::Interpolation,
                                                                                           x)
    kernel = interpolation_kernel(itp)
    xis = centers(itp)
    c = kernel_coefficients(itp)
    s = zero(eltype(x))
    for j in eachindex(c)
        s += c[j] * diff_op_or_pde(kernel, x, xis[j])
    end
    return s
end

function (g::Gradient)(itp::Interpolation, x)
    kernel = interpolation_kernel(itp)
    xis = centers(itp)
    c = kernel_coefficients(itp)
    s = zero(x)
    for j in eachindex(c)
        s += c[j] * g(kernel, x, xis[j])
    end
    return s
end

# TODO: Does this also make sense for conditionally positive definite kernels?
@doc raw"""
    kernel_inner_product(itp1, itp2)

Inner product of the native space for two interpolants `itp1` and `itp2`
with the same kernel. The inner product is defined as
```math
    \langle f, g\rangle_K = \sum_{i = 1}^N\sum_{j = 1}^Mc_i^fc_j^gK(x_i, \xi_j)
```
for the interpolants ``f(x) = \sum_{i = 1}^Nc_i^fK(x, x_i)`` and
``g(x) = \sum_{j = 1}^Mc_j^gK(x, \xi_j)``.

See also [`kernel_norm`](@ref).
"""
function kernel_inner_product(itp1, itp2)
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

Norm of the native space defined by the kernel of the interpolant `itp`.
The norm is defined as
```math
    \|f\|_K^2 = \sum_{i,j=1}^Nc_ic_jK(x_i, x_j)
```
for the interpolant ``f(x) = \sum_{j = 1}^nc_jK(x, x_j)``.

See also [`kernel_inner_product`](@ref).
"""
kernel_norm(itp) = sqrt(kernel_inner_product(itp, itp))

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
    @unpack centers, kernel = basis
    c = ode_sol(t)
    # Do not support additional polynomial basis for now
    xx = polyvars(dim(semi))
    ps = monomials(xx, 0:-1)
    nodeset = merge(nodeset_inner, nodeset_boundary)
    itp = Interpolation(StandardBasis(centers, kernel), nodeset, c,
                        semi.cache.mass_matrix, ps, xx)
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
