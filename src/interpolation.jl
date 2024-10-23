abstract type AbstractInterpolation{Kernel, Dim, RealT} end

@doc raw"""
    Interpolation

Interpolation object that can be evaluated at a node and represents a kernel interpolation of the form
```math
    s(x) = \sum_{j = 1}^N c_jK(x, x_j) + \sum_{k = 1}^Q d_kp_k(x),
```
where ``x_j`` are the nodes in the nodeset and ``s(x)`` the interpolant satisfying ``s(x_j) = f(x_j)``, where
``f(x_j)`` are given by `values` in [`interpolate`](@ref) and ``p_k`` is a basis of the `Q`-dimensional space
of multivariate polynomials of order [`order`](@ref). The additional conditions
```math
    \sum_{j = 1}^N c_jp_k(x_j) = 0, \quad k = 1,\ldots, Q
```
are enforced.
"""
struct Interpolation{Kernel, Dim, RealT, A, Monomials, PolyVars} <:
       AbstractInterpolation{Kernel, Dim, RealT}
    kernel::Kernel
    nodeset::NodeSet{Dim, RealT}
    centers::NodeSet{Dim, RealT}
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
dim(::Interpolation{Kernel, Dim, RealT, A}) where {Kernel, Dim, RealT, A} = Dim

"""
    interpolation_kernel(itp)

Return the kernel from an interpolation object.
"""
interpolation_kernel(itp::AbstractInterpolation) = itp.kernel

"""
	nodeset(itp)

Return the node set from an interpolation object.
"""
nodeset(itp::AbstractInterpolation) = itp.nodeset

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
kernel_coefficients(itp::Interpolation) = itp.c[eachindex(itp.centers)]

"""
    polynomial_coefficients(itp::Interpolation)

Obtain the coefficients of the polynomial part of the linear combination for the
interpolant.

See also [`coefficients`](@ref) and [`kernel_coefficients`](@ref).
"""
polynomial_coefficients(itp::Interpolation) = itp.c[(length(itp.centers) + 1):end]

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
    interpolate(nodeset, centers = nodeset, values, kernel = GaussKernel{dim(nodeset)}();
                m = order(kernel), regularization = NoRegularization())

Interpolate the `values` evaluated at the nodes in the `nodeset` to a function using the kernel `kernel`
and polynomials up to a order `m` (i.e. degree - 1), i.e., determine the coefficients ``c_j`` and ``d_k`` in the expansion
```math
    s(x) = \sum_{j = 1}^N c_jK(x, x_j) + \sum_{k = 1}^Q d_kp_k(x),
```
where ``x_j`` are the nodes in the nodeset and ``s(x)`` the interpolant ``s(x_j) = f(x_j)``, where ``f(x_j)``
are given by `values` and ``p_k`` is a basis of the ``Q``-dimensional space of multivariate polynomials with
maximum degree of `m - 1`. If `m = 0`, no polynomial is added. The additional conditions
```math
    \sum_{j = 1}^N c_jp_k(x_j) = 0, \quad k = 1,\ldots, Q = \begin{pmatrix}m - 1 + d\\d\end{pmatrix}
```
are enforced. Returns an [`Interpolation`](@ref) object.

If `centers` is provided, the interpolant is a least squares approximation with the centers used for the basis.

A regularization can be applied to the kernel matrix using the `regularization` argument, cf. [`regularize!`](@ref).
"""
function interpolate(nodeset::NodeSet{Dim, RealT}, centers::NodeSet{Dim, RealT},
                     values::Vector{RealT}, kernel = GaussKernel{Dim}();
                     m = order(kernel),
                     regularization = NoRegularization()) where {Dim, RealT}
    @assert dim(kernel) == Dim
    n = length(nodeset)
    @assert length(values) == n
    xx = polyvars(Dim)
    ps = monomials(xx, 0:(m - 1))
    q = length(ps)

    if nodeset == centers
        system_matrix = interpolation_matrix(nodeset, kernel, ps, regularization)
    else
        system_matrix = least_squares_matrix(nodeset, centers, kernel, ps, regularization)
    end
    b = [values; zeros(q)]
    c = system_matrix \ b
    return Interpolation(kernel, nodeset, centers, c, system_matrix, ps, xx)
end

function interpolate(nodeset::NodeSet{Dim, RealT},
                     values::Vector{RealT}, kernel = GaussKernel{Dim}();
                     kwargs...) where {Dim, RealT}
    interpolate(nodeset, nodeset, values, kernel; kwargs...)
end

# Evaluate interpolant
function (itp::Interpolation)(x)
    s = 0
    kernel = interpolation_kernel(itp)
    xis = itp.centers
    c = kernel_coefficients(itp)
    d = polynomial_coefficients(itp)
    ps = polynomial_basis(itp)
    xx = polyvars(itp)
    for j in eachindex(c)
        s += c[j] * kernel(x, xis[j])
    end

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
    xis = itp.centers
    c = kernel_coefficients(itp)
    s = zero(eltype(x))
    for j in eachindex(c)
        s += c[j] * diff_op_or_pde(kernel, x, xis[j])
    end
    return s
end

function (g::Gradient)(itp::Interpolation, x)
    kernel = interpolation_kernel(itp)
    xis = itp.centers
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
    xs = itp1.centers
    xis = itp2.centers
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
    return print(io,
                 "Temporal interpolation with $(length(sd.nodeset_inner)) inner nodes, $(length(sd.nodeset_boundary)) boundary nodes, kernel $(sd.kernel), and time span $tspan")
end

function (titp::TemporalInterpolation)(t)
    ode_sol = titp.ode_sol
    semi = ode_sol.prob.p
    @unpack kernel, nodeset_inner, boundary_condition, nodeset_boundary = semi.spatial_discretization
    c = ode_sol(t)
    # Do not support additional polynomial basis for now
    xx = polyvars(dim(semi))
    ps = monomials(xx, 0:-1)
    nodeset = merge(nodeset_inner, nodeset_boundary)
    itp = Interpolation(kernel, nodeset, nodeset, c,
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
