abstract type AbstractInterpolation{Kernel, Dim, RealT} end

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

@doc raw"""
    Interpolation

Interpolation object that can be evaluated at a node and represents a kernel interpolation of the form
```math
    s(x) = \sum_{j = 1}^n c_jK(x, x_j) + \sum_{k = 1}^q d_kp_k(x),
```
where ``x_j`` are the nodes in the nodeset and ``s(x)`` the interpolant satisfying ``s(x_j) = f(x_j)``, where
``f(x_j)`` are given by `values` in [`interpolate`](@ref) and ``p_k`` is a basis of the `q`-dimensional space
of multivariate polynomials of order [`order`](@ref). The additional conditions
```math
    \sum_{j = 1}^n c_jp_k(x_j) = 0, \quad k = 1,\ldots, q
```
are enforced.
"""
struct Interpolation{Kernel, Dim, RealT, A, Monomials, PolyVars} <:
       AbstractInterpolation{Kernel, Dim, RealT}
    kernel::Kernel
    nodeset::NodeSet{Dim, RealT}
    c::Vector{RealT}
    system_matrix::A
    ps::Monomials
    xx::PolyVars
end

function Base.show(io::IO, itp::Interpolation)
    return print(io,
                 "Interpolation with $(length(nodeset(itp))) nodes, $(interpolation_kernel(itp)) kernel and polynomial of order $(order(itp)).")
end

"""
    dim(itp::Interpolation)

Return the dimension of the input variables of the interpolation.
"""
dim(itp::Interpolation{Kernel, Dim, RealT, A}) where {Kernel, Dim, RealT, A} = Dim

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
kernel_coefficients(itp::Interpolation) = itp.c[1:length(nodeset(itp))]

"""
    polynomial_coefficients(itp::Interpolation)

Obtain the coefficients of the polynomial part of the linear combination for the
interpolant.

See also [`coefficients`](@ref) and [`kernel_coefficients`](@ref).
"""
polynomial_coefficients(itp::Interpolation) = itp.c[(length(nodeset(itp)) + 1):end]

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

Return the system matrix, i.e., the matrix
```math
    \begin{pmatrix}
    A & P \\
    P^T & 0
    \end{pmatrix},
```
where ``A\in\mathbb{R}^{n\times n}`` is the matrix with entries
``a_{ij} = K(x_i, x_j)`` for the kernel function `K` and nodes `x_i`
and ``P\in\mathbb{R}^{n\times q}`` is the matrix with entries
``p_{ij} = p_j(x_i)``, where ``p_j`` is the ``j``-th multivariate monomial
of the space of polynomials up to degree ``m``.
"""
system_matrix(itp::Interpolation) = itp.system_matrix

@doc raw"""
	interpolate(nodeset, values, kernel = GaussKernel{dim(nodeset)}(), m = order(kernel))

Interpolate the `values` evaluated at the nodes in the `nodeset` to a function using the kernel `kernel`
and polynomials up to a degree `polynomial_degree`, i.e., determine the coefficients `c_j` and `d_k` in the expansion
```math
    s(x) = \sum_{j = 1}^n c_jK(x, x_j) + \sum_{k = 1}^q d_kp_k(x),
```
where ``x_j`` are the nodes in the nodeset and ``s(x)`` the interpolant ``s(x_j) = f(x_j)``, where ``f(x_j)``
are given by `values` and ``p_k`` is a basis of the `q`-dimensional space of multivariate polynomials with
maximum degree of `m - 1`. If `m = 0`, no polynomial is added. The additional conditions
```math
    \sum_{j = 1}^n c_jp_k(x_j) = 0, \quad k = 1,\ldots, q
```
are enforced. Returns an [`Interpolation`](@ref) object.
"""
function interpolate(nodeset::NodeSet{Dim, RealT}, values::Vector{RealT},
                     kernel = GaussKernel{Dim}(),
                     m = order(kernel)) where {Dim, RealT}
    @assert dim(kernel) == Dim
    n = length(nodeset)
    @assert length(values) == n
    xx = polyvars(Dim)
    ps = monomials(xx, 0:(m - 1))
    q = length(ps)

    kernel_matrix = Matrix{RealT}(undef, n, n)
    for i in 1:n
        for j in 1:n
            kernel_matrix[i, j] = kernel(nodeset[i], nodeset[j])
        end
    end
    polynomial_matrix = Matrix{RealT}(undef, n, q)
    for i in 1:n
        for j in 1:q
            polynomial_matrix[i, j] = ps[j](xx => nodeset[i])
        end
    end
    system_matrix = [kernel_matrix polynomial_matrix
                     transpose(polynomial_matrix) zeros(q, q)]
    b = [values; zeros(q)]
    symmetric_system_matrix = Symmetric(system_matrix)
    c = symmetric_system_matrix \ b
    return Interpolation(kernel, nodeset, c, symmetric_system_matrix, ps, xx)
end

"""
    solve(pde, nodeset_inner, nodeset_boundary, values_boundary, kernel = GaussKernel{dim(nodeset_inner)})

Solve a partial differential equation `pde` with Dirichlet boundary conditions by non-symmetric collocation
(Kansa method) using the kernel `kernel`. The `nodeset_inner` are the nodes in the domain and `nodeset_boundary`
are the nodes on the boundary. The `values_boundary` are the values of the boundary condition at the nodes given
by Dirichlet boundary conditions. Returns an [`Interpolation`](@ref) object.
"""
function solve(pde, nodeset_inner::NodeSet{Dim, RealT},
               nodeset_boundary::NodeSet{Dim, RealT},
               values_boundary::Vector{RealT},
               kernel = GaussKernel{Dim}()) where {Dim, RealT}
    @assert dim(kernel) == Dim
    n_i = length(nodeset_inner)
    n_b = length(nodeset_boundary)
    nodeset = merge(nodeset_inner, nodeset_boundary)
    n = n_i + n_b

    pde_matrix = Matrix{RealT}(undef, n_i, n)
    for i in 1:n_i
        for j in 1:n
            pde_matrix[i, j] = pde(kernel, nodeset_inner[i], nodeset[j])
        end
    end
    boundary_matrix = Matrix{RealT}(undef, n_b, n)
    for i in 1:n_b
        for j in 1:n
            # Dirichlet boundary condition
            boundary_matrix[i, j] = kernel(nodeset_boundary[i], nodeset[j])
        end
    end
    system_matrix = [pde_matrix
                     boundary_matrix]
    b = [rhs(pde, nodeset_inner); values_boundary]
    c = system_matrix \ b

    # Do not support additional polynomial basis for now
    xx = polyvars(Dim)
    ps = monomials(xx, 0:-1)
    return Interpolation(kernel, nodeset, c, system_matrix, ps, xx)
end

# Evaluate interpolant
function (itp::Interpolation)(x)
    s = 0
    kernel = interpolation_kernel(itp)
    xs = nodeset(itp)
    c = kernel_coefficients(itp)
    d = polynomial_coefficients(itp)
    ps = polynomial_basis(itp)
    xx = polyvars(itp)
    for j in 1:length(c)
        s += c[j] * kernel(x, xs[j])
    end

    for k in 1:length(d)
        s += d[k] * ps[k](xx => x)
    end
    return s
end

# Allow scalar input if interpolant is one-dimensional
function (itp::Interpolation)(x::RealT) where {RealT <: Real}
    @assert dim(itp) == 1
    return itp([x])
end

function (diff_op::AbstractDifferentialOperator)(itp::Interpolation, x)
    kernel = interpolation_kernel(itp)
    xs = nodeset(itp)
    c = kernel_coefficients(itp)
    s = 0
    for j in 1:length(c)
        s += c[j] * diff_op(kernel, x, xs[j])
    end
    return s
end

function (pde::AbstractPDE)(itp::Interpolation, x)
    kernel = interpolation_kernel(itp)
    xs = nodeset(itp)
    c = kernel_coefficients(itp)
    s = 0
    for j in 1:length(c)
        s += c[j] * pde(kernel, x, xs[j])
    end
    return s
end

# TODO: Does this also make sense for conditionally positive definite kernels?
@doc raw"""
    kernel_inner_product(itp1, itp2)

Inner product of the native space for two interpolants `itp1` and `itp2`
with the same kernel. The inner product is defined as
```math
    (f, g)_K = \sum_{j = 1}^n\sum_{k = 1}^mc_jd_kK(x_j, y_k)
```
for the interpolants ``f(x) = \sum_{j = 1}^nc_jK(x, x_j)`` and
``g(x) = \sum_{k = 1}^md_kK(x, y_k)``.

See also [`kernel_norm`](@ref).
"""
function kernel_inner_product(itp1, itp2)
    kernel = interpolation_kernel(itp1)
    @assert kernel == interpolation_kernel(itp2)
    c = kernel_coefficients(itp1)
    d = kernel_coefficients(itp2)
    xs = nodeset(itp1)
    ys = nodeset(itp2)
    s = 0
    for j in 1:length(c)
        for k in 1:length(d)
            s += c[j] * d[k] * kernel(xs[j], ys[k])
        end
    end
    return s
end

@doc raw"""
    kernel_norm(itp)

Norm of the native space defined by the kernel of the interpolant `itp`.
The norm is defined as
```math
    \|f\|_K^2 = \sum_{j,k=1}^nc_jc_kK(x_j, x_k)
```
for the interpolant ``f(x) = \sum_{j = 1}^nc_jK(x, x_j)``.

See also [`kernel_inner_product`](@ref).
"""
kernel_norm(itp) = sqrt(kernel_inner_product(itp, itp))
