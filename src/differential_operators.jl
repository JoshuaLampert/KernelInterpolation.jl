abstract type AbstractDifferentialOperator end

function (D::AbstractDifferentialOperator)(kernel::RadialSymmetricKernel, x, y)
    @assert length(x) == length(y) == dim(kernel)
    return save_call(D, kernel, x .- y)
end

function (D::AbstractDifferentialOperator)(kernel::RadialSymmetricKernel)
    return x -> D(kernel, x)
end

# Workaround to avoid evaluating the derivative at zeros to allow automatic differentiation,
# see https://github.com/JuliaDiff/ForwardDiff.jl/issues/303
# the same issue appears with Zygote.jl
function save_call(D::AbstractDifferentialOperator, kernel::RadialSymmetricKernel, x)
    if all(iszero, x)
        x[1] = eps(typeof(x[1]))
    end
    # x .+= eps(typeof(x[1]))
    return D(kernel, x)
end

# Convert a kernel or polynomial to a plain Julia function, so that differential operators
# can be defined once for `Function` and applied to either.
callable(kernel::RadialSymmetricKernel) = x -> Phi(kernel, x)
callable(p::AbstractPolynomialLike) = let xx = variables(p)
    y -> p(xx => y)
end

# Abstract fallback: convert kernel or polynomial to a callable, then apply the operator.
# This covers the 2-arg kernel form (used internally via save_call) and polynomials.
function (D::AbstractDifferentialOperator)(f::Union{RadialSymmetricKernel,
                                                    AbstractPolynomialLike}, x)
    return D(callable(f), x)
end

"""
    PartialDerivative(i)

Partial derivative operator with respect to the `i`-th component.
The operator can be called with a [`RadialSymmetricKernel`](@ref) and points
`x` and `y` to evaluate the derivative of the `kernel` at `x - y`.
It can also be called with an [`Interpolation`](@ref) object and a point `x` to evaluate
the first partial derivative of the interpolation at `x` in the `i`-th direction. Note that this
is only supported for the kernel part of the interpolation, i.e. the polynomial part, if existent, is ignored.
"""
struct PartialDerivative <: AbstractDifferentialOperator
    i::Int
end

function Base.show(io::IO, operator::PartialDerivative)
    print(io, "∂_{x_", operator.i, "}")
    return nothing
end

function (operator::PartialDerivative)(f::Function, x)
    return ForwardDiff.gradient(f, x)[operator.i]
end

"""
    Gradient()

The gradient operator. It can be called with a [`RadialSymmetricKernel`](@ref) and points
`x` and `y` to evaluate the gradient of the `kernel` at `x - y`.
It can also be called with an [`Interpolation`](@ref) object and a point `x` to evaluate
the gradient of the interpolation at `x`. Note that this is only supported
for the kernel part of the interpolation, i.e. the polynomial part, if existent, is ignored.
"""
struct Gradient <: AbstractDifferentialOperator
end

function Base.show(io::IO, ::Gradient)
    print(io, "∇")
    return nothing
end

function (::Gradient)(f::Function, x)
    return ForwardDiff.gradient(f, x)
end

"""
    Laplacian()

The Laplacian operator. It can be called with a [`RadialSymmetricKernel`](@ref) and points
`x` and `y` to evaluate the Laplacian of the `kernel` at `x - y`.
It can also be called with an [`Interpolation`](@ref) object and a point `x` to evaluate
the Laplacian of the interpolation at `x`. Note that this is only supported
for the kernel part of the interpolation, i.e. the polynomial part, if existent, is ignored.
"""
struct Laplacian <: AbstractDifferentialOperator
end

function Base.show(io::IO, ::Laplacian)
    print(io, "Δ")
    return nothing
end

function (::Laplacian)(f::Function, x)
    return tr(ForwardDiff.hessian(f, x))
end

@doc raw"""
    EllipticOperator(A, b, c)

Linear second-order elliptic operator with matrix ``A(x)\in\mathbb{R}^{d\times d}``, vector
``b(x)\in\mathbb{R}^d``, and scalar ``c(x)``.
The operator is defined as
```math
    \mathcal{L}u = -\sum_{i,j = 1}^d a_{ij}(x)\partial_{x_i,x_j}^2u + \sum_{i = 1}^db_i(x)\partial_{x_i}u + c(x)u.
```
`A`, `b` and `c` are space-dependent functions returning a matrix, a vector, and a scalar,
respectively. The matrix `A` should be symmetric and positive definite for any input `x`.
The operator can be called with a [`RadialSymmetricKernel`](@ref) and points `x` and `y` to
evaluate the operator of the `kernel` at `x - y`.
It can also be called with an [`Interpolation`](@ref) object and a point `x` to evaluate
the elliptic operator of the interpolation at `x`. Note that this is only supported
for the kernel part of the interpolation, i.e. the polynomial part, if existent, is ignored.
"""
struct EllipticOperator{AType, BType, CType} <:
       AbstractDifferentialOperator where {AType, BType, CType}
    A::AType
    b::BType
    c::CType
end

function Base.show(io::IO, ::EllipticOperator)
    print(io, "-∑_{i,j = 1}^d aᵢⱼ (x)∂_{x_i,x_j}^2 + ∑_{i = 1}^d bᵢ(x)∂_{x_i} + c(x)")
    return nothing
end

function (operator::EllipticOperator)(f::Function, x)
    @unpack A, b, c = operator
    AA = A(x)
    bb = b(x)
    cc = c(x)
    H = ForwardDiff.hessian(f, x)
    gr = ForwardDiff.gradient(f, x)
    return sum(-AA[i, j] * H[i, j] for i in eachindex(gr), j in eachindex(gr)) +
           sum(bb[i] * gr[i] for i in eachindex(gr)) +
           cc * f(x)
end
