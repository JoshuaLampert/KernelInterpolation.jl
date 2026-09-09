abstract type AbstractDifferentialOperator end

function (D::AbstractDifferentialOperator)(kernel::RadialSymmetricKernel)
    return x -> D(kernel, x)
end

# Convert a kernel or polynomial to a plain Julia function, so that operators/equations
# can be defined once for `Function` and applied to either.
#
# For a radial-symmetric kernel the result is a `CallableKernel` rather than an anonymous
# closure. It behaves like any other `Function`, but it remembers which kernel it came from,
# so operators that know a closed-form chain rule (see below) can dispatch on it. Everything
# else, including all equations and any user-defined operator, keeps a single implementation
# taking a `Function` and picks up those radial implementations automatically.
struct CallableKernel{K <: RadialSymmetricKernel} <: Function
    kernel::K
end

(f::CallableKernel)(x) = Phi(f.kernel, x)

callable(kernel::RadialSymmetricKernel) = CallableKernel(kernel)
function callable(p::AbstractPolynomialLike)
    xx = variables(p)
    return y -> p(xx => y)
end

# Derivatives of a radial-symmetric kernel are evaluated via the chain rule on the scalar
# radial profile `phi`, and never by differentiating `x -> Phi(kernel, x)` directly. The
# reason is that the only singularity is in `norm(x)` at the origin, not in `phi` itself, so
# differentiating the profile is well behaved and the removable singularity at the kernel
# center can be handled by its analytic limit instead of by perturbing the argument.
#
# With r = ||x|| and s = x / r:
#
#   ∂ᵢ Φ    = φ'(r) sᵢ
#   ∂ᵢⱼ Φ   = (φ'(r) / r) δᵢⱼ + (φ''(r) - φ'(r) / r) sᵢ sⱼ
#   ΔΦ      = φ''(r) + (d - 1) φ'(r) / r
#
# At the origin these have limits if and only if the kernel is smooth enough, which is what
# [`smoothness`](@ref) records: an operator of order `m` may be evaluated there exactly when
# `smoothness(kernel) >= m`. The limits are then `∂ᵢΦ(0) = 0`, `∂ᵢⱼΦ(0) = φ''(0) δᵢⱼ` and
# `ΔΦ(0) = d φ''(0)`.
phi_deriv(kernel::RadialSymmetricKernel, r) = ForwardDiff.derivative(s -> phi(kernel, s), r)
function phi_deriv2(kernel::RadialSymmetricKernel, r)
    return ForwardDiff.derivative(s -> phi_deriv(kernel, s), r)
end

function assert_smooth_enough(kernel::RadialSymmetricKernel, m::Int, name)
    if smoothness(kernel) < m
        throw(ArgumentError("$name of $(get_name(kernel)) is not defined at the center of " *
                            "the kernel: the kernel is only C^$(smoothness(kernel)), but " *
                            "an operator of order $m requires smoothness at least $m."))
    end
    return nothing
end

"""
    Identity()

The identity operator, i.e. the differential operator of order zero. Applied to a function
it returns its value, and like the other operators it can be called with a
[`RadialSymmetricKernel`](@ref) and points `x` and `y` to evaluate the `kernel` at `x - y`,
or with an [`Interpolation`](@ref) object and a point `x` to evaluate the interpolation at
`x`. This lets plain evaluation reuse the same interface as the differential operators (for
example in the RBF-FD weight computation, see [`local_weights`](@ref)).
"""
struct Identity <: AbstractDifferentialOperator end

function Base.show(io::IO, ::Identity)
    print(io, "Id")
    return nothing
end

(::Identity)(f::Function, x) = f(x)
# Evaluate the kernel directly (no derivative, so no smoothness requirement). This
# is more specific than the generic `(op)(kernel, x, y)` in `equations.jl`, hence unambiguous.
(::Identity)(kernel::RadialSymmetricKernel, x, y) = kernel(x, y)

"""
    PartialDerivative(i)

Partial derivative operator with respect to the `i`-th component.
The operator can be called with a [`RadialSymmetricKernel`](@ref) and points
`x` and `y` to evaluate the derivative of the `kernel` at `x - y`.
It can also be called with an [`Interpolation`](@ref) object and a point `x` to evaluate
the first partial derivative of the interpolation at `x` in the `i`-th direction.
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

function (operator::PartialDerivative)(f::CallableKernel, x)
    r = norm(x)
    if iszero(r)
        assert_smooth_enough(f.kernel, 1, "PartialDerivative")
        return zero(eltype(x))
    end
    return phi_deriv(f.kernel, r) * x[operator.i] / r
end

"""
    Gradient()

The gradient operator. It can be called with a [`RadialSymmetricKernel`](@ref) and points
`x` and `y` to evaluate the gradient of the `kernel` at `x - y`.
It can also be called with an [`Interpolation`](@ref) object and a point `x` to evaluate
the gradient of the interpolation at `x`.
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

function (::Gradient)(f::CallableKernel, x)
    r = norm(x)
    if iszero(r)
        assert_smooth_enough(f.kernel, 1, "Gradient")
        return zero(x)
    end
    return phi_deriv(f.kernel, r) / r * x
end

"""
    Laplacian()

The Laplacian operator. It can be called with a [`RadialSymmetricKernel`](@ref) and points
`x` and `y` to evaluate the Laplacian of the `kernel` at `x - y`.
It can also be called with an [`Interpolation`](@ref) object and a point `x` to evaluate
the Laplacian of the interpolation at `x`.
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

function (::Laplacian)(f::CallableKernel, x)
    kernel = f.kernel
    r = norm(x)
    if iszero(r)
        assert_smooth_enough(kernel, 2, "Laplacian")
        return dim(kernel) * phi_deriv2(kernel, r)
    end
    return phi_deriv2(kernel, r) + (dim(kernel) - 1) * phi_deriv(kernel, r) / r
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
the elliptic operator of the interpolation at `x`.
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

function (operator::EllipticOperator)(f::CallableKernel, x)
    @unpack A, b, c = operator
    kernel = f.kernel
    d = dim(kernel)
    AA = A(x)
    bb = b(x)
    cc = c(x)
    r = norm(x)
    if iszero(r)
        assert_smooth_enough(kernel, 2, "EllipticOperator")
        # ∂ᵢⱼΦ(0) = φ''(0) δᵢⱼ and ∇Φ(0) = 0
        d2 = phi_deriv2(kernel, r)
        return -d2 * sum(AA[i, i] for i in 1:d) + cc * phi(kernel, r)
    end
    d1r = phi_deriv(kernel, r) / r
    d2 = phi_deriv2(kernel, r)
    return sum(-AA[i, j] * ((i == j ? d1r : zero(d1r)) + (d2 - d1r) * x[i] * x[j] / r^2)
               for i in 1:d, j in 1:d) +
           sum(bb[i] * d1r * x[i] for i in 1:d) +
           cc * phi(kernel, r)
end
