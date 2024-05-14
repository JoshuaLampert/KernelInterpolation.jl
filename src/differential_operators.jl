abstract type AbstractDifferentialOperator end

function (D::AbstractDifferentialOperator)(kernel::RadialSymmetricKernel, x, y)
    @assert length(x) == length(y) == dim(kernel)
    return save_call(D, kernel, x .- y)
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

"""
    Gradient()

The gradient operator. It can be called with a [`RadialSymmetricKernel`](@ref) and points
`x` and `y` to evaluate the gradient of the `kernel` at `x - y`.
"""
struct Gradient <: AbstractDifferentialOperator
end

function Base.show(io::IO, ::Gradient)
    print(io, "∇")
end

function (::Gradient)(kernel::RadialSymmetricKernel, x)
    return ForwardDiff.gradient(x -> Phi(kernel, x), x)
end

"""
    Laplacian()

The Laplacian operator. It can be called with a [`RadialSymmetricKernel`](@ref) and points
`x` and `y` to evaluate the Laplacian of the `kernel` at `x - y`.
"""
struct Laplacian <: AbstractDifferentialOperator
end

function Base.show(io::IO, ::Laplacian)
    print(io, "Δ")
end

function (::Laplacian)(kernel::RadialSymmetricKernel, x)
    H = ForwardDiff.hessian(x -> Phi(kernel, x), x)
    return tr(H)
end

@doc raw"""
    EllipticOperator(A, b, c)

Linear second-order elliptic operator with matrix ``A(x)\in\mathbb{R}^{d\times d}``, vector ``b(x)\in\mathbb{R}^d``, and scalar ``c(x)``.
The operator is defined as
```math
    \mathcal{L}u = -\sum_{i,j = 1}^d a_{ij}(x)\partial_{x_i,x_j}^2u + \sum_{i = 1}^db_i(x)\partial_{x_i}u + c(x)u.
```
`A`, `b` and `c` are space-dependent functions returning a matrix, a vector and a scalar, respectively. The matrix `A` should be symmetric and
positive definite for any input `x`.
The operator can be called with a [`RadialSymmetricKernel`](@ref) and points `x` and `y` to evaluate the operator of the `kernel` at `x - y`.
"""
struct EllipticOperator{AType, BType, CType} <:
       AbstractDifferentialOperator where {AType, BType, CType}
    A::AType
    b::BType
    c::CType
end

function Base.show(io::IO, ::EllipticOperator)
    print(io, "-∑_{i,j = 1}^d aᵢⱼ (x)∂_{x_i,x_j}^2 + ∑_{i = 1}^d bᵢ(x)∂_{x_i} + c(x)")
end

function (operator::EllipticOperator)(kernel::RadialSymmetricKernel, x)
    @unpack A, b, c = operator
    AA = A(x)
    bb = b(x)
    cc = c(x)
    H = ForwardDiff.hessian(x -> Phi(kernel, x), x)
    gr = ForwardDiff.gradient(x -> Phi(kernel, x), x)

    return sum(-AA[i, j] * H[i, j] for i in 1:dim(kernel), j in 1:dim(kernel)) +
           sum(bb[i] * gr[i] for i in 1:dim(kernel)) +
           cc * Phi(kernel, x)
end
