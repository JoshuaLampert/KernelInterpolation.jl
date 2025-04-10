@doc raw"""
    kernel_matrix(basis, nodeset = centers(basis))
    kernel_matrix(nodeset1[, nodeset2], kernel)

Return the kernel matrix for the `nodes` and `kernel`. The kernel matrix is defined as
```math
    A_{ij} = b_j(x_i),
```
where ``b_i`` are the basis function in the `basis` and `x_i` are the nodes in the `nodeset`.
If two nodesets and a `kernel` are given, the kernel matrix is computed for the [`StandardBasis`](@ref) meaning
```math
    A_{ij} = K(\xi_j, x_i),
```
where ``\xi_j`` are the nodes/centers in `nodeset1`, ``x_i`` are the nodes in `nodeset2`, and `K` is the `kernel`.
If `nodeset2` is not given, it defaults to `nodeset1`.
"""
function kernel_matrix(basis::AbstractBasis, nodeset::NodeSet = centers(basis))
    n = length(nodeset)
    m = length(basis)
    A = Matrix{eltype(nodeset)}(undef, n, m)
    for i in 1:n
        for j in 1:m
            A[i, j] = basis[j](nodeset[i])
        end
    end
    return A
end

function kernel_matrix(nodeset1::NodeSet{Dim}, nodeset2::NodeSet{Dim},
                       kernel::AbstractKernel{Dim}) where {Dim}
    kernel_matrix(StandardBasis(nodeset1, kernel), nodeset2)
end

function kernel_matrix(nodeset::NodeSet, kernel::AbstractKernel)
    kernel_matrix(nodeset, nodeset, kernel)
end

"""
    polynomial_matrix(nodeset, ps)

Return the polynomial matrix for the nodeset and polynomials. The polynomial matrix is defined as
```math
    A_{ij} = p_j(x_i),
```
where ``x_i`` are the nodes in the `nodeset` and ``p_j`` the polynomials.
"""
function polynomial_matrix(nodeset::NodeSet, ps)
    n = length(nodeset)
    q = length(ps)
    A = Matrix{eltype(nodeset)}(undef, n, q)
    xx = polyvars(dim(nodeset))
    for i in 1:n
        for j in 1:q
            A[i, j] = ps[j](xx => nodeset[i])
        end
    end
    return A
end

@doc raw"""
    interpolation_matrix(centers, kernel, ps, regularization = NoRegularization())
    interpolation_matrix(basis, ps, regularization)

Return the interpolation matrix for the `basis`, polynomials `ps`, and `regularization`.
For the [`StandardBasis`](@ref), the interpolation matrix is defined as
```math
    A = \begin{pmatrix}K & P\\P^T & 0\end{pmatrix},
```
where ``K`` is the [`regularize!`](@ref)d [`kernel_matrix`](@ref) and ``P`` the [`polynomial_matrix`](@ref).
If a node set of `centers` and a `kernel` are given, the interpolation matrix is computed for the [`StandardBasis`](@ref).
"""
function interpolation_matrix(basis::AbstractBasis, ps,
                              regularization::AbstractRegularization = NoRegularization())
    q = length(ps)
    k_matrix = kernel_matrix(basis)
    regularize!(k_matrix, regularization)
    # We could always use the first branch, but this is more efficient
    # for the case where we don't use polynomial augmentation (q == 0).
    if q > 0
        p_matrix = polynomial_matrix(centers(basis), ps)
        system_matrix = [k_matrix p_matrix
                         p_matrix' zeros(eltype(k_matrix), q, q)]
    else
        # We could also use `cholesky` here because usually `k_matrix` is
        # symmetric positive definite, but this might not be the case if
        # the user explicitly sets `m = 0` even though the kernel is not
        # strictly positive definite or the matrix might be numerically not spd.
        # TODO: Think of an interface to allow for general matrix factorizations.
        system_matrix = k_matrix
    end
    return Symmetric(system_matrix)
end

# This should be the same as `kernel_matrix(basis)`
function interpolation_matrix(::LagrangeBasis, ps,
                              ::AbstractRegularization = NoRegularization())
    return I
end

function interpolation_matrix(centers::NodeSet, kernel::AbstractKernel, ps,
                              regularization::AbstractRegularization = NoRegularization())
    interpolation_matrix(StandardBasis(centers, kernel), ps, regularization)
end

@doc raw"""
    least_squares_matrix(basis, nodeset, ps, regularization = NoRegularization())
    least_squares_matrix(centers, nodeset, kernel, ps, regularization = NoRegularization())

Return the least squares matrix for the `basis`, `nodeset`, polynomials `ps`, and `regularization`.
For the [`StandardBasis`](@ref), the least squares matrix is defined as
```math
    A = \begin{pmatrix}K & P_1\\P_2^T & 0\end{pmatrix},
```
where ``K`` is the [`regularize!`](@ref)d [`kernel_matrix`](@ref), ``P_1`` the [`polynomial_matrix`](@ref)
for the `nodeset` and ``P_2`` the [`polynomial_matrix`](@ref)` for the `centers`.
If a `nodeset` and `kernel` are given, the least squares matrix is computed for the [`StandardBasis`](@ref).
"""
function least_squares_matrix(basis::AbstractBasis, nodeset::NodeSet, ps,
                              regularization::AbstractRegularization = NoRegularization())
    q = length(ps)
    k_matrix = kernel_matrix(basis, nodeset)
    regularize!(k_matrix, regularization)
    p_matrix1 = polynomial_matrix(nodeset, ps)
    p_matrix2 = polynomial_matrix(centers(basis), ps)
    system_matrix = [k_matrix p_matrix1
                     p_matrix2' zeros(eltype(k_matrix), q, q)]
    return system_matrix
end

function least_squares_matrix(basis::LagrangeBasis, nodeset::NodeSet, ps,
                              regularization::AbstractRegularization = NoRegularization())
    k_matrix = kernel_matrix(basis, nodeset)
    regularize!(k_matrix, regularization)
    return k_matrix
end

function least_squares_matrix(centers::NodeSet, nodeset::NodeSet, kernel::AbstractKernel,
                              ps,
                              regularization::AbstractRegularization = NoRegularization())
    least_squares_matrix(StandardBasis(centers, kernel), nodeset, ps, regularization)
end

@doc raw"""
    pde_matrix(diff_op_or_pde, nodeset1, nodeset2, kernel)

Compute the matrix of a partial differential equation (or differential operator) with a given kernel. The matrix is defined as
```math
    (\tilde A_\mathcal{L})_{ij} = \mathcal{L}K(x_i, \xi_j),
```
where ``\mathcal{L}`` is the differential operator (defined by the `equations`), ``K`` the `kernel`, ``x_i`` are the nodes
in `nodeset1` and ``\xi_j`` are the nodes in `nodeset2`.
"""
function pde_matrix(diff_op_or_pde, nodeset1, nodeset2, kernel)
    n = length(nodeset1)
    m = length(nodeset2)
    A = Matrix{eltype(nodeset1)}(undef, n, m)
    for i in 1:n
        for j in 1:m
            A[i, j] = diff_op_or_pde(kernel, nodeset1[i], nodeset2[j])
        end
    end
    return A
end

@doc raw"""
    pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, [centers,] kernel)

Compute the matrix of a partial differential equation (or differential operator) with a given kernel. The matrix is defined as
```math
    A_\mathcal{L} = \begin{pmatrix}\tilde A_\mathcal{L}\\\tilde A\end{pmatrix},
```
where ``\tilde A_\mathcal{L}`` is the matrix of the differential operator (defined by the `equations`) for the inner nodes ``x_i``:
```math
    (\tilde A_\mathcal{L})_{ij} = \mathcal{L}K(x_i, \xi_j),
```
and ``\tilde A`` is the kernel matrix for the boundary nodes:
```math
    \tilde A_{ij} = K(x_i, \xi_j),
```
where ``\mathcal{L}`` is the differential operator (defined by the `equations`), ``K`` the `kernel`, ``x_i`` are the nodes
in `nodeset_boundary` and ``\xi_j`` are the `centers`. By default, `centers` is set to `merge(nodeset_inner, nodeset_boundary)`.

See also [`pde_matrix`](@ref) and [`kernel_matrix`](@ref).
"""
function pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, centers,
                             kernel)
    pd_matrix = pde_matrix(diff_op_or_pde, nodeset_inner, centers, kernel)
    b_matrix = kernel_matrix(centers, nodeset_boundary, kernel)
    return [pd_matrix
            b_matrix]
end

function pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, kernel)
    pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary,
                        merge(nodeset_inner, nodeset_boundary), kernel)
end

@doc raw"""
    operator_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, kernel)

Compute the operator matrix ``L`` discretizing ``\mathcal{L}`` for a given kernel. The operator matrix is defined as
```math
    L = A_\mathcal{L} A^{-1},
```
where ``A_\mathcal{L}`` is the matrix of the differential operator (defined by the `equations`), and ``A`` the kernel matrix.

See also [`pde_boundary_matrix`](@ref) and [`kernel_matrix`](@ref).
"""
function operator_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, kernel)
    nodeset = merge(nodeset_inner, nodeset_boundary)
    A = kernel_matrix(nodeset, kernel)
    A_L = pde_boundary_matrix(diff_op_or_pde, nodeset_inner, nodeset_boundary, kernel)
    return A_L / A
end
