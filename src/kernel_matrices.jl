@doc raw"""
    kernel_matrix(nodeset1, nodeset2, kernel)

Return the kernel matrix for the nodeset and kernel. The kernel matrix is defined as
```math
    A_{ij} = K(x_i, \xi_j),
```
where ``x_i`` are the nodes in the `nodeset1`, ``\xi_j`` are the nodes in the `nodeset2`,
and ``K`` the `kernel`.
"""
function kernel_matrix(nodeset1, nodeset2, kernel)
    n = length(nodeset1)
    m = length(nodeset2)
    A = Matrix{eltype(nodeset1)}(undef, n, m)
    for i in 1:n
        for j in 1:m
            A[i, j] = kernel(nodeset1[i], nodeset2[j])
        end
    end
    return A
end

"""
    kernel_matrix(nodeset, kernel)

Return the kernel matrix for the nodeset and kernel. The kernel matrix is defined as
```math
    A_{ij} = K(x_i, x_j),
```
where ``x_i`` are the nodes in the `nodeset` and ``K`` the `kernel`.
"""
function kernel_matrix(nodeset, kernel)
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
function polynomial_matrix(nodeset, ps)
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

"""
    interpolation_matrix(nodeset, kernel, ps, regularization)

Return the interpolation matrix for the `nodeset`, `kernel`, polynomials `ps`, and `regularization`.
The interpolation matrix is defined as
```math
    A = \begin{pmatrix}K & P\\P^T & 0\end{pmatrix},
```
where ``K`` is the [`regularize!`](@ref)d [`kernel_matrix`](@ref) and ``P`` the [`polynomial_matrix`](@ref)`.
"""
function interpolation_matrix(nodeset, kernel, ps, regularization)
    q = length(ps)
    k_matrix = kernel_matrix(nodeset, kernel)
    regularize!(k_matrix, regularization)
    p_matrix = polynomial_matrix(nodeset, ps)
    system_matrix = [k_matrix p_matrix
                     p_matrix' zeros(q, q)]
    return Symmetric(system_matrix)
end

"""
    least_squares_matrix(nodeset, centers, kernel, ps, regularization)

Return the least squares matrix for the `nodeset`, `centers`, `kernel`, polynomials `ps`, and `regularization`.
The least squares matrix is defined as
```math
    A = \begin{pmatrix}K & P_1\\P_2' & 0\end{pmatrix},
```
where ``K`` is the [`regularize!`](@ref)d [`kernel_matrix`](@ref), ``P_1`` the [`polynomial_matrix`](@ref)`
for the `nodeset` and ``P_2`` the [`polynomial_matrix`](@ref)` for the `centers`.
"""
function least_squares_matrix(nodeset, centers, kernel, ps, regularization)
    q = length(ps)
    k_matrix = kernel_matrix(nodeset, centers, kernel)
    regularize!(k_matrix, regularization)
    p_matrix1 = polynomial_matrix(nodeset, ps)
    p_matrix2 = polynomial_matrix(centers, ps)
    system_matrix = [k_matrix p_matrix1
                     p_matrix2' zeros(q, q)]
    return system_matrix
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
    b_matrix = kernel_matrix(nodeset_boundary, centers, kernel)
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
