 using KernelInterpolation

 f(x, equations) = -exp(x[1])
 pde = PoissonEquation(f)
 u(x, equations) = exp(x[1])
 n = 20
 nodeset_inner = homogeneous_hypercube(n, 0.01, 0.99; dim = 1)
 nodeset_boundary = NodeSet([[0.0], [1.0]])
 g(x) = u(x, pde)
 kernel = WendlandKernel{1}(3)
 sd = SpatialDiscretization(pde, nodeset_inner, g, nodeset_boundary, kernel)
 itp = solve_stationary(sd)
