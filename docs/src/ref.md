# KernelInterpolation.jl API

```@meta
CurrentModule = KernelInterpolation
```

## Kernel functions

```@docs
AbstractKernel
get_name
RadialSymmetricKernel
Phi
phi
GaussKernel
MultiquadricKernel
InverseMultiquadricKernel
PolyharmonicSplineKernel
ThinPlateSplineKernel
WendlandKernel
WuKernel
RadialCharacteristicKernel
MaternKernel
Matern12Kernel
Matern32Kernel
Matern52Kernel
Matern72Kernel
RieszKernel
SumKernel
ProductKernel
TransformationKernel
```

## Node sets

```@docs
NodeSet
empty_nodeset
separation_distance
values_along_dim
distance_matrix
random_hypercube
random_hypercube_boundary
homogeneous_hypercube
homogeneous_hypercube_boundary
random_hypersphere
random_hypersphere_boundary
```

## Interpolation

```@docs
Interpolation
TemporalInterpolation
interpolate
kernel_inner_product
kernel_norm
dim
interpolation_kernel
nodeset
coefficients
kernel_coefficients
polynomial_coefficients
polynomial_basis
polyvars
order
system_matrix
```

## [Differential Operators](@id api-diffops)

```@docs
PartialDerivative
Gradient
Laplacian
EllipticOperator
```

## Stationary partial differential equations

```@docs
PoissonEquation
EllipticEquation
```

## Time-dependent partial differential equations

```@docs
AdvectionEquation
HeatEquation
AdvectionDiffusionEquation
```

## Discretization

```@docs
SpatialDiscretization
solve_stationary
Semidiscretization
semidiscretize
```

## Kernel matrices

```@docs
kernel_matrix
polynomial_matrix
pde_matrix
pde_boundary_matrix
operator_matrix
```

## Callbacks

```@docs
AliveCallback
SummaryCallback
SaveSolutionCallback
```

## Input/Output

```@docs
vtk_read
vtk_save
add_to_pvd
```

## Utilities

```@docs
examples_dir
get_example
default_example
```
