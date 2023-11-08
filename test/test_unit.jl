module TestUnit

using Test
using KernelInterpolation
using Distances
using LinearAlgebra: norm, Cholesky
using StaticArrays: SVector
using Plots

@testset "Unit tests" begin
    @testset "util" begin
        @test_nowarn get_examples()
        @test_nowarn include_example(default_example(), n = 10)
    end

    @testset "Kernels" begin
        x = [3.1, 3.0]
        y = [pi, 2.7]

        k1 = @test_nowarn GaussKernel(2.0)
        @test_nowarn print(k1)
        @test_nowarn display(k1)
        @test metric(k1) isa Euclidean
        @test isapprox(phi(k1, 0.5), 0.36787944117144233)
        @test isapprox(k1(x, y), 0.6928652138413648)

        k2 = @test_nowarn GaussKernel(2.0, metric = Cityblock())
        @test_nowarn print(k2)
        @test_nowarn display(k2)
        @test metric(k2) isa Cityblock
        @test isapprox(phi(k2, 0.5), 0.36787944117144233)
        @test isapprox(k2(x, y), 0.6270417435402862)
    end

    @testset "NodeSet" begin
        nodeset1 = @test_nowarn NodeSet([0.0 0.0
                                         1.0 0.0
                                         0.0 1.0
                                         1.0 1.0])
        @test_nowarn print(nodeset1)
        @test_nowarn display(nodeset1)
        @test eltype(nodeset1) == Float64
        @test dim(nodeset1) == 2
        @test length(nodeset1) == 4
        @test size(nodeset1) == (4, 2)
        for node in nodeset1
            @test node isa SVector{2, Float64}
        end
        f(x) = x[1] + x[2]
        ff = @test_nowarn f.(nodeset1)
        @test ff == [0.0, 1.0, 1.0, 2.0]
        dim1 = @test_nowarn values_along_dim(nodeset1, 1)
        @test dim1 == [0.0, 1.0, 0.0, 1.0]
        dim2 = @test_nowarn values_along_dim(nodeset1, 2)
        @test dim2 == [0.0, 0.0, 1.0, 1.0]

        nodeset2 = @test_nowarn NodeSet([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        @test dim(nodeset2) == 2
        @test length(nodeset2) == 4
        @test size(nodeset2) == (4, 2)
        for i in 1:length(nodeset1)
            @test nodeset1[i] == nodeset2[i]
        end

        @test_nowarn push!(nodeset1, [2.0, 2.0])
        @test length(nodeset1) == 5
        @test nodeset1[5] == [2.0, 2.0]

        nodeset3 = @test_nowarn similar(nodeset1)
        @test nodeset3 isa NodeSet{2, Float64}
        @test eltype(nodeset3) == Float64
        @test length(nodeset3) == 5
        @test_nowarn nodeset3[1] = [1.0, 2.0]
        @test nodeset3[1] == [1.0, 2.0]
        nodeset4 = @test_nowarn similar(nodeset1, Int64)
        @test nodeset4 isa NodeSet{2, Int64}
        nodeset5 = @test_nowarn similar(nodeset1, 10)
        @test nodeset5 isa NodeSet{2, Float64}
        @test length(nodeset5) == 10
        nodeset6 = @test_nowarn similar(nodeset1, Int64, 10)
        @test nodeset6 isa NodeSet{2, Int64}
        @test length(nodeset6) == 10

        nodeset7 = @test_nowarn NodeSet(1:4)
        @test eltype(nodeset7) == Int64
        @test dim(nodeset7) == 1
        @test length(nodeset7) == 4
        @test size(nodeset7) == (4, 1)

        @test_nowarn merge!(nodeset1, nodeset2)
        @test length(nodeset1) == 9
        @test nodeset1[end] == nodeset2[end]
        @test_nowarn merge!(nodeset1, nodeset2, nodeset2)
        @test length(nodeset1) == 17
        @test nodeset1[end] == nodeset2[end]
        nodeset8 = @test_nowarn merge(nodeset2, nodeset3)
        @test nodeset8 isa NodeSet{2, Float64}
        @test length(nodeset8) == 9

        x_min = (-2, -1, 4)
        x_max = (-1, 4, 6)
        nodeset9 = @test_nowarn random_hypercube(10, 3, x_min, x_max)
        for node in nodeset9
            for (i, val) in enumerate(node)
                @test x_min[i] <= val <= x_max[i]
            end
        end
        nodeset10 = @test_nowarn homogeneous_hypercube(3, 2, (-2, 1), (1, 3))
        expected_nodes = [
            [-2.0, 1.0],
            [-0.5, 1.0],
            [1.0, 1.0],
            [-2.0, 2.0],
            [-0.5, 2.0],
            [1.0, 2.0],
            [-2.0, 3.0],
            [-0.5, 3.0],
            [1.0, 3.0],
        ]
        @test nodeset10 isa NodeSet{2, Float64}
        @test length(nodeset10) == length(expected_nodes)
        for i in 1:length(nodeset10)
            @test nodeset10[i] == expected_nodes[i]
        end
        r = 2.0
        center = [-1.0, 3.0, 2.0, -pi]
        nodeset11 = @test_nowarn random_hypersphere(50, 4, r, center)
        @test nodeset11 isa NodeSet{4, Float64}
        for node in nodeset11
            @test norm(node .- center) <= r
        end
        nodeset12 = @test_nowarn random_hypersphere_boundary(50, 4, r, center)
        @test nodeset12 isa NodeSet{4, Float64}
        for node in nodeset12
            @test isapprox(norm(node .- center), r)
        end
    end

    @testset "Interpolation" begin
        nodes = NodeSet([0.0 0.0
                         1.0 0.0
                         0.0 1.0
                         1.0 1.0])
        f(x) = x[1] + x[2]
        ff = f.(nodes)
        k = GaussKernel(0.5)
        itp = @test_nowarn interpolate(nodes, ff, k)
        @test_nowarn print(itp)
        @test_nowarn display(itp)
        @test kernel(itp) == k
        @test nodeset(itp) == nodes
        expected_coefficients = [
            -2.225451664388596,
            0.31604241814819756,
            0.31604241814819745,
            2.857536500685,
        ]
        coeffs = coefficients(itp)
        @test length(coeffs) == length(expected_coefficients)
        for i in 1:length(coeffs)
            @test isapprox(coeffs[i], expected_coefficients[i])
        end
        @test kernel_matrix(itp) isa Cholesky
        @test isapprox(itp([0.5, 0.5]), 1.115625820404527)
    end

    @testset "Visualization" begin
        f = sum
        for dim in 1:3
            nodes = homogeneous_hypercube(5, dim)
            @test_nowarn plot(nodes)
            if dim < 3
                ff = f.(nodes)
                itp = interpolate(nodes, ff)
                nodes_fine = homogeneous_hypercube(10, dim)
                @test_nowarn plot(nodes_fine, itp)
            end
        end
    end
end

end # module
