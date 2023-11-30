module TestUnit

using Test
using KernelInterpolation
using LinearAlgebra: norm, Cholesky, BunchKaufman
using StaticArrays: MVector
using Plots

@testset "Unit tests" begin
    @testset "util" begin
        @test_nowarn get_examples()
        @test_nowarn include_example(default_example(), n = 10)
    end

    @testset "Kernels" begin
        x = [3.1, 3.0]
        y = [pi, 2.7]

        kernel1 = @test_nowarn GaussKernel{2}(shape_parameter = 2.0)
        @test_nowarn println(kernel1)
        @test_nowarn display(kernel1)
        @test dim(kernel1) == 2
        @test order(kernel1) == 0
        @test isapprox(phi(kernel1, 0.5), 0.36787944117144233)
        @test isapprox(kernel1(x, y), 0.6928652138413648)

        kernel2 = @test_nowarn MultiquadricKernel{2}()
        @test_nowarn println(kernel2)
        @test_nowarn display(kernel2)
        @test order(kernel2) == 1
        @test isapprox(phi(kernel2, 0.5), 1.118033988749895)
        @test isapprox(kernel2(x, y), 1.0448588176555913)

        kernel3 = @test_nowarn InverseMultiquadricKernel{2}()
        @test_nowarn println(kernel3)
        @test_nowarn display(kernel3)
        @test order(kernel3) == 0
        @test isapprox(phi(kernel3, 0.5), 0.8944271909999159)
        @test isapprox(kernel3(x, y), 0.9570671014135252)

        kernel4 = @test_nowarn PolyharmonicSplineKernel{2}(3)
        @test_nowarn println(kernel4)
        @test_nowarn display(kernel4)
        @test order(kernel4) == 2
        @test isapprox(phi(kernel4, 0.5), 0.125)
        @test isapprox(kernel4(x, y), 0.02778220597956396)

        kernel5 = @test_nowarn ThinPlateSplineKernel{2}()
        @test_nowarn println(kernel5)
        @test_nowarn display(kernel5)
        @test order(kernel5) == 2
        @test isapprox(phi(kernel5, 0.5), -0.17328679513998632)
        @test isapprox(kernel5(x, y), -0.10956712895893082)
        kernel5_1 = @test_nowarn PolyharmonicSplineKernel{2}(2)
        @test isapprox(phi(kernel5, 0.5), phi(kernel5_1, 0.5))
        @test isapprox(kernel5(x, y), kernel5_1(x, y))

        expected_values = [0.25, 0.1875, 0.10807291666666666, 0.0595703125]
        expected_differences = [
            0.48599089995881917,
            0.5223227199041456,
            0.44621444895933693,
            0.36846891545136595,
        ]
        for k in 0:3
            kernel6 = @test_nowarn WendlandKernel{2}(k)
            @test_nowarn println(kernel6)
            @test_nowarn display(kernel6)
            @test order(kernel6) == 0
            @test isapprox(phi(kernel6, 0.5), expected_values[k + 1])
            @test isapprox(kernel6(x, y), expected_differences[k + 1])
        end

        kernel7 = @test_nowarn RadialCharacteristicKernel{2}()
        @test_nowarn println(kernel7)
        @test_nowarn display(kernel7)
        @test order(kernel7) == 0
        @test isapprox(phi(kernel7, 0.5), 0.25)
        @test isapprox(kernel7(x, y), 0.48599089995881917)
        kernel7_1 = @test_nowarn WendlandKernel{2}(0)
        @test isapprox(phi(kernel7, 0.5), phi(kernel7_1, 0.5))
        @test isapprox(kernel7(x, y), kernel7_1(x, y))

        nus = [0.5, 1.5, 2.5, 3.5]
        kernels = [Matern12Kernel, Matern32Kernel, Matern52Kernel, Matern72Kernel]
        expected_values = [
            0.6065306597126335,
            0.7848876539574506,
            0.8286491424181255,
            0.8463080665533403,
        ]
        expected_differences = [
            0.7386954717906608,
            0.9022506660348356,
            0.9297353942237024,
            0.9389666269913006,
        ]
        for i in 1:length(nus)
            kernel8 = @test_nowarn MaternKernel{2}(nus[i])
            @test_nowarn println(kernel8)
            @test_nowarn display(kernel8)
            @test order(kernel8) == 0
            @test isapprox(phi(kernel8, 0.5), expected_values[i])
            @test isapprox(kernel8(x, y), expected_differences[i])

            kernel8_1 = @test_nowarn kernels[i]{2}()
            @test_nowarn println(kernel8_1)
            @test_nowarn display(kernel8_1)
            @test order(kernel8_1) == 0

            @test isapprox(phi(kernel8, 0.5), phi(kernel8_1, 0.5))
            @test isapprox(kernel8(x, y), kernel8_1(x, y))
        end
    end

    @testset "NodeSet" begin
        nodeset1 = @test_nowarn NodeSet([0.0 0.0
                                         1.0 0.0
                                         0.0 1.0
                                         1.0 1.0])
        @test_nowarn println(nodeset1)
        @test_nowarn display(nodeset1)
        @test eltype(nodeset1) == Float64
        @test isapprox(separation_distance(nodeset1), 0.5)
        @test dim(nodeset1) == 2
        @test length(nodeset1) == 4
        @test size(nodeset1) == (4, 2)
        for node in nodeset1
            @test node isa MVector{2, Float64}
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

        @test_nowarn push!(nodeset1, [1.1, 1.3])
        @test length(nodeset1) == 5
        @test nodeset1[5] == [1.1, 1.3]
        @test isapprox(separation_distance(nodeset1), 0.158113883008419)

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
        @test isapprox(separation_distance(nodeset7), 0.5)

        @test_nowarn merge!(nodeset1, nodeset2)
        @test isapprox(separation_distance(nodeset1), 0.0)
        @test isapprox(separation_distance(nodeset2), 0.5)
        @test length(nodeset1) == 9
        @test nodeset1[end] == nodeset2[end]
        @test_nowarn merge!(nodeset1, nodeset2, nodeset2)
        @test length(nodeset1) == 17
        @test nodeset1[end] == nodeset2[end]
        nodeset8 = @test_nowarn merge(nodeset2, nodeset3)
        @test nodeset8 isa NodeSet{2, Float64}
        @test length(nodeset8) == 9
        @test length(unique(nodeset1)) == 5
        @test_nowarn unique!(nodeset1)
        @test isapprox(separation_distance(nodeset1), 0.158113883008419)
        @test length(nodeset1) == 5
        @test_nowarn deleteat!(nodeset1, 2)
        expected_nodes = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.1, 1.3],
        ]
        @test length(nodeset1) == length(expected_nodes)
        for i in 1:length(nodeset1)
            @test nodeset1[i] == expected_nodes[i]
        end
        @test_nowarn deleteat!(nodeset1, [2, 4])
        expected_nodes = [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
        @test length(nodeset1) == length(expected_nodes)
        for i in 1:length(nodeset1)
            @test nodeset1[i] == expected_nodes[i]
        end

        x_min = (-2, -1, 4)
        x_max = (-1, 4, 6)
        nodeset9 = @test_nowarn random_hypercube(10, 3, x_min, x_max)
        @test nodeset9 isa NodeSet{3, Float64}
        for node in nodeset9
            for (i, val) in enumerate(node)
                @test x_min[i] <= val <= x_max[i]
            end
        end

        nodeset10 = @test_nowarn random_hypercube_boundary(10, 3, x_min, x_max)
        @test nodeset10 isa NodeSet{3, Float64}
        for node in nodeset10
            on_boundary = false
            for (i, val) in enumerate(node)
                @test x_min[i] <= val <= x_max[i]
                if isapprox(val, x_min[i]) || isapprox(val, x_max[i])
                    on_boundary = true
                end
            end
            @test on_boundary
        end

        nodeset11 = @test_nowarn homogeneous_hypercube(3, 2, (-2, 1), (1, 3))
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
        @test nodeset11 isa NodeSet{2, Float64}
        @test isapprox(separation_distance(nodeset11), 0.5)
        @test length(nodeset11) == length(expected_nodes)
        for i in 1:length(nodeset11)
            @test nodeset11[i] == expected_nodes[i]
        end

        nodeset12 = @test_nowarn homogeneous_hypercube_boundary(3, 2, (-2, 1), (1, 3))
        expected_nodes = [
            [-2.0, 1.0],
            [-2.0, 2.0],
            [-2.0, 3.0],
            [-0.5, 1.0],
            [-0.5, 3.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
        @test nodeset12 isa NodeSet{2, Float64}
        @test isapprox(separation_distance(nodeset12), 0.5)
        @test length(nodeset12) == length(expected_nodes)
        display(nodeset12)
        for i in 1:length(nodeset12)
            @test nodeset12[i] == expected_nodes[i]
        end

        r = 2.0
        center = [-1.0, 3.0, 2.0, -pi]
        nodeset13 = @test_nowarn random_hypersphere(50, 4, r, center)
        @test nodeset13 isa NodeSet{4, Float64}
        for node in nodeset13
            @test norm(node .- center) <= r
        end

        nodeset14 = @test_nowarn random_hypersphere_boundary(50, 4, r, center)
        @test nodeset14 isa NodeSet{4, Float64}
        for node in nodeset14
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
        kernel = GaussKernel{dim(nodes)}(shape_parameter = 0.5)
        itp = @test_nowarn interpolate(nodes, ff, kernel)
        @test_nowarn println(itp)
        @test_nowarn display(itp)
        @test interpolation_kernel(itp) == kernel
        @test nodeset(itp) == nodes
        @test dim(itp) == dim(kernel)
        @test dim(itp) == dim(nodes)
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
        @test length(kernel_coefficients(itp)) == length(coeffs)
        @test length(polynomial_coefficients(itp)) == 0
        @test order(itp) == 0
        @test length(polynomial_basis(itp)) == 0
        @test length(polyvars(itp)) == dim(itp)
        @test system_matrix(itp) isa Cholesky
        @test isapprox(itp([0.5, 0.5]), 1.115625820404527)
        kernel = ThinPlateSplineKernel{dim(nodes)}()
        itp = @test_nowarn interpolate(nodes, ff, kernel)
        expected_coefficients = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ]
        coeffs = coefficients(itp)
        @test length(coeffs) == length(expected_coefficients)
        for i in 1:length(coeffs)
            @test isapprox(coeffs[i], expected_coefficients[i])
        end
        @test order(itp) == order(kernel)
        @test length(kernel_coefficients(itp)) == length(nodes)
        @test length(polynomial_coefficients(itp)) == order(itp) + 1
        @test length(polynomial_basis(itp)) ==
              binomial(order(itp) - 1 + dim(nodes), dim(nodes))
        @test system_matrix(itp) isa BunchKaufman
        @test isapprox(itp([0.5, 0.5]), 1.0)
        # TODO: test convergence orders of condition numbers depending on separation distance
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
        k = GaussKernel{3}(shape_parameter = 0.5)
        @test_nowarn plot(0.1:0.1:5.0, k)
    end
end

end # module
