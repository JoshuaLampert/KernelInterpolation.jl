@testitem "util" setup=[Setup] begin
    @test_nowarn get_examples()
    @trixi_test_nowarn trixi_include(default_example(), n = 10)
end

@testitem "Kernels" setup=[Setup, AdditionalImports] begin
    x = [3.1, 3.0]
    y = [pi, 2.7]

    kernel1 = @test_nowarn GaussKernel{2}(shape_parameter = 2.0)
    @test_nowarn println(kernel1)
    @test_nowarn display(kernel1)
    @test_nowarn get_name(kernel1)
    @test dim(kernel1) == 2
    @test order(kernel1) == 0
    @test isapprox(phi(kernel1, 0.0), 1.0)
    @test isapprox(phi(kernel1, 0.5), 0.36787944117144233)
    @test isapprox(kernel1(x, y), 0.6928652138413648)

    kernel2 = @test_nowarn MultiquadricKernel{2}()
    @test_nowarn println(kernel2)
    @test_nowarn display(kernel2)
    @test_nowarn get_name(kernel2)
    @test order(kernel2) == 1
    @test isapprox(phi(kernel2, 0.0), 1.0)
    @test isapprox(phi(kernel2, 0.5), 1.118033988749895)
    @test isapprox(kernel2(x, y), 1.0448588176555913)

    kernel3 = @test_nowarn InverseMultiquadricKernel{2}()
    @test_nowarn println(kernel3)
    @test_nowarn display(kernel3)
    @test_nowarn get_name(kernel3)
    @test order(kernel3) == 0
    @test isapprox(phi(kernel3, 0.0), 1.0)
    @test isapprox(phi(kernel3, 0.5), 0.8944271909999159)
    @test isapprox(kernel3(x, y), 0.9570671014135252)

    kernel4 = @test_nowarn PolyharmonicSplineKernel{2}(3)
    @test_nowarn println(kernel4)
    @test_nowarn display(kernel4)
    @test_nowarn get_name(kernel4)
    @test order(kernel4) == 2
    @test isapprox(phi(kernel4, 0.5), 0.125)
    @test isapprox(kernel4(x, y), 0.02778220597956396)

    kernel5 = @test_nowarn ThinPlateSplineKernel{2}()
    @test_nowarn println(kernel5)
    @test_nowarn display(kernel5)
    @test_nowarn get_name(kernel5)
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
        0.36846891545136595]
    for k in 0:3
        kernel6 = @test_nowarn WendlandKernel{2}(k)
        @test_nowarn println(kernel6)
        @test_nowarn display(kernel6)
        @test_nowarn get_name(kernel6)
        @test order(kernel6) == 0
        @test isapprox(phi(kernel6, 0.0), 1.0)
        @test isapprox(phi(kernel6, 1.1), 0.0)
        @test isapprox(phi(kernel6, 0.5), expected_values[k + 1])
        @test isapprox(kernel6(x, y), expected_differences[k + 1])
    end

    expected_values = [
        0.5,
        0.34375,
        0.3125,
        0.201171875,
        0.240234375,
        0.20703125,
        0.1150146484375,
        0.14461263020833331,
        0.169677734375,
        0.14111328125
    ]
    expected_differences = [
        0.6971304755630894,
        0.6777128254016545,
        0.5595868163344161,
        0.5741858708746038,
        0.5922403769292889,
        0.46589172653038635,
        0.47839230403264094,
        0.5106135476269533,
        0.5197783607481103,
        0.39497468985502254
    ]

    for l in 0:3
        for k in 0:l
            kernel6_1 = @test_nowarn WuKernel{2}(l, k)
            @test_nowarn println(kernel6_1)
            @test_nowarn display(kernel6_1)
            @test_nowarn get_name(kernel6_1)
            @test order(kernel6_1) == 0
            @test isapprox(phi(kernel6_1, 0.0), 1.0)
            @test isapprox(phi(kernel6_1, 1.1), 0.0)
            i = div(l * (l + 1), 2) + k + 1
            @test isapprox(phi(kernel6_1, 0.5), expected_values[i])
            @test isapprox(kernel6_1(x, y), expected_differences[i])
        end
    end

    kernel7 = @test_nowarn RadialCharacteristicKernel{2}()
    @test_nowarn println(kernel7)
    @test_nowarn display(kernel7)
    @test_nowarn get_name(kernel7)
    @test order(kernel7) == 0
    @test isapprox(phi(kernel7, 0.0), 1.0)
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
        0.8463080665533403]
    expected_differences = [
        0.7386954717906608,
        0.9022506660348356,
        0.9297353942237024,
        0.9389666269913006]
    for i in eachindex(nus)
        kernel8 = @test_nowarn MaternKernel{2}(nus[i])
        @test_nowarn println(kernel8)
        @test_nowarn display(kernel8)
        @test_nowarn get_name(kernel8)
        @test order(kernel8) == 0
        @test isapprox(phi(kernel8, 0.0), 1.0)
        @test isapprox(phi(kernel8, 0.5), expected_values[i])
        @test isapprox(kernel8(x, y), expected_differences[i])

        kernel8_1 = @test_nowarn kernels[i]{2}()
        @test_nowarn println(kernel8_1)
        @test_nowarn display(kernel8_1)
        @test_nowarn get_name(kernel8_1)
        @test order(kernel8_1) == 0

        @test isapprox(phi(kernel8, 0.5), phi(kernel8_1, 0.5))
        @test isapprox(kernel8(x, y), kernel8_1(x, y))
    end

    kernel9 = @test_nowarn RieszKernel{2}(1.1)
    @test_nowarn println(kernel9)
    @test_nowarn display(kernel9)
    @test_nowarn get_name(kernel9)
    @test order(kernel9) == 1
    @test isapprox(phi(kernel9, 0.0), 0.0)
    @test isapprox(phi(kernel9, 0.5), -0.4665164957684037)
    @test isapprox(kernel9(x, y), -0.26877021157823217)

    trafo(x) = [x[1] + x[2]^2 + 2 * x[3] * x[2], x[3] - x[1]]
    kernel10 = @test_nowarn TransformationKernel{3}(kernel1, trafo)
    @test_nowarn println(kernel10)
    @test_nowarn display(kernel10)
    @test_nowarn get_name(kernel10)
    @test order(kernel10) == 0
    x3 = [-1.0, 2.0, pi / 8]
    y3 = [2.3, 4.2, -12.3]
    @test isapprox(kernel10(x3, y3), kernel1(trafo(x3), trafo(y3)))

    kernel11 = @test_nowarn ProductKernel{2}([kernel1, kernel2])
    @test_nowarn println(kernel11)
    @test_nowarn display(kernel11)
    @test_nowarn get_name(kernel11)
    @test order(kernel11) == 1
    @test isapprox(kernel11(x, y), kernel1(x, y) * kernel2(x, y))
    @test isapprox((kernel1 * kernel2)(x, y), kernel1(x, y) * kernel2(x, y))

    kernel12 = @test_nowarn SumKernel{2}([kernel1, kernel2])
    @test_nowarn println(kernel12)
    @test_nowarn display(kernel12)
    @test_nowarn get_name(kernel12)
    @test order(kernel12) == 0
    @test isapprox(kernel12(x, y), kernel1(x, y) + kernel2(x, y))
    @test isapprox((kernel1 + kernel2)(x, y), kernel1(x, y) + kernel2(x, y))

    # Test evaluating 1D kernels at a scalar
    kernel13 = @test_nowarn GaussKernel{1}(shape_parameter = 2.0)
    @test kernel13(3.1, 3.0) == kernel13(3.1, SVector(3.0)) == kernel13([3.1], 3.0)
end

@testitem "NodeSet" setup=[Setup, AdditionalImports] begin
    nodeset1 = @test_nowarn NodeSet([0.0 0.0
                                     1.0 0.0
                                     0.0 1.0
                                     1.0 1.0])
    @test NodeSet(nodeset1) == nodeset1
    @test_nowarn println(nodeset1)
    @test_nowarn display(nodeset1)
    @test eltype(nodeset1) == Float64
    @test isapprox(separation_distance(nodeset1), 0.5)
    @test dim(nodeset1) == 2
    @test length(nodeset1) == 4
    @test size(nodeset1) == (4, 2)
    @test axes(nodeset1) == (1:4,)
    @test eachindex(nodeset1) == firstindex(nodeset1):lastindex(nodeset1)
    @test keys(nodeset1) == 1:4
    for node in nodeset1
        @test node isa MVector{2, Float64}
    end

    f(x) = x[1] + x[2]
    ff = @test_nowarn f.(nodeset1)
    @test ff == [0.0, 1.0, 1.0, 2.0]
    dim1 = @test_nowarn nodeset1[:, 1]
    @test dim1 == [0.0, 1.0, 0.0, 1.0]
    dim2 = @test_nowarn nodeset1[:, 2]
    @test dim2 == [0.0, 0.0, 1.0, 1.0]
    @test distance_matrix(nodeset1, nodeset1) == [0.0 1.0 1.0 1.4142135623730951
           1.0 0.0 1.4142135623730951 1.0
           1.0 1.4142135623730951 0.0 1.0
           1.4142135623730951 1.0 1.0 0.0]

    # Saving the nodeset to a VTK file
    @test_nowarn vtk_save("nodeset1", nodeset1)
    nodeset1_2, point_data = @test_nowarn vtk_read("nodeset1.vtu")
    @test length(nodeset1_2) == length(nodeset1)
    for i in eachindex(nodeset1)
        @test [nodeset1[i]; 0.0] == nodeset1_2[i]
    end
    @test length(point_data) == 0
    @test_nowarn rm("nodeset1.vtu", force = true)
    nodeset1_1 = @test_nowarn empty_nodeset(2, Float64)
    @test length(nodeset1_1) == 0
    @test dim(nodeset1_1) == 2
    @test separation_distance(nodeset1_1) == Inf
    @test_nowarn push!(nodeset1_1, [0.0, 0.0])
    @test length(nodeset1_1) == 1
    @test separation_distance(nodeset1_1) == Inf
    @test_nowarn push!(nodeset1_1, [1.0, 0.0])
    @test separation_distance(nodeset1_1) == 0.5

    nodeset2 = @test_nowarn NodeSet([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    @test dim(nodeset2) == 2
    @test length(nodeset2) == 4
    @test size(nodeset2) == (4, 2)
    for i in eachindex(nodeset1)
        @test nodeset1[i] == nodeset2[i]
    end
    sub_nodes = nodeset1[1:2]
    @test length(sub_nodes) == 2
    @test sub_nodes isa Vector{MVector{2, Float64}}
    for i in eachindex(sub_nodes)
        @test nodeset1[i] == sub_nodes[i]
    end

    @test_nowarn push!(nodeset1, [1.1, 1.3])
    @test length(nodeset1) == 5
    @test nodeset1[5] == [1.1, 1.3]
    @test isapprox(separation_distance(nodeset1), 0.158113883008419)
    @test_nowarn push!(nodeset1, MVector{2}([1.1, 1.4]))
    @test length(nodeset1) == 6
    @test nodeset1[6] == [1.1, 1.4]
    @test isapprox(separation_distance(nodeset1), 0.05)
    @test_nowarn pop!(nodeset1)
    @test length(nodeset1) == 5
    @test nodeset1[5] == [1.1, 1.3]
    @test isapprox(separation_distance(nodeset1), 0.158113883008419)
    @test distance_matrix(nodeset1, nodeset2) == [0.0 1.0 1.0 1.4142135623730951
           1.0 0.0 1.4142135623730951 1.0
           1.0 1.4142135623730951 0.0 1.0
           1.4142135623730951 1.0 1.0 0.0
           1.7029386365926402 1.3038404810405297 1.140175425099138 0.316227766016838]

    nodeset3 = @test_nowarn similar(nodeset1)
    @test nodeset3 isa NodeSet{2, Float64}
    @test eltype(nodeset3) == Float64
    @test length(nodeset3) == 5
    @test_nowarn nodeset3[1] = [1.0, 2.0]
    @test nodeset3[1] == [1.0, 2.0]
    @test_nowarn nodeset3[1] = MVector{2}([2.0, 3.0])
    @test_nowarn nodeset3[1] = [2.0, 3.0]
    nodeset4 = @test_nowarn similar(nodeset1, Float32)
    @test nodeset4 isa NodeSet{2, Float32}
    nodeset5 = @test_nowarn similar(nodeset1, 10)
    @test nodeset5 isa NodeSet{2, Float64}
    @test length(nodeset5) == 10
    nodeset6 = @test_nowarn similar(nodeset1, Float32, 10)
    @test nodeset6 isa NodeSet{2, Float32}
    @test length(nodeset6) == 10

    # Integer nodes should be converted to float by design
    nodeset7 = @test_nowarn NodeSet(1:4)
    @test eltype(nodeset7) == Float64
    @test dim(nodeset7) == 1
    @test length(nodeset7) == 4
    @test size(nodeset7) == (4, 1)
    @test isapprox(separation_distance(nodeset7), 0.5)
    @test_nowarn nodeset7[2] = 4
    @test nodeset7[2] == [4]
    @test isapprox(separation_distance(nodeset7), 0.0)

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
        [1.1, 1.3]]
    @test length(nodeset1) == length(expected_nodes)
    for i in eachindex(nodeset1)
        @test nodeset1[i] == expected_nodes[i]
    end
    @test_nowarn deleteat!(nodeset1, [2, 4])
    expected_nodes = [
        [0.0, 0.0],
        [1.0, 1.0]]
    @test length(nodeset1) == length(expected_nodes)
    for i in eachindex(nodeset1)
        @test nodeset1[i] == expected_nodes[i]
    end

    x_min = -1
    x_max = 1
    nodeset9 = @test_nowarn random_hypercube(10, x_min, x_max; dim = 3)
    @test nodeset9 isa NodeSet{3, Float64}
    for node in nodeset9
        for (i, val) in enumerate(node)
            @test x_min <= val <= x_max
        end
    end

    @test length(random_hypercube_boundary(10, x_min, x_max; dim = 1)) == 2
    nodeset10 = @test_nowarn random_hypercube_boundary(10, x_min, x_max; dim = 3)
    @test nodeset10 isa NodeSet{3, Float64}
    for node in nodeset10
        on_boundary = false
        for (i, val) in enumerate(node)
            @test x_min <= val <= x_max
            if isapprox(val, x_min) || isapprox(val, x_max)
                on_boundary = true
            end
        end
        @test on_boundary
    end

    x_min = (-2, -1, 4)
    x_max = (-1, 4, 6)
    nodeset9_1 = @test_nowarn random_hypercube(10, x_min, x_max)
    @test nodeset9_1 isa NodeSet{3, Float64}
    for node in nodeset9_1
        for (i, val) in enumerate(node)
            @test x_min[i] <= val <= x_max[i]
        end
    end

    nodeset10_1 = @test_nowarn random_hypercube_boundary(10, x_min, x_max)
    @test nodeset10_1 isa NodeSet{3, Float64}
    for node in nodeset10_1
        on_boundary = false
        for (i, val) in enumerate(node)
            @test x_min[i] <= val <= x_max[i]
            if isapprox(val, x_min[i]) || isapprox(val, x_max[i])
                on_boundary = true
            end
        end
        @test on_boundary
    end

    nodeset11 = @test_nowarn homogeneous_hypercube(3, -2, 1; dim = 2)
    expected_nodes = [
        [-2.0, -2.0],
        [-0.5, -2.0],
        [1.0, -2.0],
        [-2.0, -0.5],
        [-0.5, -0.5],
        [1.0, -0.5],
        [-2.0, 1.0],
        [-0.5, 1.0],
        [1.0, 1.0]]
    @test nodeset11 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset11), 0.75)
    @test length(nodeset11) == length(expected_nodes)
    for i in eachindex(nodeset11)
        @test nodeset11[i] == expected_nodes[i]
    end

    nodeset12 = @test_nowarn homogeneous_hypercube_boundary(3, -2, 1; dim = 2)
    expected_nodes = [
        [-2.0, -2.0],
        [-2.0, -0.5],
        [-2.0, 1.0],
        [-0.5, -2.0],
        [-0.5, 1.0],
        [1.0, -2.0],
        [1.0, -0.5],
        [1.0, 1.0]]
    @test nodeset12 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset12), 0.75)
    @test length(nodeset12) == length(expected_nodes)
    for i in eachindex(nodeset12)
        @test nodeset12[i] == expected_nodes[i]
    end

    nodeset11_1 = @test_nowarn homogeneous_hypercube((3, 4), -2, 1)
    expected_nodes = [
        [-2.0, -2.0],
        [-0.5, -2.0],
        [1.0, -2.0],
        [-2.0, -1.0],
        [-0.5, -1.0],
        [1.0, -1.0],
        [-2.0, 0.0],
        [-0.5, 0.0],
        [1.0, 0.0],
        [-2.0, 1.0],
        [-0.5, 1.0],
        [1.0, 1.0]]
    @test nodeset11_1 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset11_1), 0.5)
    @test length(nodeset11_1) == length(expected_nodes)
    for i in eachindex(nodeset11_1)
        @test nodeset11_1[i] == expected_nodes[i]
    end

    nodeset12_1 = @test_nowarn homogeneous_hypercube_boundary((3, 4), -2, 1)
    expected_nodes = [
        [-2.0, -2.0],
        [-2.0, -1.0],
        [-2.0, 0.0],
        [-2.0, 1.0],
        [-0.5, -2.0],
        [-0.5, 1.0],
        [1.0, -2.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0]]
    @test nodeset12_1 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset12_1), 0.5)
    @test length(nodeset12_1) == length(expected_nodes)
    for i in eachindex(nodeset12_1)
        @test nodeset12_1[i] == expected_nodes[i]
    end

    nodeset11_2 = @test_nowarn homogeneous_hypercube(3, (-2, 1), (1, 3))
    expected_nodes = [
        [-2.0, 1.0],
        [-0.5, 1.0],
        [1.0, 1.0],
        [-2.0, 2.0],
        [-0.5, 2.0],
        [1.0, 2.0],
        [-2.0, 3.0],
        [-0.5, 3.0],
        [1.0, 3.0]]
    @test nodeset11_2 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset11_2), 0.5)
    @test length(nodeset11_2) == length(expected_nodes)
    for i in eachindex(nodeset11_2)
        @test nodeset11_2[i] == expected_nodes[i]
    end

    nodeset12_2 = @test_nowarn homogeneous_hypercube_boundary(3, (-2, 1), (1, 3))
    expected_nodes = [
        [-2.0, 1.0],
        [-2.0, 2.0],
        [-2.0, 3.0],
        [-0.5, 1.0],
        [-0.5, 3.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0]]
    @test nodeset12_2 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset12_2), 0.5)
    @test length(nodeset12_2) == length(expected_nodes)
    display(nodeset12_2)
    for i in eachindex(nodeset12_2)
        @test nodeset12_2[i] == expected_nodes[i]
    end

    nodeset11_3 = @test_nowarn homogeneous_hypercube((4, 3), (-2, 1), (1, 3))
    expected_nodes = [
        [-2.0, 1.0],
        [-1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-2.0, 2.0],
        [-1.0, 2.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [-2.0, 3.0],
        [-1.0, 3.0],
        [0.0, 3.0],
        [1.0, 3.0]]
    @test nodeset11_3 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset11_3), 0.5)
    @test length(nodeset11_3) == length(expected_nodes)
    for i in eachindex(nodeset11_3)
        @test nodeset11_3[i] == expected_nodes[i]
    end

    nodeset11_4 = @test_nowarn homogeneous_hypercube((3, 3))
    expected_nodes = [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [1.0, 0.5],
        [0.0, 1.0],
        [0.5, 1.0],
        [1.0, 1.0]]
    @test nodeset11_4 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset11_4), 0.25)
    @test length(nodeset11_4) == length(expected_nodes)
    for i in eachindex(nodeset11_4)
        @test nodeset11_4[i] == expected_nodes[i]
    end

    nodeset12_3 = @test_nowarn homogeneous_hypercube_boundary((4, 3), (-2, 1), (1, 3))
    expected_nodes = [
        [-2.0, 1.0],
        [-2.0, 2.0],
        [-2.0, 3.0],
        [-1.0, 1.0],
        [-1.0, 3.0],
        [0.0, 1.0],
        [0.0, 3.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0]]
    @test nodeset12_3 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset12_3), 0.5)
    @test length(nodeset12_3) == length(expected_nodes)
    display(nodeset12_3)
    for i in eachindex(nodeset12_3)
        @test nodeset12_3[i] == expected_nodes[i]
    end

    nodeset12_4 = @test_nowarn homogeneous_hypercube_boundary((3, 3))
    expected_nodes = [
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 1.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0]]
    @test nodeset12_4 isa NodeSet{2, Float64}
    @test isapprox(separation_distance(nodeset12_4), 0.25)
    @test length(nodeset12_4) == length(expected_nodes)
    for i in eachindex(nodeset12_4)
        @test nodeset12_4[i] == expected_nodes[i]
    end

    r = 2.0
    center = (-1.0, 3.0, 2.0, -pi)
    nodeset13 = @test_nowarn random_hypersphere(50, r, center)
    @test nodeset13 isa NodeSet{4, Float64}
    for node in nodeset13
        @test norm(node .- center) <= r
    end

    nodeset13_1 = @test_nowarn random_hypersphere(50; dim = 4)
    @test nodeset13_1 isa NodeSet{4, Float64}
    for node in nodeset13_1
        @test norm(node) <= 1.0
    end

    @test length(random_hypersphere_boundary(10, r; dim = 1)) == 2

    nodeset14 = @test_nowarn random_hypersphere_boundary(50, r, center)
    @test nodeset14 isa NodeSet{4, Float64}
    for node in nodeset14
        @test isapprox(norm(node .- center), r)
    end

    nodeset14_1 = @test_nowarn random_hypersphere_boundary(50; dim = 4)
    @test nodeset14_1 isa NodeSet{4, Float64}
    for node in nodeset14_1
        @test isapprox(norm(node), 1.0)
    end

    # Meshes.jl extension
    sphere = Sphere(Point(0.0, 0.0, 0.0), 1.0)
    sampler = RegularSampling(5, 6)
    points = Meshes.sample(sphere, sampler)
    nodeset_15 = @test_nowarn NodeSet(collect(points))
    @test nodeset_15 isa NodeSet{3, Float64}
    @test length(nodeset_15) == 32
    ps = PointSet(Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0))
    @test PointSet(NodeSet(ps)) == ps
end

@testitem "Basis" setup=[Setup, AdditionalImports] begin
    nodeset = NodeSet([0.0 0.0
                       1.0 0.0
                       0.0 1.0
                       1.0 1.0])
    kernel = GaussKernel{dim(nodeset)}(shape_parameter = 0.5)
    basis = @test_nowarn StandardBasis(nodeset, kernel)
    @test_throws DimensionMismatch StandardBasis(nodeset,
                                                 GaussKernel{1}(shape_parameter = 0.5))
    @test_nowarn println(basis)
    @test_nowarn display(basis)
    A = kernel_matrix(basis)
    @test isapprox(stack(basis.(nodeset)), A)
    @test isapprox(stack(basis.(nodeset)), kernel.(distance_matrix(nodeset, nodeset)))
    basis_functions = collect(basis)
    for (i, b) in enumerate(basis)
        @test b.(nodeset) == basis_functions[i].(nodeset)
    end

    kernel = ThinPlateSplineKernel{dim(nodeset)}()
    basis = @test_nowarn LagrangeBasis(nodeset, kernel)
    @test_throws DimensionMismatch LagrangeBasis(nodeset,
                                                 GaussKernel{1}(shape_parameter = 0.5))
    @test_nowarn println(basis)
    @test_nowarn display(basis)
    A = kernel_matrix(basis)
    @test isapprox(stack(basis.(nodeset)), A)
    @test isapprox(A, I)
    basis_functions = collect(basis)
    for (i, b) in enumerate(basis)
        @test b.(nodeset) == basis_functions[i].(nodeset)
    end
    # Test for Theorem 11.1 in Wendland's book
    stdbasis = StandardBasis(nodeset, kernel)
    R(x) = stdbasis(x)
    function S(x)
        v = zeros(length(basis.ps))
        for i in eachindex(v)
            v[i] = basis.ps[i](basis.xx => x)
        end
        return v
    end
    b(x) = [R(x); S(x)]
    K = KernelInterpolation.interpolation_matrix(stdbasis, basis.ps)
    x = rand(dim(nodeset))
    uv = K \ b(x)
    u = basis(x)
    # Difficult to test for v
    @test isapprox(u, uv[1:length(u)])
end

@testitem "Interpolation" setup=[Setup, AdditionalImports] begin
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
    @test system_matrix(itp) ==
          KernelInterpolation.interpolation_matrix(nodes, kernel, itp.ps)
    # Saving the interpolation and the function to a VTK file
    @test_nowarn vtk_save("itp", nodes, f, itp, ff; keys = ["f", "itp", "f2"])
    nodes2, point_data = @test_nowarn vtk_read("itp.vtu")
    @test length(nodes) == length(nodes2)
    for i in eachindex(nodes)
        @test [nodes[i]; 0.0] == nodes2[i]
    end
    @test point_data["f"] == ff
    @test point_data["itp"] == itp.(nodes)
    @test point_data["f2"] == ff
    @test_nowarn rm("itp.vtu", force = true)

    expected_coefficients = [
        -2.225451664388596,
        0.31604241814819756,
        0.31604241814819745,
        2.857536500685]
    coeffs = coefficients(itp)
    @test length(coeffs) == length(expected_coefficients)
    for i in eachindex(coeffs)
        @test isapprox(coeffs[i], expected_coefficients[i])
    end
    @test length(kernel_coefficients(itp)) == length(coeffs)
    @test length(polynomial_coefficients(itp)) == 0
    @test order(itp) == 0
    @test length(polynomial_basis(itp)) == 0
    @test length(polyvars(itp)) == dim(itp)
    @test system_matrix(itp) isa Symmetric
    @test isapprox(itp([0.5, 0.5]), 1.115625820404527)
    @test isapprox(kernel_norm(itp), 2.5193566316951626)

    itp = @test_nowarn interpolate(nodes, ff, kernel; factorization_method = cholesky)
    @test system_matrix(itp) isa Cholesky

    # Conditionally positive definite kernel
    # Interpolation
    kernel = ThinPlateSplineKernel{dim(nodes)}()
    itp = @test_nowarn interpolate(nodes, ff, kernel)
    expected_coefficients = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0]
    coeffs = coefficients(itp)
    @test length(coeffs) == length(expected_coefficients)
    for i in eachindex(coeffs)
        @test isapprox(coeffs[i], expected_coefficients[i])
    end
    @test order(itp) == order(kernel)
    @test length(kernel_coefficients(itp)) == length(nodes)
    @test length(polynomial_coefficients(itp)) == order(itp) + 1
    @test length(polynomial_basis(itp)) ==
          binomial(order(itp) - 1 + dim(nodes), dim(nodes))
    @test system_matrix(itp) isa Symmetric
    @test size(system_matrix(itp)) == (7, 7)
    @test isapprox(itp([0.5, 0.5]), 1.0)
    @test isapprox(kernel_norm(itp), 0.0)

    # Regularization
    itp = @test_nowarn interpolate(nodes, ff, kernel,
                                   regularization = L2Regularization(1e-3))
    coeffs = coefficients(itp)
    @test length(coeffs) == length(expected_coefficients)
    for i in eachindex(coeffs)
        @test isapprox(coeffs[i], expected_coefficients[i], atol = 1e-15)
    end
    @test order(itp) == order(kernel)
    @test length(kernel_coefficients(itp)) == length(nodes)
    @test length(polynomial_coefficients(itp)) == order(itp) + 1
    @test length(polynomial_basis(itp)) ==
          binomial(order(itp) - 1 + dim(nodes), dim(nodes))
    @test system_matrix(itp) isa Symmetric
    @test size(system_matrix(itp)) == (7, 7)
    @test isapprox(itp([0.5, 0.5]), 1.0)

    # Least squares approximation
    centers = NodeSet([0.0 0.0
                       1.0 0.0
                       0.0 1.0])
    itp = @test_nowarn interpolate(centers, nodes, ff, kernel)
    expected_coefficients = [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0]
    coeffs = coefficients(itp)
    @test system_matrix(itp) ==
          KernelInterpolation.least_squares_matrix(centers, nodes, kernel, itp.ps)
    @test length(coeffs) == length(expected_coefficients)
    for i in eachindex(coeffs)
        @test isapprox(coeffs[i], expected_coefficients[i], atol = 1e-15)
    end
    @test order(itp) == order(kernel)
    @test length(kernel_coefficients(itp)) == length(KernelInterpolation.centers(itp))
    @test length(polynomial_coefficients(itp)) == order(itp) + 1
    @test length(polynomial_basis(itp)) ==
          binomial(order(itp) - 1 + dim(nodes), dim(nodes))
    @test system_matrix(itp) isa Matrix
    @test size(system_matrix(itp)) == (7, 6)
    @test isapprox(itp([0.5, 0.5]), 1.0)
    @test isapprox(kernel_norm(itp), 0.0, atol = 1e-15)

    itp = @test_nowarn interpolate(centers, nodes, ff,
                                   Matern52Kernel{dim(nodes)}(shape_parameter = 0.5);
                                   factorization_method = qr)
    @test system_matrix(itp) isa LinearAlgebra.QRCompactWY

    # Least squares with LagrangeBasis (not really recommended because you still need to solve a linear system)
    basis = LagrangeBasis(centers, kernel)
    basis_functions = collect(basis)
    # There is no RBF part
    for b in basis_functions
        @test isapprox(kernel_coefficients(b), zeros(length(centers)))
    end
    itp = interpolate(basis, ff, nodes)
    coeffs = coefficients(itp)
    # Polynomial coefficients add up correctly
    expected_coefficients = [
        0.0,
        1.0,
        1.0]
    for i in eachindex(expected_coefficients)
        coeff = 0.0
        for (j, b) in enumerate(basis_functions)
            coeff += coeffs[j] * polynomial_coefficients(b)[i]
        end
        @test isapprox(coeff, expected_coefficients[i], atol = 1e-15)
    end
    @test isapprox(itp([0.5, 0.5]), 1.0)

    # 1D interpolation and evaluation
    nodes = NodeSet(LinRange(0.0, 1.0, 10))
    f(x) = sinpi(x[1])
    ff = f.(nodes)
    kernel = Matern52Kernel{1}()
    itp = @test_nowarn interpolate(nodes, ff, kernel)
    @test isapprox(itp([0.12345]), 0.3783014037753514)
    @test isapprox(itp(0.12345), 0.3783014037753514) # Evaluate at scalar input

    # Applying operators to the interpolation
    f_prime(x) = pi * cospi(x[1])
    many_nodes = NodeSet(LinRange(0.0, 1.0, 50))
    d1 = PartialDerivative(1)
    d1_itp = @test_nowarn d1.(Ref(itp), many_nodes)
    for i in eachindex(many_nodes)
        @test isapprox(d1_itp[i], f_prime(many_nodes[i]), atol = 0.1)
    end
    g = Gradient()
    g_itp = @test_nowarn g.(Ref(itp), many_nodes)
    for i in eachindex(many_nodes)
        @test isapprox(g_itp[i][1], d1_itp[i])
    end

    # TODO: test convergence orders of condition numbers depending on separation distance
end

@testitem "Differential operators" setup=[Setup, AdditionalImports] begin
    l = @test_nowarn Laplacian()
    @test_nowarn println(l)
    @test_nowarn display(l)
    d1 = @test_nowarn PartialDerivative(1)
    @test_nowarn println(d1)
    @test_nowarn display(d1)
    d2 = @test_nowarn PartialDerivative(2)
    @test_nowarn println(d2)
    @test_nowarn display(d2)
    g = @test_nowarn Gradient()
    @test_nowarn println(g)
    @test_nowarn display(g)
    A(x) = [x[1]*x[2] sin(x[1])
            sin(x[2]) x[1]^2]
    b(x) = [x[1]^2 + x[2], x[1] + x[2]^2]
    c(x) = x[1] + x[2]
    el = @test_nowarn EllipticOperator(A, b, c)
    @test_nowarn println(el)
    @test_nowarn display(el)
    # Test if automatic differentiation gives same results as analytical derivatives
    # Define derivatives of Gauss kernel analytically and use them in the solver
    # instead of automatic differentiation
    function phi_deriv(kernel::GaussKernel, r::Real, k = 1)
        eps2 = kernel.shape_parameter^2
        if k == 0
            return phi(kernel, r)
        elseif k == 1
            return -2 * eps2 * r * phi(kernel, r)
        elseif k == 2
            return 2 * eps2 * (2 * eps2 * r^2 - 1) * phi(kernel, r)
        else
            error("Only first and second derivative are implemented for GaussKernel")
        end
    end
    function phi_deriv_over_r(kernel::GaussKernel, r::Real, k = 1)
        if k == 1
            eps2 = kernel.shape_parameter^2
            return -2 * eps2 * phi(kernel, r)
        else
            error("Only first derivative is implemented for GaussKernel")
        end
    end

    struct AnalyticalLaplacian <: KernelInterpolation.AbstractDifferentialOperator
    end

    function (::AnalyticalLaplacian)(kernel::KernelInterpolation.AbstractKernel{Dim},
                                     x) where {Dim}
        r = norm(x)
        return (Dim - 1) * phi_deriv_over_r(kernel, r, 1) + phi_deriv(kernel, r, 2)
    end
    kernel = GaussKernel{2}(shape_parameter = 0.5)
    el_l = EllipticOperator(x -> I, zero, x -> 0) # Laplacian with general elliptic operator

    x1 = [0.4, 0.6]
    @test isapprox(l(kernel, x1), AnalyticalLaplacian()(kernel, x1))
    @test isapprox(g(kernel, x1), [-0.17561908618411226, -0.2634286292761684])
    @test isapprox(d1(kernel, x1), -0.17561908618411226)
    @test isapprox(d2(kernel, x1), -0.2634286292761684)
    @test isapprox(el(kernel, x1), 0.6486985764273818)
    @test isapprox(el_l(kernel, x1), -AnalyticalLaplacian()(kernel, x1))
    kernel = GaussKernel{3}(shape_parameter = 0.5)
    x2 = [0.1, 0.2, 0.3]
    @test isapprox(l(kernel, x2), AnalyticalLaplacian()(kernel, x2))
    @test isapprox(g(kernel, x2),
                   [-0.04828027081287833, -0.09656054162575665, -0.14484081243863497])
    @test isapprox(el_l(kernel, x2), -AnalyticalLaplacian()(kernel, x2))
    kernel = GaussKernel{4}(shape_parameter = 0.5)
    x3 = rand(4)
    @test isapprox(l(kernel, x3, x3), AnalyticalLaplacian()(kernel, x3, x3))
    @test isapprox(el_l(kernel, x3, x3), -AnalyticalLaplacian()(kernel, x3, x3))
end

@testitem "PDEs" setup=[Setup, AdditionalImports] begin
    # stationary PDEs
    # Passing a function
    f1(x, equations) = x[1] + x[2]
    poisson = @test_nowarn PoissonEquation(f1)
    @test_nowarn println(poisson)
    @test_nowarn display(poisson)
    nodeset = NodeSet([0.0 0.0
                       1.0 0.0
                       0.0 1.0
                       1.0 1.0])
    @test KernelInterpolation.rhs(nodeset, poisson) == [0.0, 1.0, 1.0, 2.0]
    # Passing a vector
    poisson = @test_nowarn PoissonEquation([0.0, 1.0, 1.0, 3.0])
    @test KernelInterpolation.rhs(nodeset, poisson) == [0.0, 1.0, 1.0, 3.0]

    A(x) = [x[1]*x[2] sin(x[1])
            sin(x[2]) x[1]^2]
    b(x) = [x[1]^2 + x[2], x[1] + x[2]^2]
    c(x) = x[1] + x[2]
    elliptic = @test_nowarn EllipticEquation(A, b, c, f1)
    @test_nowarn println(elliptic)
    @test_nowarn display(elliptic)
    @test KernelInterpolation.rhs(nodeset, elliptic) == [0.0, 1.0, 1.0, 2.0]
    # Passing a vector
    elliptic = @test_nowarn EllipticEquation(A, b, c, [0.0, 1.0, 1.0, 3.0])
    @test KernelInterpolation.rhs(nodeset, elliptic) == [0.0, 1.0, 1.0, 3.0]

    # time-dependent PDEs
    # Passing a function
    f2(t, x, equations) = x[1] + x[2] + t
    advection = @test_nowarn AdvectionEquation((2.0, 0.5), f2)
    advection = @test_nowarn AdvectionEquation([2.0, 0.5], f2)
    @test_nowarn println(advection)
    @test_nowarn display(advection)
    @test KernelInterpolation.rhs(1.0, nodeset, advection) == [1.0, 2.0, 2.0, 3.0]
    # Passing a vector
    advection = @test_nowarn AdvectionEquation((2.0, 0.5), [1.0, 2.0, 2.0, 4.0])
    @test KernelInterpolation.rhs(1.0, nodeset, advection) == [1.0, 2.0, 2.0, 4.0]

    heat = @test_nowarn HeatEquation(2.0, f2)
    @test_nowarn println(heat)
    @test_nowarn display(heat)
    @test KernelInterpolation.rhs(1.0, nodeset, heat) == [1.0, 2.0, 2.0, 3.0]
    # Passing a vector
    heat = @test_nowarn HeatEquation(2.0, [1.0, 2.0, 2.0, 4.0])
    @test KernelInterpolation.rhs(1.0, nodeset, heat) == [1.0, 2.0, 2.0, 4.0]

    advection_diffusion = @test_nowarn AdvectionDiffusionEquation(2.0, (2.0, 0.5), f2)
    advection_diffusion = @test_nowarn AdvectionDiffusionEquation(2.0, [2.0, 0.5], f2)
    @test_nowarn println(advection_diffusion)
    @test_nowarn display(advection_diffusion)
    @test KernelInterpolation.rhs(1.0, nodeset, advection_diffusion) ==
          [1.0, 2.0, 2.0, 3.0]
    # Passing a vector
    advection_diffusion = @test_nowarn AdvectionDiffusionEquation(2.0, (2.0, 0.5),
                                                                  [1.0, 2.0, 2.0, 4.0])
    @test KernelInterpolation.rhs(1.0, nodeset, advection_diffusion) ==
          [1.0, 2.0, 2.0, 4.0]

    # Test consistency between equations
    kernel = Matern52Kernel{2}(shape_parameter = 0.5)
    x = rand(2)
    y = rand(2)
    @test advection_diffusion(kernel, x, y) ==
          advection(kernel, x, y) + heat(kernel, x, y)
    el_laplace = EllipticEquation(x -> [1 0; 0 1], x -> [0, 0], x -> 0, f1)
    @test el_laplace(kernel, x, y) == poisson(kernel, x, y)
    el_advection = EllipticEquation(x -> [0 0; 0 0], x -> [2, 0.5], x -> 0, f1)
    @test el_advection(kernel, x, y) == advection(kernel, x, y)
    el_advection_diffusion = EllipticEquation(x -> [2 0; 0 2], x -> [2, 0.5], x -> 0.0,
                                              f1)
    @test el_advection_diffusion(kernel, x, y) == advection_diffusion(kernel, x, y)
end

@testitem "Discretization" setup=[Setup, AdditionalImports] begin
    # SpatialDiscretization
    nodeset_inner = NodeSet([0.25 0.25
                             0.75 0.25
                             0.25 0.75
                             0.75 0.75])
    u1(x) = x[1] * (x[1] - 1.0) + (x[2] - 1.0) * x[2]
    f1(x, equations) = -4.0
    nodeset_boundary = NodeSet([0.0 0.0
                                1.0 0.0
                                0.0 1.0
                                1.0 1.0])
    g1(x) = u1(x)
    kernel = Matern52Kernel{2}(shape_parameter = 0.5)
    pde = PoissonEquation(f1)
    sd = @test_nowarn SpatialDiscretization(pde, nodeset_inner, g1, nodeset_boundary,
                                            kernel)
    @test_nowarn println(sd)
    @test_nowarn display(sd)
    @test dim(sd) == 2
    @test eltype(sd) == Float64

    # SemiDiscretization
    u2(t, x) = x[1] * (x[1] - 1.0) + (x[2] - 1.0) * x[2] + t
    f2(t, x, equations) = -4.0 # -Δu
    g2(t, x) = u2(t, x)
    pde = HeatEquation(2.0, f2)
    semi = @test_nowarn Semidiscretization(pde, nodeset_inner, g2, nodeset_boundary,
                                           kernel)
    @test_nowarn println(semi)
    @test_nowarn display(semi)
    @test dim(semi) == 2
    @test eltype(semi) == Float64
end

@testitem "solving PDEs" setup=[Setup, AdditionalImports] begin
    # stationary PDE
    nodeset_inner = NodeSet([0.25 0.25
                             0.75 0.25
                             0.25 0.75
                             0.75 0.75])
    u1(x) = x[1] * (x[1] - 1.0) + (x[2] - 1.0) * x[2]
    f1(x, equations) = -4.0 # -Δu
    nodeset_boundary = NodeSet([0.0 0.0
                                1.0 0.0
                                0.0 1.0
                                1.0 1.0])
    g1(x) = u1(x)
    kernel = Matern52Kernel{2}(shape_parameter = 0.5)
    pde = PoissonEquation(f1)
    sd = SpatialDiscretization(pde, nodeset_inner, g1, nodeset_boundary, kernel)
    itp = @test_nowarn solve_stationary(sd)

    # Test if the solution satisfies the PDE in the inner nodes
    for node in nodeset_inner
        @test isapprox(pde(itp, node), f1(node, pde), atol = 1e-14)
        @test isapprox(Laplacian()(itp, node), -f1(node, pde), atol = 1e-14)
    end
    # Test if the solution satisfies the boundary conditions
    values_boundary = g1.(nodeset_boundary)
    for (node, value) in zip(nodeset_boundary, values_boundary)
        @test isapprox(itp(node), value, atol = 1e-14)
    end
    nodes = nodeset(itp)
    # Test if u = A * c
    A = kernel_matrix(nodes, kernel)
    c = coefficients(itp)
    u1_values = A * c
    u2_values = itp.(nodes)
    for (u1_val, u2_val) in zip(u1_values, u2_values)
        @test isapprox(u1_val, u2_val, atol = 1e-14)
    end
    # Test if L * u = b
    L = operator_matrix(pde, nodeset_inner, nodeset_boundary, kernel)
    b = [f1.(nodeset_inner, Ref(pde)); g1.(nodeset_boundary)]
    b_test = L * u2_values
    for (b_val, b_test_val) in zip(b, b_test)
        @test isapprox(b_val, b_test_val, atol = 1e-12)
    end
    # Test if the solution is close to the analytical solution in other points
    x = [0.1, 0.08]
    @test isapprox(itp(x), u1(x), atol = 0.12)

    # time-dependent PDE
    u2(t, x, equations) = x[1] * (x[1] - 1.0) + (x[2] - 1.0) * x[2] + t
    f2(t, x, equations) = -3.0 # ∂_t u -Δu
    pde = HeatEquation(2.0, f2)
    g2(t, x) = u2(t, x, pde)
    semi = Semidiscretization(pde, nodeset_inner, g2, nodeset_boundary, u2, kernel)
    ode = @test_nowarn semidiscretize(semi, (0.0, 0.1))
    sol = @test_nowarn solve(ode, Rodas5P())
    titp = @test_nowarn TemporalInterpolation(sol)
    @test_nowarn println(titp)
    @test_nowarn display(titp)
    # Test if the solution satisfies the initial condition
    t = sol.t[1]
    for node in merge(nodeset_inner, nodeset_boundary)
        @test isapprox(titp(t, node), u2(t, node, pde), atol = 1e-13)
        @test isapprox(titp(t)(node), titp(t, node))
    end
    # Test if the solution satisfies the boundary conditions
    t = sol.t[2]
    values_boundary = g2.(Ref(t), nodeset_boundary)
    for (node, value) in zip(nodeset_boundary, values_boundary)
        @test isapprox(titp(t, node), value, atol = 1e-13)
        @test isapprox(titp(t)(node), titp(t, node))
    end
    t = sol.t[end]
    values_boundary = g2.(Ref(t), nodeset_boundary)
    for (node, value) in zip(nodeset_boundary, values_boundary)
        @test isapprox(titp(t, node), value, atol = 1e-14)
        @test isapprox(titp(t)(node), titp(t, node))
    end
    # Test if u = A * c
    A = Matrix(semi.cache.kernel_matrix)
    c = sol(t)
    u1_values = A * c
    u2_values = titp.(Ref(t), merge(nodeset_inner, nodeset_boundary))
    for (u1_val, u2_val) in zip(u1_values, u2_values)
        @test isapprox(u1_val, u2_val, atol = 1e-14)
    end
    # Test if the solution is close to the analytical solution in other points
    t = 0.03423
    x = [0.1, 0.08]
    @test isapprox(titp(t, x), u2(t, x, pde), atol = 0.12)
end

@testitem "Different floating point types" setup=[Setup] begin
    # Special nodesets
    @test eltype(@inferred random_hypercube(10, (0.5f0, 0.5f0), (1.0f0, 1.0f0))) == Float32
    @test eltype(@inferred random_hypercube_boundary(10, (0.5f0, 0.5f0), (1.0f0, 1.0f0))) ==
          Float32
    @test eltype(@inferred homogeneous_hypercube(10, (0.5f0, 0.5f0), (1.0f0, 1.0f0))) ==
          Float32
    @test eltype(@inferred homogeneous_hypercube_boundary(10, (0.5f0, 0.5f0),
                                                          (1.0f0, 1.0f0))) == Float32
    @test eltype(@inferred random_hypersphere(10, 1.0f0, (1.0f0, 1.0f0))) == Float32
    @test eltype(@inferred random_hypersphere_boundary(10, 1.0f0, (1.0f0, 1.0f0))) ==
          Float32

    # Interpolation with `StandardBasis`
    centers = NodeSet(Float32[0.0 0.0
                              1.0 0.0
                              0.0 1.0
                              1.0 1.0])
    @test eltype(centers) == Float32
    kernel = MultiquadricKernel{dim(centers)}(; shape_parameter = 0.5f0)
    f(x) = x[1] + x[2]
    ff = f.(centers)
    itp = @test_nowarn interpolate(centers, ff, kernel)
    @test eltype(coefficients(itp)) == Float32
    @test eltype(system_matrix(itp)) == Float32
    @test typeof(@inferred itp([0.5f0, 0.5f0])) == Float32

    # Interpolation with `LagrangeBasis`
    basis = @test_nowarn LagrangeBasis(centers, kernel)
    K = kernel_matrix(basis)
    @test eltype(K) == Float32
    nodes = NodeSet(Float32[0.0 0.0
                            1.0 0.0
                            0.5 0.5
                            0.0 1.0
                            1.0 1.0])
    ff = f.(nodes)
    itp = @test_nowarn interpolate(basis, ff, nodes)
    @test eltype(coefficients(itp)) == Float32
    @test eltype(system_matrix(itp)) == Float32
    @test typeof(@inferred itp([0.5f0, 0.5f0])) == Float32

    # Solving stationary PDE
    nodeset_inner = NodeSet(Float32[0.25 0.25
                                    0.75 0.25
                                    0.25 0.75
                                    0.75 0.75])
    u1(x) = x[1] * (x[1] - 1) + (x[2] - 1) * x[2]
    f1(x, equations) = -4.0f0 # -Δu
    nodeset_boundary = NodeSet(Float32[0.0 0.0
                                       1.0 0.0
                                       0.0 1.0
                                       1.0 1.0])
    g1(x) = u1(x)
    kernel = WendlandKernel{2}(3; shape_parameter = 0.5f0)
    pde = PoissonEquation(f1)
    sd = SpatialDiscretization(pde, nodeset_inner, g1, nodeset_boundary, kernel)
    itp = @test_nowarn solve_stationary(sd)
    @test eltype(coefficients(itp)) == Float32
    @test eltype(system_matrix(itp)) == Float32
    @test typeof(@inferred itp([0.5f0, 0.5f0])) == Float32
end

@testitem "Callbacks" setup=[Setup, AdditionalImports] begin
    # AliveCallback
    alive_callback = AliveCallback(dt = 0.1)
    @test_nowarn println(alive_callback)
    @test_nowarn display(alive_callback)
    alive_callback = AliveCallback(interval = 10)
    @test_nowarn println(alive_callback)
    @test_nowarn display(alive_callback)
    @test_throws ArgumentError AliveCallback(interval = 10, dt = 0.1)
    # SaveSolutionCallback
    save_solution_callback = SaveSolutionCallback(dt = 0.1)
    @test_nowarn println(save_solution_callback)
    @test_nowarn display(save_solution_callback)
    save_solution_callback = SaveSolutionCallback(interval = 10)
    @test_nowarn println(save_solution_callback)
    @test_nowarn display(save_solution_callback)
    @test_throws ArgumentError SaveSolutionCallback(interval = 10, dt = 0.1)
    # SummaryCallback
    summary_callback = SummaryCallback()
    @test_nowarn println(summary_callback)
    @test_nowarn display(summary_callback)
end
