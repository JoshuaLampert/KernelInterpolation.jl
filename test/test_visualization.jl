@testitem "Visualization" begin
    using Plots
    f = sum
    kernel = GaussKernel{3}(shape_parameter = 0.5)
    kernel_1d = Matern12Kernel{1}()
    trafo_kernel = TransformationKernel{2}(kernel, x -> [x[1] + x[2]^2, x[1]])
    @test_nowarn plot(-1.0:0.1:1.0, kernel)
    for dim in 1:3
        nodes = homogeneous_hypercube(5; dim = dim)
        @test_nowarn plot(nodes)
        if dim < 3
            @test_nowarn plot(nodes, kernel)
            @test_nowarn plot(kernel_1d)
            @test_nowarn plot(trafo_kernel, x_min = -2, x_max = 2, N = 100)
            # Transformation kernel can only be plotted in the dimension of the input of the trafo
            if dim == 2
                @test_nowarn plot(nodes, trafo_kernel)
            end
            ff = f.(nodes)
            itp = interpolate(nodes, ff)
            nodes_fine = homogeneous_hypercube(10; dim = dim)
            @test_nowarn plot(nodes_fine, itp)
            @test_nowarn plot(itp)
            if dim == 2
                # Test if 2D nodes can be plotted into 3D plot
                nodes2d = homogeneous_hypercube(5; dim = 2)
                @test_nowarn plot!(nodes2d)
            end
        end
    end
end
