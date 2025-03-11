using BiNormalDistribution
using Test
using Aqua

@testset "BiNormalDistribution.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(BiNormalDistribution)
    end
    # Write your tests here.
end
