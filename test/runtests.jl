using BiNormalDistribution
using Test
using Aqua

@testset "BiNormalDistribution" begin
    @testset "Code quality (Aqua)" begin
        Aqua.test_all(BiNormalDistribution, piracies = false)
    end
    # Write your tests here.

    # test distribution with same underlying means

    # ditto with same variances
end
