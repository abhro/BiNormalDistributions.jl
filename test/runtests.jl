using BiNormalDistribution
using Test
using Aqua
using StableRNGs
using Distributions: Uniform

@testset "BiNormalDistribution" begin
    @testset "Code quality (Aqua)" begin
        Aqua.test_all(BiNormalDistribution, piracies = false)
    end

    # test distribution with same underlying means
    @testset "μ₁ = μ₂ = μ" begin
        rng = StableRNG(123)
        μ = rand(rng)
        λ = rand(rng, Uniform(1//2, 1))
        @info "Testing distributions with same mean $μ (λ = $λ)"
        dist = BiNormal(λ, μ, 1, μ, 1)

        # Create a dataset with "enough" samples
        x = rand(rng, dist, 1000)

        @test mean(x) ≈ μ
    end

    # ditto with same variances
end
