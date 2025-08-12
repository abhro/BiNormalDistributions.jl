using BiNormalDistribution
using Test
using Aqua
using StableRNGs
using Distributions: Uniform
using Statistics

@testset "BiNormalDistribution" begin
    @testset "Code quality (Aqua)" begin
        Aqua.test_all(BiNormalDistribution, piracies = false)
    end

    # test distribution with unit mean, unit variance
    @testset "μ₁ = μ₂ = σ₁ = σ₂ = 1" begin
        rng = StableRNG(123)
        λ = rand(rng, Uniform(1//2, 1))
        dist = BiNormal(λ, 1, 1, 1, 1)

        # Create a dataset with "enough" samples
        x = rand(rng, dist, 100_000_000)

        @test mean(x) ≈ 1                 atol=5e-4
        @test std(x) ≈ std(dist)
    end

    # test distribution with same underlying means
    @testset "μ₁ = μ₂ = μ" begin
        rng = StableRNG(123)
        μ = rand(rng)
        λ = rand(rng, Uniform(1//2, 1))
        σ₁ = randn(rng) |> abs
        σ₂ = randn(rng) |> abs
        dist = BiNormal(λ, μ, σ₁, μ, σ₂)
        @info "Testing distributions with same mean $μ" dist

        # Create a dataset with "enough" samples
        x = rand(rng, dist, 100_000_000)

        @test mean(x) ≈ μ                 atol=5e-4
        @test std(x) ≈ std(dist)
    end

    # test distribution with same underlying variances
    @testset "σ₁ = σ₂ = σ" begin
        rng = StableRNG(123)
        μ₁ = rand(rng)
        μ₂ = rand(rng)
        λ = rand(rng, Uniform(1//2, 1))
        σ = randn(rng) |> abs
        dist = BiNormal(λ, μ, σ₁, μ, σ₂)
        @info "Testing distributions with same variance $(σ^2)" dist

        # Create a dataset with "enough" samples
        x = rand(rng, dist, 100_000_000)

        @test mean(x) ≈ μ                 atol=5e-4
        @test std(x) ≈ std(dist)
    end
end
