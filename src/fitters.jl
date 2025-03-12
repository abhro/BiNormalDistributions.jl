function loglikelihood(d::BiNormalDistribution, x) # optimizing for params
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)
    N₁ = Normal(μ₁, σ₁)
    N₂ = Normal(μ₂, σ₂)
    logL = 0.0
    for xᵢ in x
        logL += log(λ * pdf(N₁, xᵢ) + (1 - λ) * pdf(N₂, xᵢ))
    end
    return logL
end

function ∇loglikelihood(p, x)
    λ, μ₁, σ₁, μ₂, σ₂ = p
    N₁ = Normal(μ₁, σ₁)
    N₂ = Normal(μ₂, σ₂)
    ∂λ = 0.0
    ∂μ₁ = 0.0
    ∂σ₁ = 0.0
    ∂μ₂ = 0.0
    ∂σ₂ = 0.0
    for xᵢ in x
        pdf₁ = pdf(N₁, xᵢ)
        pdf₂ = pdf(N₂, xᵢ)
        denom = λ * pdf₁ + (1 - λ) * pdf₂

        ∂λ += (pdf₁ - pdf₂) / denom
        ∂μ₁ += (xᵢ - μ₁) * pdf₁ / denom
        ∂σ₁ += ((xᵢ - μ₁)^2/(4σ₁^2) - 1) * pdf₁ / denom
        ∂μ₂ += (xᵢ - μ₂) * pdf₂ / denom
        ∂σ₂ += ((xᵢ - μ₂)^2/(4σ₂^2) - 1) * pdf₂ / denom
    end
    ∂λ *= λ
    ∂μ₁ *= λ/σ₁^2
    ∂σ₁ *= λ/σ₁
    ∂μ₂ *= 1-λ/σ₂^2
    ∂σ₂ *= (1-λ)/σ₂

    return [∂λ, ∂μ₁, ∂σ₁, ∂μ₂, ∂σ₂]
end
