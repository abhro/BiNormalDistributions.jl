using KernelDensity: kde
using Peaks: findmaxima, peakproms

function loglikelihood(d::BiNormal, x) # optimizing for params
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)
    N₁ = Normal(μ₁, σ₁)
    N₂ = Normal(μ₂, σ₂)
    logL = 0.0
    for xᵢ in x
        logL += log(λ * pdf(N₁, xᵢ) + (1 - λ) * pdf(N₂, xᵢ))
    end
    return logL
end

function ∇loglikelihood(d::BiNormal, x)
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)
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

function histmaxes(x, n = nothing)
    interped = kde(x, npoints = length(x))
    return (interped, maxes(interped.density, n))
end

function maxes(x, n = nothing)
    peaks = findmaxima(x) |> peakproms
    fields = [:indices, :heights, :proms]
    peaks = zip(peaks[fields]...) |> collect # effectively 'transpose' fields

    sort!(peaks, by = x -> x[end], rev = true) # sort based on proms

    # keep the first n values
    if !isnothing(n)
        peaks = first(peaks, n)
    end

    indices = broadcast(x->x[1], peaks)
    heights = broadcast(x->x[2], peaks)
    proms = broadcast(x->x[3], peaks)
    return (; indices, heights, proms)
end
