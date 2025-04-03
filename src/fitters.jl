using StatsAPI
using KernelDensity: kde
using Peaks: findmaxima, peakproms

"""
    loglikelihood(d::BiNormal, x)

Get the total log-likelihood of a BiNormal distribution `d` producing samples `x`.
"""
function StatsAPI.loglikelihood(d::BiNormal{T}, x::AbstractArray{T}) where {T<:Real} # optimizing for params
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)
    logL = 0.0
    for xᵢ in x
        logL += logpdf(d, xᵢ)
    end
    return logL
end

"""
    ∇loglikelihood(d::BiNormal, x)

Gradient of the [`loglikelihood`](@ref) function with respect to
the parameters of `d`.
"""
function ∇loglikelihood(d::BiNormal, x)
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)
    ∂λ = 0.0
    ∂μ₁ = 0.0
    ∂σ₁ = 0.0
    ∂μ₂ = 0.0
    ∂σ₂ = 0.0
    for xᵢ in x
        pdf₁, pdf₂ = componentpdfs(d, xᵢ)
        pdf₁ /= λ
        pdf₂ /= 1-λ
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

"""
    kdemaxes(x, n = nothing)

Find the most prominent occurences in a data series.
Calculated by making a kernel density estimate and then finding the peaks of the
KDE when treated as a signal.

Returns the interpolated KDE and the maxima of the KDE.

See also: [`maxes`](@ref).
"""
function kdemaxes(x, n = nothing)
    interped = kde(x, npoints = length(x))
    return (interped, maxes(interped.density, n))
end

"""
    maxes(x, n = nothing)

Find the `n` most prominent maxima of a signal/curve `x`.
"""
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
