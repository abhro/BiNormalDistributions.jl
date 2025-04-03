"""
    moment(d::BiNormal, k)

Compute the k-th moment of the BiNormal distribution around 0.
"""
StatsBase.moment(d::BiNormal, k) = d.λ*moment(d.N₁, k) + (1-d.λ)*moment(d.N₂, k)

@doc """
    moment(d::Normal, Val(k))

Compute the k-th moment of the normal distribution around 0.
""" StatsBase.moment(::Normal, ::Any)

StatsBase.moment(d::Normal, ::Val{1}) = d.μ
StatsBase.moment(d::Normal, ::Val{2}) = d.μ^2 + d.σ^2
StatsBase.moment(d::Normal, ::Val{3}) = d.μ * (d.μ^2 + 3*d.σ^2)
StatsBase.moment(d::Normal, ::Val{4}) = d.μ^4 + 6*d.μ^2*d.σ^2 + 3*d.σ^4
StatsBase.moment(d::Normal, ::Val{5}) = d.μ^5 + 10*d.μ^3*d.σ^2 + 15*d.μ*d.σ^4
function StatsBase.moment(d::Normal, ::Val{6})
    μ, σ = params(d)
    return μ^6 + 15*μ^5*σ^2 + 45*μ^2*σ^4 + 15*σ^6
end
function StatsBase.moment(d::Normal, ::Val{7})
    μ, σ = params(d)
    return μ^7 + 21*μ^5*σ^2 + 105*μ^3*σ^4 + 105*σ^7
end
function StatsBase.moment(d::Normal, ::Val{8})
    μ, σ = params(d)
    return μ^8 + 28*μ^6*σ^2 + 210*μ^4*σ^4 + 420*μ^2*σ^6 + 105*σ^8
end
