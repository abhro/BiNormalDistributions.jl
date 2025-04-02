@doc raw"""
    moments(x::AbstractVector, n::Integer)

Get the first `n`-moments of a given dataset. (``⟨ x^k ⟩``, ``k = 1, \dots, n``)
"""
function moments(x::AbstractVector, n::Integer)
    m = zeros(eltype(x), n)
    for k in eachindex(m)
        m[k] = mean(xᵢ -> xᵢ^k, x)
    end
    return m
end
@doc raw"""
    centralmoments(x::AbstractVector, n::Integer)

Get the first `n` central moments of a given dataset. (``⟨ (x-μ)^k ⟩``, ``k = 1, \dots, n``)
"""
function centralmoments(x::AbstractVector, n::Integer)
    m = zeros(eltype(x), n)
    μ = mean(x)
    # the first central moment is 0, don't bother computing
    for k in eachindex(m)[begin+1:end]
        m[k] = mean(xᵢ -> (xᵢ-μ)^k, x)
    end
    return m
end

moment(d::BiNormal, k) = d.λ*moment(d.N₁, k) + (1-d.λ)*moment(d.N₂, k)

moment(d::Normal, ::Val{1}) = d.μ
moment(d::Normal, ::Val{2}) = d.μ^2 + d.σ^2
moment(d::Normal, ::Val{3}) = d.μ * (d.μ^2 + 3*d.σ^2)
function moment(d::Normal, ::Val{4})
    μ, σ = params(d)
    return μ^4 + 6*μ^2*σ^2 + 3*σ^4
end
function moment(d::Normal, ::Val{5})
    μ, σ = params(d)
    return μ^5 + 10*μ^3*σ^2 + 15*μ*σ^4
end
function moment(d::Normal, ::Val{6})
    μ, σ = params(d)
    return μ^6 + 15*μ^5*σ^2 + 45*μ^2*σ^4 + 15*σ^6
end
function moment(d::Normal, ::Val{7})
    μ, σ = params(d)
    return μ^7 + 21*μ^5*σ^2 + 105*μ^3*σ^4 + 105*σ^7
end
function moment(d::Normal, ::Val{8})
    μ, σ = params(d)
    return μ^8 + 28*μ^6*σ^2 + 210*μ^4*σ^4 + 420*μ^2*σ^6 + 105*σ^8
end
