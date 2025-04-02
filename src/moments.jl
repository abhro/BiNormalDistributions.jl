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
    m[begin] = μ
    for k in eachindex(m)[begin+1:end]
        m[k] = mean(xᵢ -> (xᵢ-μ)^k, x)
    end
    return m
end
