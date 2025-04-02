module BiNormalDistribution

using Distributions
import Random
import Random: AbstractRNG
import ForwardDiff: derivative
using Roots: find_zero, Newton
using Statistics: mean
using QuadGK: quadgk

export BiNormal, moments, centralmoments

@doc raw"""
    BiNormal{T<:Real,W<:Real} <: ContinuousUnivariateDistribution
    BiNormal(λ::W, μ₁::T, σ₁::T, μ₂::T, σ₂::T)

Linear combination of two Gaussians.

The distribution is defined by five parameters:
- `λ`: weight of the first normal distribution N₁. Since the N₁ is presumed to have more influence, `λ` must be within 1/2 and 1.
- `μ₁`: mean of the first normal distribution N₁.
- `σ₁`: variance of the first normal distribution N₁.
- `μ₂`: mean of the second normal distribution N₂.
- `σ₂`: variance of the second normal distribution N₂.

Domain transformation:

If the weight (`λ`) is available as a real number instead, the following domain transformation can be used:
```math
β = \ln \frac{2λ-1}{2-2λ},
\quad
λ = \frac{2+e^{-β}}{2+2e^{-β}}
```
Here, ``λ ∈ [1/2, 1]`` and ``β ∈ ℝ``.
"""
struct BiNormal{T<:Real,W<:Real} <: ContinuousUnivariateDistribution
    λ::W # Should this be constrained to be in [1/2, 1]?
    N₁::Normal{T}
    N₂::Normal{T}
end
BiNormal(λ, μ₁, σ₁, μ₂, σ₂) = BiNormal(λ, Normal(μ₁, σ₁), Normal(μ₂, σ₂))

"""
    params(d::BiNormal)

Return the parameters of a `BiNormal` distribution: ``(λ, μ_1, σ_1, μ_2, σ_2)``.
"""
Distributions.params(d::BiNormal) = (d.λ, params(d.N₁)..., params(d.N₂)...)
Base.eltype(::Type{BiNormal{T}}) where {T} = T

function Random.rand(rng::AbstractRNG, d::BiNormal)
    x₁ = Random.rand(rng, d.N₁)
    x₂ = Random.rand(rng, d.N₂)
    return d.λ * x₁ + (1 - d.λ) * x₂
end

"""
    componentpdfs(d::BiNormal, x::Real)

Return the (weighted) component pdfs of the BiNormal distribution, that is,
``λ N(x; μ_1, σ_1)`` and ``(1-λ) N(x; μ_2, σ_2)``,
where ``N`` is the pdf of the normal distribution.
"""
componentpdfs(d::BiNormal, x::Real) = (d.λ * pdf(d.N₁, x), (1 - d.λ) * pdf(d.N₂, x))

"""
    componentcdfs(d::BiNormal, x::Real)

Return the (weighted) component pdfs of the BiNormal distribution, that is,
``λ F_N(x; μ_1, σ_1)`` and ``(1-λ) F_N(x; μ_2, σ_2)``,
where ``F_N`` is the cdf of the normal distribution.
"""
componentcdfs(d::BiNormal, x::Real) = (d.λ * cdf(d.N₁, x), (1 - d.λ) * cdf(d.N₂, x))

@doc raw"""
    pdf(d::BiNormal, x::Real)

The probability density function (pdf) is
```math
f(x; λ, μ_1, σ_1, μ_2, σ_2) = λ N(x; μ_1, σ_1) + (1-λ) N(x; μ_2, σ_2)
```
where ``N`` is the pdf of the normal distribution.
""" Distributions.pdf(d::BiNormal, x::Real)
Distributions.logpdf(d::BiNormal, x::Real) = log(sum(componentpdfs(d, x)))

@doc raw"""
    cdf(d::BiNormal, x::Real)

The cumulative density function (cdf) is
```math
F(x; λ, μ_1, σ_1, μ_2, σ_2) = λ F_N(x; μ_1, σ_1) + (1-λ) F_N(x; μ_2, σ_2)
```
where ``F_N`` is the cdf of the normal distribution.
"""
Distributions.cdf(d::BiNormal, x::Real) = sum(componentcdfs(d, x))

"""
    quantile(d::BiNormal, q::Real)

Use Newton's method to find the quantile for `BiNormal` distribution `d`.
"""
function Distributions.quantile(d::BiNormal, q::Real)
    # function to find roots of
    cdf_minus_q = x -> cdf(d, x) -q
    cdf_minus_q_prime = x -> derivative(cdf_minus_q, x)
    find_zero((cdf_minus_q, cdf_minus_q_prime), mean(d), Newton())
end

# support is over all the reals
Base.minimum(::Union{Type{BiNormal{T}}, BiNormal{T}}) where {T<:Real} = typemin(T)
Base.maximum(::Union{Type{BiNormal{T}}, BiNormal{T}}) where {T<:Real} = typemax(T)
Distributions.insupport(d::BiNormal, x::Real) = true

"""
    mean(d::BiNormal)

Mean of the bi-normal distribution is ``μ = λ μ_1 + (1 - λ) μ_2``
"""
Distributions.mean(d::BiNormal) = d.λ * mean(d.N₁) + (1 - d.λ) * mean(d.N₂)

@doc raw"""
    var(d::BiNormal)

Variance of the bi-normal distribution is
```math
\begin{align*}
σ^2 &= λσ_1^2 + (1-λ)σ_2^2 + λ(1-λ) (μ_1 - μ_2)^2 \\
&= λ (σ_1^2 + μ_1^2) + (1 - λ) (σ_2^2 + μ_2^2) - μ^2
\end{align*}
```
"""
function Distributions.var(d::BiNormal)
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)
    return λ * σ₁^2 + (1 - λ) * σ₂^2 + λ * (1 - λ) * (μ₁ - μ₂)^2
end

Distributions.mode(d::BiNormal) = d.N₁.μ

Distributions.modes(d::BiNormal) = [d.N₁.μ, d.N₂.μ]

@doc raw"""
    skewness(d::BiNormal)

Mathematical definition:
```math
γ = \frac{λ μ_1 (μ_1^2 + σ_1^2) + (1 - λ) μ_2 (μ_2^2 + σ_2^2) - μ (3 σ^2 + μ^2)}{σ^3}
```
where ``μ`` is the [mean of `d`](@ref Distributions.mean(::BiNormal)) and
``σ`` is the [standard deviation of `d`](@ref Distributions.var(d::BiNormal))
"""
function Distributions.skewness(d::BiNormal)
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)
    μ = mean(d)
    σ² = var(d)

    γ = (λ*μ₁*(μ₁^2+σ₁^2) + (1-λ)*μ₂*(μ₂^2+σ₂^2) - μ*(3σ²+μ^2)) / (σ²^(2//3))

    return γ
end

@doc raw"""
    kurtosis(d::BiNormal)

Mathematical definition:
```math
\frac{
         λ  (μ_1^4 + 3 σ_1^4 + 6 μ_1^2 σ_1^2)
    + (1-λ) (μ_2^4 + 3 σ_2^4 + 6 μ_2^2 σ_2^2)
    + 3 μ^2 (μ^2 + 2 σ^2)
    - 4 μ [λ μ_1 (μ_1^2 + 3 σ_1^2) + (1-λ) μ_2 (μ_2^2 + 3 σ_2^2)]
}{
    σ^4
}
```
where ``μ`` is the [mean of `d`](@ref Distributions.mean(::BiNormal)) and
``σ`` is the [standard deviation of `d`](@ref Distributions.var(d::BiNormal))
"""
function Distributions.kurtosis(d::BiNormal)
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)
    μ = mean(d)
    σ² = var(d)

    numerator = (
             λ *(μ₁^4 + 3*σ₁^4 + 6*μ₁^2*σ₁^2)
        + (1-λ)*(μ₂^4 + 3*σ₂^4 + 6*μ₂^2*σ₂^2)
        + 3*μ^2*(μ^2 + 2σ²)
        - 4*μ*(λ*μ₁*(μ₁^2 + 3σ₁^2) + (1-λ)*μ₂*(μ₂^2 + 3σ₂^2))
   )

    return numerator / σ²^2
end

"""
    entropy(d::BiNormal)

Calculate the entropy of a BiNormal distribution `d`, evaluated numerically.
"""
function Distributions.entropy(d::BiNormal)
    integrand = x -> pdf(d, x) * log(pdf(d, x))
    integral, residual = quadgk(integrand, -Inf, Inf)
    @debug("Found entropy with residual", d, residual)
    return -integral
end

@doc raw"""
    mgf(d::BiNormal, t)

Moment generating function (``M_d``) of a bi-normal distribution `d`.
The mathematical definition is:
```math
M_d(t) = λ \exp\left(t μ_1 + \tfrac{1}{2} t^2 σ_1^2\right)
       + (1-λ) \exp\left(t μ_2 + \tfrac{1}{2} t^2 σ_2^2\right)
```
"""
Distributions.mgf(d::BiNormal, t) = λ*mgf(d.N₁, t) + (1-λ)*mgf(d.N₂, t)

@doc raw"""
    cf(d::BiNormal, t)

Characteristic function (``φ_d``) of a bi-normal distribution `d`.
The mathematical definition is:
```math
φ_d(t) = λ \exp\left(itμ_1 - \tfrac{1}{2} t^2 σ_1^2\right)
       + (1-λ) \exp\left(itμ_2 - \tfrac{1}{2} t^2 σ_2^2\right)
```
"""
Distributions.cf(d::BiNormal, ::Any) = λ*cf(d.N₁, t) + (1-λ)*cf(d.N₂, t)

function Base.show(io::IO, d::BiNormal)
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)
    print(io, typeof(d)) # handle type parameters
    print(io, "(λ=", λ, ", μ₁=", μ₁, ", σ₁=", σ₁, ", μ₂=", μ₂, ", σ₂=", σ₂, ")")
end

"""
    median(d::BiNormal)

Find the median of a BiNormal distribution `d`. Note that this doesn't have an
analytical solution, so a root-finding algorithm from Roots.jl is employed.
"""
function Distributions.median(d::BiNormal)
    λ, μ₁, σ₁, μ₂, σ₂ = params(d)

    # function which is zero for x = median
    f = x -> λ * erf((x - μ₁)/σ₁) + (1 - λ) * erf((x - μ₂)/σ₂)

    # XXX bracket the search between
    # `extrema([μ₁ + 2σ₁, μ₁ - 2σ₁, μ₂ + 2σ₂, μ₂ - 2σ₂])`?

    return find_zero(f, mean(d)) # initial guess at mean
end

include("fitters.jl")

end
