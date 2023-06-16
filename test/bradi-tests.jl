# bradi-tests.jl -
#
# Tests for BRADI algorithm.
#

module BraDiTests

using Test
import OptimPackNextGen: BraDi

struct Problem{F<:Function,R<:AbstractVector{Float64}}
    name::String
    f::F
    range::R
    period::Float64
    xmin::Float64
    xmax::Float64
end
function Problem(name::AbstractString,
                 f::Function,
                 range::AbstractVector{<:Real};
                 period::Real = NaN,
                 xmin::Real = NaN,
                 xmax::Real = NaN)
    return Problem(name, f, range, period, xmin, xmax)
end

# Simple parabola. To be minimized over [-1,2].
const parabola = Problem(
    "x^2",
    (x) -> x*x,
    range(start=-1, stop=2, length=2),
    xmin = 0.0)

# Brent's 5th function.  To be minimized over [-10,10].
const brent5 = Problem(
    "Brent's 5th function",
    (x) -> (x - sin(x))*exp(-x*x),
    range(start=-10, stop=10, length=5),
    xmin = -1.1951366418407416)

# Michalewicz's 1st function.  To be minimized over [-1,2].
const michalewicz1 = Problem(
    "Michalewicz's 1st function",
    (x) -> x*sin(10.0*x),
    range(start=1, stop=2, length=21),
    xmin = 1.7336377815999973)

# Michalewicz's 2nd function.  To be maximized over [0,pi].
const michalewicz2 = Problem(
    "Michalewicz's 2nd function",
    (x) -> begin
        s = 0.0
        a = sin(x)
        b = x*x/pi
        for i in 1:10
            s += a*(sin(b*i)^20)
        end
        return s
    end,
    range(start=0, stop=π, length=60),
    xmax = 2.2208651539586493)

struct MultiFringe{T<:AbstractFloat} <: Function
    ω₁::T
    ϕ₁::T
    ω₂::T
    ϕ₂::T
end
MultiFringe(args::Vararg{Real,4}) =
    MultiFringe{float(promote_type(map(typeof, args)...))}(args...)
(f::MultiFringe{T})(x) where {T} = (one(T) + cos(f.ω₁*x - f.ϕ₁))*cos(f.ω₂*x - f.ϕ₂)
# period is 2π if all ωᵢ integer
# maximized if x ≈ (ϕᵢ + 2kπ)/ωᵢ
# number of local min. (resp. max.) over a period = maxᵢ|ωᵢ|
# hence n ≥ 2maxᵢ|ωᵢ| + 1
const fringes1 = Problem(
     "Multi-fringe #1",
    MultiFringe(1, 2, 50, 3),
    range(start=-π, stop=π, length=101),
    period = 2π,
    xmax = 1.944966601611255,
    xmin = 2.007785888040201)

# This one has a global maximum a bit larger than -π and a global minimum a bit
# smaller than π, so it is perfect to test behavior at the edges of the search
# interval [-π,π]:
const fringes2 = Problem(
    "Multi-fringe #2",
    MultiFringe(1, π, 50, 1//5),
    range(start=-π, stop=π, length=101),
    period = 2π,
    xmax = -3.137593453430891,
    xmin =  3.082772567926711)

# This one has a global minimum a bit larger than -π and a global maximum a bit
# smaller than π so perfect to test behavior at the edges of the search
# interval [-π,π]:
const fringes3 = Problem(
    "Multi-fringe #3",
    MultiFringe(1, π, 50, -1//5),
    range(start=-π, stop=π, length=101),
    period = 2π,
    xmax =  3.137593453430891,
    xmin = -3.082772567926711)

const problems = (parabola, brent5, michalewicz1, michalewicz2,
                  fringes1, fringes2, fringes3)

function runtests(; quiet::Bool=false, tol=sqrt(eps(Float64)))
    @testset "$(rpad(p.name, 30))" for p in problems
        period = isnan(p.period) ? nothing : p.period
        quiet || println("\n# $(p.name):")
        if !isnan(p.xmin)
            (xbest, fbest) = BraDi.minimize(p.f, p.range; period)
            quiet || println("#   min: x = $xbest ($(p.xmin)), f(x) = $fbest ($(p.f(p.xmin)))")
            atol = iszero(p.xmin) ? tol : abs(p.xmin)*tol
            @test xbest ≈ p.xmin atol=atol rtol=0
        end
        if !isnan(p.xmax)
            (xbest, fbest) = BraDi.maximize(p.f, p.range; period)
            quiet || println("#   max: x = $xbest ($(p.xmax)), f(x) = $fbest ($(p.f(p.xmax)))")
            atol = iszero(p.xmax) ? tol : abs(p.xmax)*tol
            @test xbest ≈ p.xmax atol=atol rtol=0
        end
    end
end
@testset "BraDi algorithm" begin
    runtests(quiet=true)
end

end # module
