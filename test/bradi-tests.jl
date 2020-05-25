# bradi-tests.jl -
#
# Tests for BRADI algorithm.
#

module BraDiTests

using Test
import OptimPackNextGen: BraDi

# Simple parabola.  To be minimized over [-1,2].
parabola(x) = x*x

# Brent's 5th function.  To be minimized over [-10,10].
brent5(x) = (x - sin(x))*exp(-x*x)

# Michalewicz's 1st function.  To be minimized over [-1,2].
michalewicz1(x) = x*sin(10.0*x)

# Michalewicz's 2nd function.  To be maximized over [0,pi].
function michalewicz2(x)
    s = 0.0
    a = sin(x)
    b = x*x/pi
    for i in 1:10
        s += a*(sin(b*i)^20)
    end
    return s
end

function runtests(; quiet::Bool=false)
    tol = sqrt(eps(Float64))
    @testset "BraDi algorithm" begin
        quiet || println("\n# Simple parabola:")
        (xbest, fbest) = BraDi.minimize(parabola, range(-1, stop=2, length=2))
        println("x = $xbest, f(x) = $fbest")
        @test xbest ≈ 0.0 atol=tol

        quiet || println("\n# Brent's 5th function:")
        (xbest, fbest) = BraDi.minimize(brent5, range(-10, stop=10, length=5))
        quiet || println("x = $xbest, f(x) = $fbest")
        @test xbest ≈ -1.1951366418407416 rtol=tol

        quiet || println("\n# Michalewicz's 1st function:")
        (xbest, fbest) = BraDi.minimize(michalewicz1, range(-1, stop=2, length=21))
        quiet || println("x = $xbest, f(x) = $fbest")
        @test xbest ≈ 1.7336377815999973 rtol=tol

        quiet || println("\n# Michalewicz's 2nd function:")
        (xbest, fbest) = BraDi.maximize(michalewicz2, range(0, stop=pi, length=60))
        quiet || println("x = $xbest, f(x) = $fbest")
        @test xbest ≈ 2.2208651539586493 rtol=tol
    end
end
runtests()

end # module
