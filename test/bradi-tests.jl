# bradi-tests.jl -
#
# Tests for BRADI algorithm.
#

if ! isdefined(:OptimPackNextGen)
    include("../src/OptimPackNextGen.jl")
end

module BradiTests

using Base.Test
import OptimPackNextGen: Bradi

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

function runtests()
    println("\n# Simple parabola:")
    (xbest, fbest) = Bradi.minimize(parabola, linspace(-1, 2, 2))
    println("x = $xbest, f(x) = $fbest")

    println("\n# Brent's 5th function:")
    (xbest, fbest) = Bradi.minimize(brent5, linspace(-10, 10, 5))
    println("x = $xbest, f(x) = $fbest")

    println("\n# Michalewicz's 1st function:")
    (xbest, fbest) = Bradi.minimize(michalewicz1, linspace(-1, 2, 21))
    println("x = $xbest, f(x) = $fbest")

    println("\n# Michalewicz's 2nd function:")
    (xbest, fbest) = Bradi.maximize(michalewicz2, linspace(0, pi, 60))
    println("x = $xbest, f(x) = $fbest")
end

#verb = true
#@testset "Brent fzero" begin
#    for (T, prec) in ((Float16, 3e-2),
#                      (Float32, 3e-6),
#                      (Float64, 6e-15))
#        n = 0
#        f(x) = (n += 1; T(1)/(x - T(3)) - T(6))
#        (x, fx) = Brent.fzero(T, f, 3, 4)
#        if verb
#            @printf("n = %d, x = %.15f, f(x) = %.15f\n", n, x, fx)
#        end
#        @test abs(fx) ≤ prec
#    end
#end

end # module
