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

function runtests()
    println("\n# Simple parabola:")
    (xbest, fbest) = BraDi.minimize(parabola, range(-1, stop=2, length=2))
    println("x = $xbest, f(x) = $fbest")

    println("\n# Brent's 5th function:")
    (xbest, fbest) = BraDi.minimize(brent5, range(-10, stop=10, length=5))
    println("x = $xbest, f(x) = $fbest")

    println("\n# Michalewicz's 1st function:")
    (xbest, fbest) = BraDi.minimize(michalewicz1, range(-1, stop=2, length=21))
    println("x = $xbest, f(x) = $fbest")

    println("\n# Michalewicz's 2nd function:")
    (xbest, fbest) = BraDi.maximize(michalewicz2, range(0, stop=pi, length=60))
    println("x = $xbest, f(x) = $fbest")
end
runtests()
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
