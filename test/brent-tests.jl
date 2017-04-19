# brent-tests.jl -
#
# Tests for Brent's algorithms.
#

if ! isdefined(:OptimPackNextGen)
    include("../src/OptimPackNextGen.jl")
end

module BrentTests

using Base.Test
import OptimPackNextGen: Brent

n = 0

# Problem 4 from AMPGO, xm = 2.868034, fm = -3.85045:
problem04(x) = (global n; n += 1; -(16x^2 - 24x + 5)*exp(-x))
bounds04 = [1.9, 3.9]
solution04 = [2.868034, -3.85045]

# Problem 13 from AMPGO, xm = 1/sqrt(2), fm = -1.5874:
problem13(x) = (global n; n += 1; -x^(2/3) - (1 - x^2)^(1/3))
bounds13 = [0.001, 0.99]
solution13 = [1/sqrt(2), -1.5874]

# Problem 18 from AMPGO, xm = 2, fm = 0:
problem18(x) = (global n; n += 1; x ≤ 3 ? (x - 2)^2 : 2log(x - 2) + 1)
bounds18 = [0, 6]
solution18 = [2, 0]

verb = true
@testset "Brent fzero" begin
    for (T, prec) in ((Float16, 3e-2),
                      (Float32, 3e-6),
                      (Float64, 6e-15))
        n = 0
        f(x) = (global n; n += 1; T(1)/(x - T(3)) - T(6))
        (x, fx) = Brent.fzero(T, f, 3, 4)
        if verb
            @printf("n = %d, x = %.15f, f(x) = %.15f\n", n, x, fx)
        end
        @test abs(fx) ≤ prec
    end
end

@testset "Brent fmin" begin
    for (T, prec) in ((Float16, 3e-2),
                      (Float32, 3e-6),
                      (Float64, 6e-15))
        atol = Brent.fmin_atol(T)
        rtol = Brent.fmin_rtol(T)

        n = 0
        (xm, fm, lo, hi) = Brent.fmin(T, problem04, bounds04...)
        if verb
            @printf("Problem 4: n = %d, x = %.15f ± %.3g, f(x) = %.15f\n",
                    n, xm, (hi - lo)/2, fm)
        end
        @test abs(xm - solution04[1]) ≤ atol + rtol*abs(solution04[1])
        #@test abs(fm - solution04[2]) ≤ prec

        n = 0
        (xm, fm, lo, hi) = Brent.fmin(T, problem13, bounds13...)
        if verb
            @printf("Problem 13: n = %d, x = %.15f ± %.3g, f(x) = %.15f\n",
                    n, xm, (hi - lo)/2, fm)
        end
        @test abs(xm - solution13[1]) ≤ atol + rtol*abs(solution13[1])
        #@test abs(fm - solution13[2]) ≤ prec

        n = 0
        (xm, fm, lo, hi) = Brent.fmin(T, problem18, bounds18...)
        if verb
            @printf("Problem 18: n = %d, x = %.15f ± %.3g, f(x) = %.15f\n",
                    n, xm, (hi - lo)/2, fm)
        end
        @test abs(xm - solution18[1]) ≤ atol + rtol*abs(solution18[1])
        #@test abs(fm - solution18[2]) ≤ prec
    end
end

end # module
