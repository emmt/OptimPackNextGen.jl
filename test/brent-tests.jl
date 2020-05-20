# brent-tests.jl -
#
# Tests for Brent's algorithms.
#

module BrentTests

using Test, Printf
import OptimPackNextGen.Brent

# Counter of function evaluations.
const cnt = Ref{Int}(0)

fzero_test0(x) = (cnt[] += 1; 1/(x - 3) - 6)
interval(::typeof(fzero_test0)) = (3, 4)
solution(::Type{T}, ::typeof(fzero_test0)) where {T<:AbstractFloat} = T(19)/T(6)

# Test functions for root finding from Vakkalagadda Satya Sai Prakash,
# "Implementation of Brent-Dekker and A Better Root Finding Method and
# Brent-Dekker Method's Parallelization".

fzero_test1(x::AbstractFloat) = (cnt[] += 1; exp(x)/2 - 5x + 2)
interval(::typeof(fzero_test1)) = (1, 4)

fzero_test2(x::AbstractFloat) = (cnt[] += 1;
                                 -2x^4 + 2x^3 - 16x^2 - 60x + 100)
interval(::typeof(fzero_test2)) = (0, 4)

fzero_test3(x::AbstractFloat) = (cnt[] += 1; exp(x)*cos(x) - x*sin(x))
interval(::typeof(fzero_test3)) = (0, 3)

fzero_test4(x::AbstractFloat) = (cnt[] += 1; x^5 - 5x + 3)
interval(::typeof(fzero_test4)) = (0, 1)

fzero_test5(x::AbstractFloat) = (cnt[] += 1;
                                 (x^3 - oftype(x,0.926)*x^2 +
                                  oftype(x,0.0371)*x + oftype(x,0.043)))
interval(::typeof(fzero_test5)) = (0, 0.8)

fzero_test6(x::AbstractFloat) = (cnt[] += 1;
                                 (-9 + sqrt(99 + 2x - x^2) + cos(2x)))
interval(::typeof(fzero_test6)) = (-2, 6)

fzero_test7(x::AbstractFloat) = (cnt[] += 1; sin(cosh(x)))
interval(::typeof(fzero_test7)) = (0, 2)

fzero_test8(x::AbstractFloat) = (cnt[] += 1; exp(-exp(-x)) - x)
interval(::typeof(fzero_test8)) = (0, 1)


# Problem 4 from AMPGO, xm = 2.868034, fm = -3.85045:
problem04(x) = (cnt[] += 1; -(16x^2 - 24x + 5)*exp(-x))
bounds04 = [1.9, 3.9]
solution04 = [2.868034, -3.85045]

# Problem 13 from AMPGO, xm = 1/sqrt(2), fm = -1.5874:
problem13(x) = (cnt[] += 1; -x^(2/3) - (1 - x^2)^(1/3))
bounds13 = [0.001, 0.99]
solution13 = [1/sqrt(2), -1.5874]

# Problem 18 from AMPGO, xm = 2, fm = 0:
problem18(x) = (cnt[] += 1; x ≤ 3 ? (x - 2)^2 : 2log(x - 2) + 1)
bounds18 = [0, 6]
solution18 = [2, 0]

verb = true
@testset "Brent fzero" begin
    for (T, prec) in ((Float16, 3e-2),
                      (Float32, 3e-6),
                      (Float64, 6e-15))
        k = 0
        for fn in (fzero_test0,
                   fzero_test1, fzero_test2, fzero_test3, fzero_test4,
                   fzero_test5, fzero_test6, fzero_test7, fzero_test8)
            (a, b) = interval(fn)
            cnt[] = 0
            (x, fx) = Brent.fzero(T, fn, a, b)
            if verb
                @printf("fzero_test%d: T = %s, n = %2d, x = % .15f, f(x) = % .15f\n",
                        k, repr(T), cnt[], x, fx)
            end
            @test abs(fx) ≤ prec
            if fn === fzero_test0
                x0 = solution(T, fn)
                @test abs(x - x0) ≤ eps(T)
            end
            k += 1
        end
    end
end

@testset "Brent fmin" begin
    for (T, prec) in ((Float16, 3e-2),
                      (Float32, 3e-6),
                      (Float64, 6e-15))
        atol = Brent.fmin_atol(T)
        rtol = Brent.fmin_rtol(T)

        cnt[] = 0
        (xm, fm, lo, hi) = Brent.fmin(T, problem04, bounds04...)
        if verb
            @printf("Problem 4: n = %d, x = %.15f ± %.3g, f(x) = %.15f\n",
                    cnt[], xm, (hi - lo)/2, fm)
        end
        @test abs(xm - solution04[1]) ≤ atol + rtol*abs(solution04[1])
        #@test abs(fm - solution04[2]) ≤ prec

        cnt[] = 0
        (xm, fm, lo, hi) = Brent.fmin(T, problem13, bounds13...)
        if verb
            @printf("Problem 13: n = %d, x = %.15f ± %.3g, f(x) = %.15f\n",
                    cnt[], xm, (hi - lo)/2, fm)
        end
        @test abs(xm - solution13[1]) ≤ atol + rtol*abs(solution13[1])
        #@test abs(fm - solution13[2]) ≤ prec

        cnt[] = 0
        (xm, fm, lo, hi) = Brent.fmin(T, problem18, bounds18...)
        if verb
            @printf("Problem 18: n = %d, x = %.15f ± %.3g, f(x) = %.15f\n",
                    cnt[], xm, (hi - lo)/2, fm)
        end
        @test abs(xm - solution18[1]) ≤ atol + rtol*abs(solution18[1])
        #@test abs(fm - solution18[2]) ≤ prec
    end
end

end # module
