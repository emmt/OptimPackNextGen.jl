# brent-tests.jl -
#
# Tests for Brent's algorithms.
#

module BrentTests

using Test, Printf
import OptimPackNextGen.Brent

# Counter of function evaluations.
const cnt = Ref{Int}(0)

fzero_test0(x::AbstractFloat) = (cnt[] += 1; 1/(x - 3) - 6)
params(::typeof(fzero_test0)) where {T<:AbstractFloat} =
    (3, 4, 19/6)

# Test functions for root finding from Vakkalagadda Satya Sai Prakash,
# "Implementation of Brent-Dekker and A Better Root Finding Method and
# Brent-Dekker Method's Parallelization".

fzero_test1(x::AbstractFloat) = (cnt[] += 1; exp(x)/2 - 5x + 2)
params(::typeof(fzero_test1)) = (1, 4, NaN)

fzero_test2(x::AbstractFloat) = (cnt[] += 1;
                                 -2x^4 + 2x^3 - 16x^2 - 60x + 100)
params(::typeof(fzero_test2)) = (0, 4, NaN)

fzero_test3(x::AbstractFloat) = (cnt[] += 1; exp(x)*cos(x) - x*sin(x))
params(::typeof(fzero_test3)) = (0, 3, NaN)

fzero_test4(x::AbstractFloat) = (cnt[] += 1; x^5 - 5x + 3)
params(::typeof(fzero_test4)) = (0, 1, NaN)

fzero_test5(x::AbstractFloat) = (cnt[] += 1;
                                 (x^3 - oftype(x,0.926)*x^2 +
                                  oftype(x,0.0371)*x + oftype(x,0.043)))
params(::typeof(fzero_test5)) = (0, 0.8, NaN)

fzero_test6(x::AbstractFloat) = (cnt[] += 1;
                                 (-9 + sqrt(99 + 2x - x^2) + cos(2x)))
params(::typeof(fzero_test6)) = (-2, 6, NaN)

fzero_test7(x::AbstractFloat) = (cnt[] += 1; sin(cosh(x)))
params(::typeof(fzero_test7)) = (0, 2, NaN)

fzero_test8(x::AbstractFloat) = (cnt[] += 1; exp(-exp(-x)) - x)
params(::typeof(fzero_test8)) = (0, 1, NaN)

# Test functions from the GNU Scientific Library (GSL).

fzero_gsl1(x::AbstractFloat) = (cnt[] += 1; x^20 - 1)
params(::typeof(fzero_gsl1)) = (0.1, 2.0, 1.0)

fzero_gsl2(x::AbstractFloat) = (cnt[] += 1; sqrt(abs(x))*sign(x))
params(::typeof(fzero_gsl2)) = (-1.0/3.0, 1.0, 0.0)

fzero_gsl3(x::AbstractFloat) = (cnt[] += 1; x^2 - oftype(x, 1e-8))
params(::typeof(fzero_gsl3)) = (0, 1, sqrt(1e-8))

fzero_gsl4(x::AbstractFloat) = (cnt[] += 1; x*exp(-x))
params(::typeof(fzero_gsl4)) = (-1.0/3.0, 2.0, 0.0)

fzero_gsl5(x::AbstractFloat) = (cnt[] += 1; (x - 1)^7)
params(::typeof(fzero_gsl5)) = (0, 3, 1) # (0.9995, 1.0002, 1)

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
    for T in (Float32, Float64)
        k = 0
        for fn in (fzero_test0,
                   fzero_test1, fzero_test2, fzero_test3, fzero_test4,
                   fzero_test5, fzero_test6, fzero_test7, fzero_test8,
                   fzero_gsl1, fzero_gsl2, fzero_gsl3, fzero_gsl4, fzero_gsl5)
            rtol = 4*eps(T)
            atol = eps(T)
            prec = sqrt(eps(T))
            (a, b, x0) = params(fn)
            cnt[] = 0
            (x, fx) = Brent.fzero(T, fn, a, b; rtol = rtol, atol = atol)
            if verb
                if k ≤ 8
                    name = "fzero_test$k"
                else
                    name = "fzero_gsl$(k - 8)"
                end
                @printf("%-12s T = %s, n = %3d, x = % .15f, f(x) = % .15f\n",
                        name, repr(T), cnt[], x, fx)
            end
            @test abs(fx) ≈ zero(T) rtol=zero(T) atol=prec
            if !isnan(x0)
                @test x ≈ T(x0) rtol=rtol atol=atol
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
