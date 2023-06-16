# brent-tests.jl -
#
# Tests for Brent's algorithms.
#

module BrentTests

using Test, Printf
using OptimPackNextGen.Brent

# Counter of function evaluations.
const cnt = Ref{Int}(0)

# Yields `y` converted to a floating-point type according to the type of `x`.
fp(x::Number, y::Real) = convert(float(typeof(x)), y)

fzero_test0(x::Real) = (cnt[] += 1; 1/(x - 3) - 6)
params(::typeof(fzero_test0)) = (3, 4, 19/6)

# Test functions for root finding from Vakkalagadda Satya Sai Prakash,
# "Implementation of Brent-Dekker and A Better Root Finding Method and
# Brent-Dekker Method's Parallelization".

fzero_test1(x::Real) = (cnt[] += 1; exp(x)/2 - 5x + 2)
params(::typeof(fzero_test1)) = (1, 4, NaN)

fzero_test2(x::Real) = (cnt[] += 1;
                        -2x^4 + 2x^3 - 16x^2 - 60x + 100)
params(::typeof(fzero_test2)) = (0, 4, NaN)

fzero_test3(x::Real) = (cnt[] += 1; exp(x)*cos(x) - x*sin(x))
params(::typeof(fzero_test3)) = (0, 3, NaN)

fzero_test4(x::Real) = (cnt[] += 1; x^5 - 5x + 3)
params(::typeof(fzero_test4)) = (0, 1, NaN)

fzero_test5(x::Real) = (cnt[] += 1;
                        (x^3 - fp(x,0.926)*x^2 +
                         fp(x,0.0371)*x + fp(x,0.043)))
params(::typeof(fzero_test5)) = (0, 0.8, NaN)

fzero_test6(x::Real) = (cnt[] += 1;
                        (-9 + sqrt(99 + 2x - x^2) + cos(2x)))
params(::typeof(fzero_test6)) = (-2, 6, NaN)

fzero_test7(x::Real) = (cnt[] += 1; sin(cosh(x)))
params(::typeof(fzero_test7)) = (0, 2, NaN)

fzero_test8(x::Real) = (cnt[] += 1; exp(-exp(-x)) - x)
params(::typeof(fzero_test8)) = (0, 1, NaN)

# Test functions from the GNU Scientific Library (GSL).

fzero_gsl1(x::Real) = (cnt[] += 1; x^20 - 1)
params(::typeof(fzero_gsl1)) = (0.1, 2.0, 1.0)

fzero_gsl2(x::Real) = (cnt[] += 1; sqrt(abs(x))*sign(x))
params(::typeof(fzero_gsl2)) = (-1.0/3.0, 1.0, 0.0)

fzero_gsl3(x::Real) = (cnt[] += 1; x^2 - fp(x, 1e-8))
params(::typeof(fzero_gsl3)) = (0, 1, sqrt(1e-8))

fzero_gsl4(x::Real) = (cnt[] += 1; x*exp(-x))
params(::typeof(fzero_gsl4)) = (-1.0/3.0, 2.0, 0.0)

fzero_gsl5(x::Real) = (cnt[] += 1; (x - 1)^7)
params(::typeof(fzero_gsl5)) = (0, 3, 1) # (0.9995, 1.0002, 1)

# Problem 4 from AMPGO, xm = 2.868034, fm = -3.85045:
problem04(x) = (cnt[] += 1; -(16x^2 - 24x + 5)*exp(-x))
bounds(::typeof(problem04)) = [1.9, 3.9]
solution(::typeof(problem04)) = [2.868034, -3.85045]

# Problem 13 from AMPGO, xm = 1/sqrt(2), fm = -1.5874:
problem13(x) = (cnt[] += 1; -x^(2/3) - (1 - x^2)^(1/3))
bounds(::typeof(problem13)) = [0.001, 0.99]
solution(::typeof(problem13)) = [1/sqrt(2), -1.5874]

# Problem 18 from AMPGO, xm = 2, fm = 0:
problem18(x) = (cnt[] += 1; x ≤ 3 ? (x - 2)^2 : 2log(x - 2) + 1)
bounds(::typeof(problem18)) = [0, 6]
solution(::typeof(problem18)) = [2, 0]

verb = true
@testset "Brent fzero" begin
    let f = fzero_test0
        (a, b, x0) = params(f)
        @test_throws ArgumentError fzero(f, a, b; rtol=0)
        @test_throws ArgumentError fzero(f, a, b; rtol=1)
        @test_throws ArgumentError fzero(f, a, b; atol=NaN)
    end
    k = 0
    for f in (fzero_test0,
              fzero_test1, fzero_test2, fzero_test3, fzero_test4,
              fzero_test5, fzero_test6, fzero_test7, fzero_test8,
              fzero_gsl1, fzero_gsl2, fzero_gsl3, fzero_gsl4, fzero_gsl5)
        for T in (Float32, Float64)
            (a, b, x0) = params(f)
            rtol = Brent.fzero_rtol(T)
            atol = (x0 == 0 ? eps(T) : Brent.fzero_atol(T))
            prec = sqrt(eps(T))
            cnt[] = 0
            (x, fx, n) = if T === Float64
                @inferred fzero(f, a, b; rtol = rtol, atol = atol)
            else
                @inferred fzero(T, f, a, b; rtol = rtol, atol = atol)
            end
            @test n == cnt[]
            if verb
                if k ≤ 8
                    name = "fzero_test$k"
                else
                    name = "fzero_gsl$(k - 8)"
                end
                @printf("%-12s T = %s, n = %3d, x = % .15f, f(x) = % .15f\n",
                        name, repr(T), n, x, fx)
            end
            @test abs(fx) ≈ zero(T) rtol=zero(T) atol=prec
            if !isnan(x0)
                @test x ≈ T(x0) rtol=rtol atol=atol
            end
            fa = f(a)
            (x1, f1, n1) =  if T === Float64
                @inferred fzero(f, a, fa, b; rtol = rtol, atol = atol)
            else
                @inferred fzero(T, f, a, fa, b; rtol = rtol, atol = atol)
            end
            @test x1 == x
            @test n1 == n - 1
            fb = f(b)
            (x2, f2, n2) =  if T === Float64
                @inferred fzero(f, a, fa, b, fb; rtol = rtol, atol = atol)
            else
                @inferred fzero(T, f, a, fa, b, fb; rtol = rtol, atol = atol)
            end
            @test x2 == x
            @test n2 == n - 2
        end
        k += 1
    end
end

@testset "Brent fmin" begin
    for (f, id) in ((problem04,  4),
                    (problem13, 13),
                    (problem18, 18),)
        for (T, prec) in ((Float16, 3e-2),
                          (Float32, 3e-6),
                          (Float64, 6e-15))
            atol = Brent.fmin_atol(T)
            rtol = Brent.fmin_rtol(T)

            cnt[] = 0
            (xm, fm, lo, hi, n) = if T === Float64
                @inferred fmin(f, bounds(f)...)
            else
                @inferred fmin(T, f, bounds(f)...)
            end
            @test cnt[] == n
            (xm_true, fm_true) = solution(f)
            if verb
                println("Problem $id (T=$T):")
                @printf(" ├─ nevals = %d\n", n)
                @printf(" ├─ x_approx = %.15f ± %.3e\n", xm, (hi - lo)/2)
                @printf(" ├─ x_true   = %.15f\n", xm_true)
                @printf(" ├─ x_error  = %.3e\n", abs(xm - xm_true))
                @printf(" ├─ f(x_approx) = %.15f\n", fm)
                @printf(" ├─ f(x_true)   = %.15f\n", fm_true)
                @printf(" └─ f_error     = %.3e\n", abs(fm - fm_true))
            end
            @test abs(xm - xm_true) ≤ atol + rtol*abs(xm_true)
            #@test abs(fm - solution(f)[2]) ≤ prec
        end
        verb && println()
    end
end

end # module
