# brent-tests.jl -
#
# Tests for Brent's algorithms.
#

module BrentTests

using Test, Printf
using OptimPackNextGen.Brent

include("fzero-tests.jl")
@testset "Brent fzero" FzeroTests.runtests(verb=false)

# Counter of function evaluations.
const cnt = Ref{Int}(0)

const verb = true

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
