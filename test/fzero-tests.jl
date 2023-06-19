#
# fzero-tests.jl -
#
# Test univariate root finding methods.
#
module FzeroTests

using Test, Unitful, Unitless

using OptimPackNextGen.Brent

struct TestFunc{F,Tx<:Number}
    name::String
    func::F
    a::Tx
    b::Tx
    x0::Tx
    function TestFunc(name::AbstractString, func::F, a::Number, b::Number, x0::Number=NaN) where {F}
        Tx = float(promote_type(typeof(a), typeof(b), typeof(x0)))
        return new{F,Tx}(name, func, a, b, x0)
    end
end
(p::TestFunc)(x::Number) = p.func(x)

params(p::TestFunc) = (p.a, p.b, p.x0)

const nevals = Ref{Int}(0)
tic(x=nothing) = (nevals[] += 1; return x)
reset_nevals() = (nevals[] = 0; nothing)
get_nevals() = nevals[]

Brent.fzero(p::TestFunc, a::Number=p.a, b::Number=p.b; kwds...) =
    fzero(p.func, a, b; kwds...)
Brent.fzero(::Type{T}, p::TestFunc, a::Number=p.a, b::Number=p.b; kwds...) where {T<:AbstractFloat} =
    fzero(T, p.func, a, b; kwds...)

# Yields `y` converted to a floating-point type according to the type of `x`.
fp(x::Number, y::Real) = convert(float(typeof(x)), y)

const fzero_test1 = TestFunc(
    "fzero_test1",
    (x) -> tic(1/(x - 3) - 6),
    3, 4, 19/6)

const fzero_test2 = TestFunc(
    "fzero_test2",
    (x) -> tic(exp(-x) - cos(x)),
    -4.7, 1.1, 0.0)

const fzero_test3 = TestFunc(
    "fzero_test3",
    (x) -> tic(log(x^2 + x + 2) - x + 1),
    -1.8, 5.7, 4.152590736757158)

const fzero_test4 = TestFunc(
    "fzero_test4",
    (x) -> tic(sin(x)^2 - x^2 + 1),
    -1.3, 7.2, 1.4044916482153411)

const fzero_test5 = TestFunc(
    "fzero_test5",
    (x) -> tic(exp(-x^2 + x + 2) - cos(x + 1) + x^3 + 1),
    -2.1, 2.7, -1.0)

const fzero_test6 = TestFunc(
    "fzero_test6",
    (x) -> tic(x^11 + x + 1),
    -2.7, 3.2, -0.844397528792023)

const fzero_test7 = TestFunc(
    "fzero_test7",
    (x) -> tic((x - 2)*(x^10 + x + 1)*exp(x - 1)),
    -2.7, 5.2, 2.0)

# Like f6 but with units.
const fzero_test8 = TestFunc(
    "fzero_test8",
    (x) -> tic(x^11 + x*u"cm^10" + 1u"cm^11"),
    -2.7u"cm", 52u"mm", -0.844397528792023u"cm")

# Test functions for root finding from Vakkalagadda Satya Sai Prakash,
# in "Implementation of Brent-Dekker and A Better Root Finding Method and
# Brent-Dekker Method's Parallelization".

const fzero_prakash1 = TestFunc(
    "fzero_prakash1",
    (x) -> tic(exp(x)/2 - 5x + 2),
    1, 6, 3.401795803857807)

const fzero_prakash2 = TestFunc(
    "fzero_prakash2",
    (x) -> tic(-2x^4 + 2x^3 - 16x^2 - 60x + 100),
    -2, 4, 1.240787113746981)

const fzero_prakash3 = TestFunc(
    "fzero_prakash3",
    (x) -> tic(exp(x)*cos(x) - x*sin(x)),
    2, 6, 4.668600322499089)

const fzero_prakash4 = TestFunc(
    "fzero_prakash4",
    (x) -> tic(x^5 - 5x + 3),
    -1.6, 1.2, 0.6180339887498948)

const fzero_prakash5 = TestFunc(
    "fzero_prakash5",
    (x) -> tic((x^3 - fp(x,0.926)*x^2 + fp(x,0.0371)*x + fp(x,0.043))),
    -0.1, 0.8, 0.2910955026957230)

const fzero_prakash6 = TestFunc(
    "fzero_prakash6",
    (x) -> tic((-9 + sqrt(99 + 2x - x^2) + cos(2x))),
    1.7, 5.5, 4.178182868653094)

const fzero_prakash7 = TestFunc(
    "fzero_prakash7",
    (x) -> tic(sin(cosh(x))),
    -1.0, 2.5, 1.811526272460853)

const fzero_prakash8 = TestFunc(
    "fzero_prakash8",
    (x) -> tic(exp(-exp(-x)) - x),
    0, 1, 0.5671432904097839)

# Test functions from the GNU Scientific Library (GSL).

const fzero_gsl1 = TestFunc(
    "fzero_gsl1",
    (x) -> tic(x^20 - 1),
    0.1, 2.0, 1.0)

const fzero_gsl2 = TestFunc(
    "fzero_gsl2",
    (x) -> tic(sqrt(abs(x))*sign(x)),
    # NOTE: Since the solution is 0, setting atol to a sensible value is
    #       mandatory.
    -1.0/3.0, 1.0, 0.0)

const fzero_gsl3 = TestFunc(
    "fzero_gsl3",
    (x) -> tic(x^2 - fp(x, 1e-8)),
    0, 1, 1e-4)

const fzero_gsl4 = TestFunc(
    "fzero_gsl4",
    (x) -> tic(x*exp(-x)),
    # NOTE: Since the solution is 0, setting atol to a sensible value is
    #       mandatory.
    -1.0, 5.0, 0.0)

const fzero_gsl5 = TestFunc(
    "fzero_gsl5",
    (x) -> tic((x - 1)^7),
    0, 3, 1) # (0.9995, 1.0002, 1)

const fzero_funcs =  (fzero_test1, fzero_test2, fzero_test3, fzero_test4, fzero_test5,
                      fzero_test6, fzero_test7, fzero_test8,
                      fzero_prakash1, fzero_prakash2, fzero_prakash3, fzero_prakash4,
                      fzero_prakash5, fzero_prakash6, fzero_prakash7, fzero_prakash8,
                      fzero_gsl1, fzero_gsl2, fzero_gsl3, fzero_gsl4, fzero_gsl5)

function runtests(; verb::Bool = false)
    @testset "$(f.name)" for f in fzero_funcs
        if f === fzero_test1
            (a, b, x0) = params(f)
            @test_throws ArgumentError fzero(f, a, b; rtol=0)
            @test_throws ArgumentError fzero(f, a, b; rtol=1)
            @test_throws ArgumentError fzero(f, a, b; atol=NaN)
        end
        for T in (Float32, Float64)
            (a, b, x0) = params(f)
            Tdef = float(promote_type(typeof(a), typeof(b)))
            rtol = Brent.fzero_rtol(T)
            atol = (x0 == 0 ? eps(T) : floatmin(T))*oneunit(typeof(x0))
            ftol = sqrt(eps(T))*oneunit(f(x0))
            reset_nevals()
            (x, fx, n) = if T === Tdef
                @inferred fzero(f, a, b; rtol = rtol, atol = atol)
            else
                @inferred fzero(T, f, a, b; rtol = rtol, atol = atol)
            end
            if verb
                println(rpad(f.name, 16), " T = ", repr(T), ", n = ", n, ", x = ", x, ", f(x) = ", fx)
            end
            @test real_type(x) === T
            @test real_type(fx) === T
            @test n == get_nevals()
            @test abs(fx) ≤ ftol # fx ≈ zero(fx)
            @test x ≈ x0 rtol=rtol atol=atol

            if x0 != a && x0 != b
                # Check with fa = f(a) and/or fb = f(b) specified.
                fa = f(a)
                fb = f(b)
                #
                reset_nevals()
                (x1, f1, n1) =  if T === Tdef
                    @inferred fzero(f, a, undef, b; rtol = rtol, atol = atol)
                else
                    @inferred fzero(T, f, a, undef, b; rtol = rtol, atol = atol)
                end
                @test x1 == x
                @test n1 == n
                #
                reset_nevals()
                (x1, f1, n1) =  if T === Tdef
                    @inferred fzero(f, a, undef, b, undef; rtol = rtol, atol = atol)
                else
                    @inferred fzero(T, f, a, undef, b, undef; rtol = rtol, atol = atol)
                end
                @test x1 == x
                @test n1 == n
                #
                reset_nevals()
                (x1, f1, n1) =  if T === Tdef
                    @inferred fzero(f, a, fa, b; rtol = rtol, atol = atol)
                else
                    @inferred fzero(T, f, a, fa, b; rtol = rtol, atol = atol)
                end
                @test x1 == x
                @test n1 == n - 1
                #
                reset_nevals()
                (x1, f1, n1) =  if T === Tdef
                    @inferred fzero(f, a, fa, b, undef; rtol = rtol, atol = atol)
                else
                    @inferred fzero(T, f, a, fa, b, undef; rtol = rtol, atol = atol)
                end
                @test x1 == x
                @test n1 == n - 1
                #
                reset_nevals()
                (x1, f1, n1) =  if T === Tdef
                    @inferred fzero(f, a, undef, b, fb; rtol = rtol, atol = atol)
                else
                    @inferred fzero(T, f, a, undef, b, fb; rtol = rtol, atol = atol)
                end
                @test x1 == x
                @test n1 == n - 1
                #
                reset_nevals()
                (x1, f1, n1) =  if T === Tdef
                    @inferred fzero(f, a, fa, b, fb; rtol = rtol, atol = atol)
                else
                    @inferred fzero(T, f, a, fa, b, fb; rtol = rtol, atol = atol)
                end
                @test x1 == x
                @test n1 == n - 2
            end
            if f === fzero_test5
                # Check that algorithm makes a single evaluation if the first
                # end-point is an exact zero.
                reset_nevals()
                (x, fx, n) = fzero(f, x0, b; rtol = rtol, atol = atol)
                @test x == x0
                @test iszero(fx)
                @test n == 1
                reset_nevals()
                (x, fx, n) = fzero(f, a, x0; rtol = rtol, atol = atol)
                @test x == x0
                @test iszero(fx)
                @test n == 2
            end
        end
    end
end

end # module
