#
# fmin-tests.jl -
#
# Test univariate optimization.
#
module FminTests

using OptimPackNextGen.Brent
using OptimPackNextGen: BraDi, Step
using Test, Unitless, Unitful

# Number of function evaluations.
const nevals = Ref{Int}(0)
tic(x=nothing) = (nevals[] += 1; return x)
reset_nevals() = (nevals[] = 0; nothing)
get_nevals() = nevals[]

# This structure defines a given extremum of a given function.
struct Extremum{Tx,Tf}
    type::Symbol # one of: :local_min, :local_max, :global_min, :global_max
    x::Tx        # correct solution
    fx::Tf       # f(x)
    a::Tx        # interval of search is [a,b]
    b::Tx
    n::Int       # number of samples for BraDi
end

function Extremum(type::Symbol; x::Number, fx::Number, a::Number, b::Number, n::Integer=0)
    Tx1 = promote_type(typeof(x), typeof(a),  typeof(b))
    Tf1 = typeof(fx)
    T = float(promote_type(real_type(Tx1), real_type(Tf1)))
    Tx = convert_real_type(T, Tx1)
    Tf = convert_real_type(T, Tf1)
    return Extremum{Tx,Tf}(type, x, fx, a, b, n)
end

local__min(; kwds...) = Extremum(:local_min; kwds...)
local__max(; kwds...) = Extremum(:local_max; kwds...)
global_min(; kwds...) = Extremum(:global_min; kwds...)
global_max(; kwds...) = Extremum(:global_max; kwds...)

Base.range(p::Extremum) = range(start = p.a, stop = p.b, length = p.n)

"""
    obj = TestFunc(name, func, args...; periodic=false)

yields an callable object that wraps objective function `func` to be tested for
univariate optimization. Arguments `args...` are any number of instances of
`Extremum`.

The wrapper is intended to bundle the settings for testing the optimization of
the function and may be called as a function to evaluate the objective
function: `obj(x)` yields `func(x)`.

To improve type-stability and correctly represent the behavior of the objective
function at a given numerical precision, `func(x)` must be written so as to
perform computations with the same floating-point type as `x`. This is tested
to some extend. Also when the objective function is called via the wrapper, the
global variable storing the number of calls is automatically incremented.

"""
struct TestFunc{F,Tx,Tf}
    name::String
    func::F
    periodic::Bool
    list::Vector{Extremum{Tx,Tf}}
end

@inline (p::TestFunc)(x::Number) = p.func(x)

function TestFunc(name::AbstractString, func::F, args::Extremum{Tx,Tf}...;
                  periodic::Bool=false) where {F,Tx,Tf}
    return TestFunc{F,Tx,Tf}(name, func, periodic, collect(args))
end

#for func in (:fmin, :fmax)
#
#    @eval Brent.$func(p::TestFunc{F,Tx,Tf}, a::Number, b::Number; kwds...) where {F,Tx,Tf} =
#        $func(float(promote_type(real_type(Tx), real_type(Tf))), p, a, b; kwds...)
#
#end

# Brent's test functions (see Brent's book p. 104).  Brent's 2nd test function
# is a simple parabola whose minimum is at xm=0, this is good to check for
# excessive number of iterations due to excessive precision when xm=0.
const brent_2 = TestFunc(
    "Brent_2",
    x -> tic(x^2),
    local__min(x = 0, fx = 0, a = -1, b = 2))

const brent_3 = TestFunc(
    "Brent_3",
    x -> tic((x + 1)*x^2),
    local__min(x = 0, fx = 0, a = -0.6, b = 2))

const brent_4 = TestFunc(
    "Brent_4",
    x -> tic((x + sin(x))*exp(-x^2)),
    local__min(x = -0.6795786600198815, fx = -0.8242393984760767, a = -6,   b = 0.6),
    global_min(x = -0.6795786600198815, fx = -0.8242393984760767, a = -10,  b = 10, n = 3),
    local__max(x = +0.6795786600198815, fx = +0.8242393984760767, a = -0.6, b = 6),
    global_max(x = +0.6795786600198815, fx = +0.8242393984760767, a = -10,  b = 10, n = 3))

const brent_5 = TestFunc(
    "Brent_5",
    x -> tic((x - sin(x))*exp(-x^2)),
    local__min(x = -1.195136641756661, fx = -0.06349052893643988, a = -6,  b = 1.1),
    global_min(x = -1.195136641756661, fx = -0.06349052893643988, a = -10, b = 10, n = 3),
    local__max(x = +1.195136641756661, fx = +0.06349052893643988, a = -1.1,  b = 6),
    global_max(x = +1.195136641756661, fx = +0.06349052893643988, a = -10, b = 10, n = 3))

# Michalewicz's functions.
const michalewicz_1 = TestFunc(
    "Michalewicz_1",
    x -> tic(x*sin(10x)),
    local__min(x = 1.733637792398336, fx = -1.730760860785851, a = 1.5, b = 1.9),
    global_min(x = 1.733637792398336, fx = -1.730760860785851, a = -1,  b = 1.9, n = 12),
    local__max(x = 1.420743672519119, fx =  1.417237411377428, a = 1.2, b = 1.7),
    global_max(x = 1.420743672519119, fx =  1.417237411377428, a = -1,  b = 1.9, n = 12))

const michalewicz_2 = TestFunc(
    "Michalewicz_2",
    x -> begin
        a = sin(x)
        b = x^2/π
        s = zero(a)
        for i in 1:10; s += sin(b*i)^20; end
        return tic(a*s)
    end,
    local__min(x = 2.567092475376642, fx = 0.2105521936702232, a = 2.53, b = 2.61),
    global_min(x = 2.567092475376642, fx = 0.2105521936702232, a = 0.7,  b = 3, n = 22),
    local__max(x = 2.220865159657191, fx = 3.979338598164439,  a = 2.15, b = 2.29),
    global_max(x = 2.220865159657191, fx = 3.979338598164439,  a = 0.7,  b = 3, n = 22))

# Problems in AMPGO (http://infinity77.net/global_optimization/test_functions_1d.html).
const ampgo_2 = TestFunc(
    "AMPGO_2",
    x -> tic(sin(x) + sin(10x/3)),
    local__min(x = 5.145735290256128, fx = -1.899599349152113, a = 4.3, b = 6.2),
    global_min(x = 5.145735290256128, fx = -1.899599349152113, a = 2.5, b = 7.5, n = 7))

const ampgo_3 = TestFunc(
    "AMPGO_3",
    x -> begin
        s = zero(x)
        # FIXME: The formula (in AMPGO web page) is for `k ∈ 1:6` but the
        #        figure and the given minimum is for `k ∈ 1:5`.
        for k in 1:5; s -= k*sin((k + 1)*x + k); end
        return tic(s)
    end,
    periodic = true,
    local__min(x = -0.4913908362593146, fx = -12.03124944216714, a = -1, b = 0),
    global_min(x = -0.4913908362593146, fx = -12.03124944216714, a = -π,  b = π, n = 13))

const ampgo_4 = TestFunc(
    "AMPGO_4",
    x -> tic(-(16x^2 - 24x + 5)*exp(-x)),
    local__min(x = 2.868033988749895, fx = -3.850450708800219, a = 0.7, b = 8.0),
    global_min(x = 2.868033988749895, fx = -3.850450708800219, a = 0.1, b = 8.0, n = 9))

const ampgo_5 = TestFunc(
    "AMPGO_5",
    x -> tic((3x - oftype(x, 1.4))*sin(18x)),
    local__min(x = 0.9660858038268510, fx = -1.489072538689604, a = 0.8, b = 1.1),
    global_min(x = 0.9660858038268510, fx = -1.489072538689604, a = 0.0, b = 1.2, n = 10),
    local__max(x = 1.139043919974300,  fx =  2.010281351381081, a = 1.0, b = 1.2),
    global_max(x = 1.139043919974300,  fx =  2.010281351381081, a = 0.0, b = 1.2, n = 10))

const ampgo_6 = TestFunc(
    "AMPGO_6",
    x -> tic(-(x + sin(x))*exp(-x^2)),
    local__min(x = 0.6795786601089, fx = -0.8242393984761, a = -0.6, b = 10),
    global_min(x = 0.6795786601089, fx = -0.8242393984761, a = -10, b = 10))

const ampgo_7 = TestFunc(
    "AMPGO_7",
    x -> tic(sin(x) + sin(10x/3) + log(x) - oftype(x,0.84)*x + 3),
    local__min(x = 5.1997783686004, fx = -1.6013075464944, a = 4.2, b = 6.1),
    global_min(x = 5.1997783686004, fx = -1.6013075464944, a = 2.7, b = 7.5))

const ampgo_8 = TestFunc(
    "AMPGO_8",
    x -> begin
        s = zero(x)
        # FIXME: The formula (in AMPGO web page) is for `k ∈ 1:6` but the
        #        figure and the given minimum is for `k ∈ 1:5`.
        for k in 1:5; s -= k*cos((k + 1)*x + k); end
        return tic(s)
    end,
    periodic = true,
    local__min(x = -0.8003211004719731, fx = -14.50800792719503, a = -1.4, b = -0.2),
    global_min(x = -0.8003211004719731, fx = -14.50800792719503, a = -pi, b = pi, n = 13))

const ampgo_9 = TestFunc(
    "AMPGO_9",
    x -> tic(sin(x) + sin(2x/3)),
    local__min(x = 17.0391989476448, fx = -1.9059611187158, a = 13.6, b = 20.4),
    global_min(x = 17.0391989476448, fx = -1.9059611187158, a = 3.1, b = 20.4))

const ampgo_10 = TestFunc(
    "AMPGO_10",
    x -> tic(-x*sin(x)),
    local__min(x = 7.9786657125325, fx = -7.9167273715878, a = 5, b = 10),
    global_min(x = 7.9786657125325, fx = -7.9167273715878, a = 0, b = 10))

const ampgo_11 = TestFunc(
    "AMPGO_11",
    x -> tic(2cos(x) + cos(2x)),
    local__min(x = 2.0943950957161, fx = -1.5, a = 0, b = 3),
    global_min(x = 2.0943950957161, fx = -1.5, a = -π/2, b = 2π))

const ampgo_12 = TestFunc(
    "AMPGO_12",
    x -> tic(sin(x)^3 + cos(x)^3),
    local__min(x = π, fx = -1, a = 1.6, b = 3.9,),
    global_min(x = π, fx = -1, a = 0,   b = 2π))

const ampgo_13 = TestFunc(
    "AMPGO_13",
    x -> tic(-x^oftype(x,2//3) - (1 - x^2)^oftype(x,1//3)),
    local__min(x = 1/sqrt(2), fx = -1.5874010519682, a = 0.001, b = 0.99))

const ampgo_14 = TestFunc(
    "AMPGO_14",
    x -> tic(-exp(-x)*sin(2*oftype(x,π)*x)),
    local__min(x = 0.2248803858915620, fx = -0.7886853874086725, a = 0,    b = 0.7),
    global_min(x = 0.2248803858915620, fx = -0.7886853874086725, a = 0,    b = 4),
    local__max(x = 0.7248803858915620, fx =  0.4783618683306960, a = 0.23, b = 1.2),
    global_max(x = 0.7248803858915620, fx =  0.4783618683306960, a = 0,    b = 4),
)

const ampgo_15 = TestFunc(
    "AMPGO_15",
    x -> tic((x^2 - 5x + 6)/(x^2 + 1)),
    local__min(x =  2.414213562373095, fx = -0.03553390593273762, a = -0.4, b = 5),
    global_min(x =  2.414213562373095, fx = -0.03553390593273762, a = -5,   b = 5),
    local__max(x = -0.414213562373095, fx =  7.035533905932738,   a = -5,   b = 2),
    global_max(x = -0.414213562373095, fx =  7.035533905932738,   a = -5,   b = 5))

const ampgo_18 = TestFunc(
    "AMPGO_18",
    x -> tic(x ≤ 3 ? (x - 2)^2 : 2log(x - 2) + 1),
    local__min(x = 2, fx = 0, a = 0, b = 6))

const ampgo_20 = TestFunc(
    "AMPGO_20",
    x -> tic((sin(x) - x)*exp(-x^2)),
    local__min(x =  1.195136641756661, fx = -0.06349052893643988, a = -1.1, b = 6.2),
    global_min(x =  1.195136641756661, fx = -0.06349052893643988, a = -10,  b = 10, n = 3),
    local__max(x = -1.195136641756661, fx =  0.06349052893643988, a = -5.0, b = 1.1),
    global_max(x = -1.195136641756661, fx =  0.06349052893643988, a = -10, b = 10, n = 3))

const ampgo_21 = TestFunc(
    "AMPGO_21",
    x -> tic((sin(x) + cos(2x))*x),
    local__min(x = 4.7954086865801, fx = -9.5083504406331, a = 3.1, b = 6.5),
    global_min(x = 4.7954086865801, fx = -9.5083504406331, a = 0, b = 10))

const ampgo_22 = TestFunc(
    "AMPGO_22",
    x -> tic(exp(-3x) - sin(x)^3),
    local__min(x = 14.1371669411515, fx = -1.0, a = 11.1, b = 17.2),
    global_min(x = 14.1371669411515, fx = -1.0, a = 0, b = 20))

# Test functions from GSL (GNU Scientific Library).
const gsl_fmin_1 = TestFunc(
    "GSL_FMIN_1",
    x -> tic(x^4 - 1),
    local__min(x = 0, fx = -1, a = -3, b = 17))

const gsl_fmin_2 = TestFunc(
    "GSL_FMIN_2",
    x -> tic(sqrt(abs(x))),
    local__min(x = 0, fx = 0, a = -2.0, b = 1.5))

const gsl_fmin_3 = TestFunc(
    # NOTE: This function is discontinuous at the location of the minimum.
    "GSL_FMIN_3",
    x -> tic(x < 1 ? float(one(x)) : -exp(-x)),
    local__min(x = 1.0, fx = -0.3678794411714423, a = -2.0, b = 4.0))

const gsl_fmin_4 = TestFunc(
    "GSL_FMIN_4",
    x -> tic(x - 30/(1 + 100_000*(x - oftype(x,0.8))^2)),
    local__min(x = 0.7999998333333325, fx = -29.20000008333333, a = 0.72, b = 2.0),
    global_min(x = 0.7999998333333325, fx = -29.20000008333333, a = -1,   b = 2),
    local__max(x = 0.7157358311799512, fx = 0.6735444095680056, a = 0.45, b = 0.79),
    global_max(x = 0.7157358311799512, fx = 0.6735444095680056, a = -1,   b = 2))

const test_funcs = (
    brent_2, brent_3, brent_4, brent_5,
    michalewicz_1, michalewicz_2,
    ampgo_2, ampgo_3, ampgo_4, ampgo_5, ampgo_6,
    ampgo_7, ampgo_8, ampgo_9, ampgo_10, ampgo_11,
    ampgo_12, ampgo_13, ampgo_14, ampgo_15,
    ampgo_18, ampgo_20, ampgo_21, ampgo_22,
    gsl_fmin_1, gsl_fmin_2, gsl_fmin_3, gsl_fmin_4)

round_value(val; sigdigits=3, base=10) = round(unitless(val); sigdigits, base)*unit(val)

runtests(; kwds...) = runtests(:fmin, :fmax; kwds...)

function runtests(algs::Symbol...; verb::Bool = false)
    @testset "$(f.name), $T" for f in test_funcs, T in (Float32, Float64)
        for p in f.list
            a, b, xm, fm = p.a, p.b, p.x, p.fx
            rtol = sqrt(eps(T))
            atol = eps(T)*oneunit(xm)
            xtol = 3*rtol*abs(xm) + atol
            ftol = (iszero(fm) ? oneunit(fm) : abs(fm))*rtol
            rtol /= 2 # NOTE: Require a bit more relative precision, this is
                      #       needed for maximizing AMPGO 15-th test function.
            if f === gsl_fmin_1
                # gsl_fmin_1(x) = x^4 - 1 which results in rapid loss of
                # significant digits. Set xtol such that f(xtol) == f(xm) with
                # xm the solution.
                xtol = (eps(T)^T(1//4))/2
            end

            reset_nevals()
            if p.type === :local_min && :fmin ∈ algs
                x, fx, lo, hi, n = if real_type(xm) === T
                    @inferred fmin(f, a, b; rtol, atol)
                else
                    @inferred fmin(T, f, a, b; rtol, atol)
                end
            elseif p.type === :local_max && :fmax ∈ algs
                x, fx, lo, hi, n = if real_type(xm) === T
                    @inferred fmax(f, a, b; rtol, atol)
                else
                    @inferred fmax(T, f, a, b; rtol, atol)
                end
            elseif p.type === :global_min && (:bradi ∈ algs || :step ∈ algs)
                if :bradi ∈ algs
                    r = range(p)
                    isempty(r) && continue
                    periodic = f.periodic
                    x, fx, lo, hi, n = if real_type(xm) === T
                        @inferred BraDi.minimize(f, r; rtol, atol, periodic)
                    else
                        @inferred BraDi.minimize(T, f, r; rtol, atol, periodic)
                    end
                else
                    continue
                end
            elseif p.type === :global_max && (:bradi ∈ algs || :step ∈ algs)
                if :bradi ∈ algs
                    r = range(p)
                    isempty(r) && continue
                    periodic = f.periodic
                    x, fx, lo, hi, n = if real_type(xm) === T
                        @inferred BraDi.maximize(f, r; rtol, atol, periodic)
                    else
                        @inferred BraDi.maximize(T, f, r; rtol, atol, periodic)
                    end
                else
                    continue
                end
            else
                continue
            end
            if verb
                println(rpad(f.name, 16), " T = ", repr(T), ", ", p.type,
                        ", n =", lpad(n, 3),
                        ", x = ", convert_real_type(Float64, x),
                        " ± ", round_value(convert_real_type(Float64, abs(x - xm))),
                        ", f(x) = ", convert_real_type(Float64, fx))
            end
            @test n == get_nevals()
            @test real_type(x) === T
            @test real_type(fx) === T
            fx1 = @inferred f(x)
            @test real_type(fx1) === T
            @test fx == fx1
            @test fx ≈ fm atol=ftol rtol=0
            @test  x ≈ xm atol=xtol rtol=0
            @test lo ≤ x ≤ hi
            if p.type === :local_min || p.type === :global_min
                @test fx ≤ min(f(lo), f(hi))
            else
                @test fx ≥ max(f(lo), f(hi))
            end

        end
    end
end

end # module
