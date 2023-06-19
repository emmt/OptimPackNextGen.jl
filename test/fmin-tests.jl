#
# fmin-tests.jl -
#
# Test univariate optimization.
#
module FminTests

export
    TestFunc,
    incr_nevals,
    reset_nevals,
    get_nevals,

    # Getters.
    func,
    ident,
    local_lower,
    local_upper,
    global_lower,
    global_upper,
    xsol,
    fsol,

    # Test functions.
    brent_2, brent_3, brent_4, brent_5,
    michalewicz_1, michalewicz_2,
    ampgo_2, ampgo_3, ampgo_4, ampgo_5, ampgo_6,
    ampgo_7, ampgo_8, ampgo_9, ampgo_10, ampgo_11,
    ampgo_12, ampgo_13, ampgo_14, ampgo_15,
    ampgo_18, ampgo_20, ampgo_21, ampgo_22,
    gsl_fmin_1, gsl_fmin_2, gsl_fmin_3, gsl_fmin_4

using OptimPackNextGen.Brent
using Unitless

# Number of function evaluations.
const nevals = Ref{Int}(0)
tic(x=nothing) = (nevals[] += 1; return x)
reset_nevals() = (nevals[] = 0; nothing)
get_nevals() = nevals[]

struct Point{Tx,Tf}
    type::Symbol
    x::Tx
    fx::Tf
    a::Tx
    b::Tx
    n::Int
end

function Point(type::Symbol; x::Number, fx::Number, a::Number, b::Number, n::Integer=0)
    Tx1 = promote_type(typeof(x), typeof(a),  typeof(b))
    Tf1 = typeof(fx)
    T = float(promote_type(real_type(Tx1), real_type(Tf1)))
    Tx = convert_real_type(T, Tx1)
    Tf = convert_real_type(T, Tf1)
    return Point{Tx,Tf}(type, x, fx, a, b, n)
end

local_min(; kwds...) = Point(:local_min; kwds...)
local_max(; kwds...) = Point(:local_max; kwds...)
global_min(; kwds...) = Point(:global_min; kwds...)
global_max(; kwds...) = Point(:global_max; kwds...)

"""
    obj = TestFunc{T}(f; id="...", a=..., b=.., x0=..., f0=NaN, ga=a, gb=b)

yields an object that wraps objective function `f` to be tested for univariate
minimization.  Keywords must be used to specify the identifier `id`, the bounds
`a` and `b` of the search interval, the (approximate) correct solution `x0`,
and corresponding minimum function value `f0`.  Keywords `ga` and `gb` are to
specify other endpoints for the search interval in global optimization (or
plotting).  All keywords but `f0`, `ga` and `gb` are mandatory.  Parameter `T`
is the floating-point type for computations, `Float64` by default.

The wrapper is intended to bundle the settings for testing the minimization of
the function and to improve type stability when the wrapper is called as
function to evaluate the objective function: `obj(x)` yields `f(convert(T,x))`.
This improved type-stability yields no regressions, on the contrary, it may reduce
allocations and speed-up tests in some cases (e.g., the AMPGO problem 18).

Also when the objective function is called via the wrapper, the global variable
storing the number of calls is automatically incremented.

"""
struct TestFunc{F,Tx,Tf}
    name::String
    func::F
    periodic::Bool
    points::Vector{Point{Tx,Tf}}
end

for func in (:fmin, :fmax)

    @eval Brent.$func(p::TestFunc{F,Tx,Tf}, a::Number, b::Number; kwds...) where {F,Tx,Tf} =
        $func(float(promote_type(real_type(Tx), real_type(Tf))), p, a, b; kwds...)

end

@inline (p::TestFunc)(x::Number) = p.func(x)

function TestFunc(name::AbstractString, func::F, args::Point{Tx,Tf}...;
                  periodic::Bool=false) where {F,Tx,Tf}
    return TestFunc{F,Tx,Tf}(name, func, periodic, collect(args))
end

# Brent's test functions (see Brent's book p. 104).  Brent's 2nd test function
# is a simple parabola whose minimum is at xm=0, this is good to check for
# excessive number of iterations due to excessive precision when xm=0.
const brent_2 = TestFunc(
    "Brent_2",
    x -> tic(x^2),
    local_min(x = 0, fx = 0, a = -1, b = 2))

const brent_3 = TestFunc(
    "Brent_3",
    x -> tic((x + 1)*x^2),
    local_min(x = 0, fx = 0, a = -0.6, b = 2))

const brent_4 = TestFunc(
    "Brent_4",
    x -> tic((x + sin(x))*exp(-x^2)),
    local_min( x = -0.6795786600544, fx = -0.8242393984761, a = -6, b = 1.1),
    global_min(x = -0.6795786600544, fx = -0.8242393984761, a = -10, b = 10))

const brent_5 = TestFunc(
    "Brent_5",
    x -> tic((x - sin(x))*exp(-x^2)),
    local_min( x = -1.1951366354693, fx = -0.0634905289364, a = -6, b = 1.1),
    global_min(x = -1.1951366354693, fx = -0.0634905289364, a = -10, b = 10))

# Michalewicz's functions.
const michalewicz_1 = TestFunc(
    "Michalewicz_1",
    x -> tic(x*sin(10x)),
    local_min( x = 1.7336377924464, fx = -1.7307608607859, a = 1.5, b = 2),
    global_min(x = 1.7336377924464, fx = -1.7307608607859, a = -1, b = 2))

const michalewicz_2 = TestFunc(
    "Michalewicz_2",
    x -> begin
        a = sin(x)
        b = x^2/π
        s = zero(a)
        for i in 1:10; s += sin(b*i)^20; end
        return tic(a*s)
    end,
    local_min( x = 2.5670924755652, fx = 0.2105521936702, a = 2.53, b = 2.61),
    global_min(x = 2.5670924755652, fx = 0.2105521936702, a = 0.7, b = 3))

# Problems in AMPGO (http://infinity77.net/global_optimization/test_functions_1d.html).
const ampgo_2 = TestFunc(
    "AMPGO_2",
    x -> tic(sin(x) + sin(10x/3)),
    local_min( x = 5.1457352882823, fx = -1.8995993491521, a = 4.3, b = 6.2),
    global_min(x = 5.1457352882823, fx = -1.8995993491521, a = 2.5, b = 7.5))

const ampgo_3 = TestFunc(
    "AMPGO_3",
    x -> begin
        s = zero(x)
        # FIXME: The formula (in AMPGO web page) is for `k ∈ 1:6` but the
        #        figure and the given minimum is for `k ∈ 1:5`.
        for k in 1:5; s -= k*sin((k + 1)*x + k); end
        return tic(s)
    end,
    local_min( x = -6.7745761455604, fx = -12.0312494421671, a = -7.3, b = -6.1),
    global_min(x = -6.7745761455604, fx = -12.0312494421671, a = -10, b = 10))

const ampgo_4 = TestFunc(
    "AMPGO_4",
    x -> tic(-(16x^2 - 24x + 5)*exp(-x)),
    local_min( x = 2.8680339892289, fx = -3.8504507088002, a = 1.9, b = 3.9),
    global_min(x = 2.8680339892289, fx = -3.8504507088002, a = 1.9, b = 3.9)) # FIXME:

const ampgo_5 = TestFunc(
    "AMPGO_5",
    x -> tic((3x - oftype(x, 1.4))*sin(18x)),
    local_min( x = 0.9660858038332, fx = -1.4890725386896, a = 0.8, b = 1.1),
    global_min(x = 0.9660858038332, fx = -1.4890725386896, a = 0.0, b = 1.2))

const ampgo_6 = TestFunc(
    "AMPGO_6",
    x -> tic(-(x + sin(x))*exp(-x^2)),
    local_min( x = 0.6795786601089, fx = -0.8242393984761, a = -0.6, b = 10),
    global_min(x = 0.6795786601089, fx = -0.8242393984761, a = -10, b = 10))

const ampgo_7 = TestFunc(
    "AMPGO_7",
    x -> tic(sin(x) + sin(10x/3) + log(x) - oftype(x,0.84)*x + 3),
    local_min( x = 5.1997783686004, fx = -1.6013075464944, a = 4.2, b = 6.1),
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
    local_min( x = -7.0835064075500, fx = -14.5080079271950, a = -7.7, b = -6.15),
    global_min(x = -7.0835064075500, fx = -14.5080079271950, a = -10, b = 10))

const ampgo_9 = TestFunc(
    "AMPGO_9",
    x -> tic(sin(x) + sin(2x/3)),
    local_min( x = 17.0391989476448, fx = -1.9059611187158, a = 13.6, b = 20.4),
    global_min(x = 17.0391989476448, fx = -1.9059611187158, a = 3.1, b = 20.4))

const ampgo_10 = TestFunc(
    "AMPGO_10",
    x -> tic(-x*sin(x)),
    local_min( x = 7.9786657125325, fx = -7.9167273715878, a = 5, b = 10),
    global_min(x = 7.9786657125325, fx = -7.9167273715878, a = 0, b = 10))

const ampgo_11 = TestFunc(
    "AMPGO_11",
    x -> tic(2cos(x) + cos(2x)),
    local_min( x = 2.0943950957161, fx = -1.5, a = 0, b = 3),
    global_min(x = 2.0943950957161, fx = -1.5, a = -π/2, b = 2π))

const ampgo_12 = TestFunc(
    "AMPGO_12",
    x -> tic(sin(x)^3 + cos(x)^3),
    local_min( x = π, fx = -1, a = 1.6, b = 3.9,),
    global_min(x = π, fx = -1, a = 0,   b = 2π))

const ampgo_13 = TestFunc(
    "AMPGO_13",
    x -> tic(-x^oftype(x,2//3) - (1 - x^2)^oftype(x,1//3)),
    local_min(x = 1/sqrt(2), fx = -1.5874010519682, a = 0.001, b = 0.99))

const ampgo_14 = TestFunc(
    "AMPGO_14",
    x -> tic(-exp(-x)*sin(2π*x)),
    local_min( x = 0.2248803847323, fx = -0.7886853874087, a = 0, b = 0.7),
    global_min(x = 0.2248803847323, fx = -0.7886853874087, a = 0, b = 4))

const ampgo_15 = TestFunc(
    "AMPGO_15",
    x -> tic((x^2 - 5x + 6)/(x^2 + 1)),
    local_min( x = 2.4142135734136, fx = -0.0355339059327, a = -0.3, b = 5),
    global_min(x = 2.4142135734136, fx = -0.0355339059327, a = -5, b = 5))

const ampgo_18 = TestFunc(
    "AMPGO_18",
    x -> tic(x ≤ 3 ? (x - 2)^2 : 2log(x - 2) + 1),
    local_min(x = 2, fx = 0, a = 0, b = 6))

const ampgo_20 = TestFunc(
    "AMPGO_20",
    x -> tic((sin(x) - x)*exp(-x^2)),
    local_min( x = 1.1951366409628, fx = -0.0634905289364, a = -1.1, b = 6.2),
    global_min(x = 1.1951366409628, fx = -0.0634905289364, a = -10, b = 10))

const ampgo_21 = TestFunc(
    "AMPGO_21",
    x -> tic((sin(x) + cos(2x))*x),
    local_min( x = 4.7954086865801, fx = -9.5083504406331, a = 3.1, b = 6.5),
    global_min(x = 4.7954086865801, fx = -9.5083504406331, a = 0, b = 10))

const ampgo_22 = TestFunc(
    "AMPGO_22",
    x -> tic(exp(-3x) - sin(x)^3),
    local_min( x = 14.1371669411515, fx = -1.0, a = 11.1, b = 17.2),
    global_min(x = 14.1371669411515, fx = -1.0, a = 0, b = 20))

# Test functions from GSL (GNU Scientific Library).
const gsl_fmin_1 = TestFunc(
    "GSL_FMIN_1",
    x -> tic(x^4 - 1),
    local_min( x = 0, fx = 0, a = -3, b = 17))

const gsl_fmin_2 = TestFunc(
    "GSL_FMIN_2",
    x -> tic(sqrt(abs(x))),
    local_min( x = 0, fx = 0, a = -2.0, b = 1.5))

const gsl_fmin_3 = TestFunc(
    "GSL_FMIN_3",
    x -> tic(x < 1 ? float(one(x)) : -exp(-x)),
    local_min(x = 1.0, fx = NaN, a = -2.0, b = 4.0))

const gsl_fmin_4 = TestFunc(
    "GSL_FMIN_4",
    x -> tic(x - 30/(1 + 100_000*(x - oftype(x,0.8))^2)),
    local_min(x = 0.7999998333408, fx = -29.2000000833333, a = 0.72, b = 2.0))

const test_funcs = (
    brent_2, brent_3, brent_4, brent_5,
    michalewicz_1, michalewicz_2,
    ampgo_2, ampgo_3, ampgo_4, ampgo_5, ampgo_6,
    ampgo_7, ampgo_8, ampgo_9, ampgo_10, ampgo_11,
    ampgo_12, ampgo_13, ampgo_14, ampgo_15,
    ampgo_18, ampgo_20, ampgo_21, ampgo_22,
    gsl_fmin_1, gsl_fmin_2, gsl_fmin_3, gsl_fmin_4)

end # module
