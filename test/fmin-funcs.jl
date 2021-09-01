#
# fmin-funcs.jl -
#
# Define functions for testing univariate optimization.
#
module FminTestFunctions

# Counter of function evaluations.
const cnt = Ref{Int}(0)
incr_nevals(n::Integer=1) = (cnt[] += n; nothing)
reset_nevals() =  (cnt[] = 0; nothing)
get_nevals() = cnt[]

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
struct TestFunc{T<:AbstractFloat,F}
    f::F
    id::String
    a::T
    b::T
    ga::T
    gb::T
    x0::T
    f0::T
    function TestFunc{T}(f::F;
                         id::AbstractString,
                         a::Real, ga::Real=a,
                         b::Real, gb::Real=b,
                         x0::Real, f0::Real=NaN) where {T,F}
        new{T,F}(f, id, a, b, ga, gb, x0,
                 isnan(f0) ? f(T(x0)) : f0)
    end
end
TestFunc(args...; kwds...) = TestFunc{Float64}(args...; kwds...)
TestFunc{T}(f::TestFunc{T}) where {T<:AbstractFloat} = f
TestFunc{T}(f::TestFunc) where {T<:AbstractFloat} =
    TestFunc{T}(func(f); id=ident(f), a=local_lower(f), b=local_upper(f),
                ga=global_lower(f), gb=global_upper(f), x0=xsol(f), f0=fsol(f))

@inline (A::TestFunc{T})(x::Real) where {T<:AbstractFloat} = A(convert(T, x))
@inline (A::TestFunc{T})(x::T) where {T<:AbstractFloat} = (incr_nevals(); func(A)(x))
func(A::TestFunc) = A.f
ident(A::TestFunc) = A.id
local_lower(A::TestFunc) = A.a
local_upper(A::TestFunc) = A.b
global_lower(A::TestFunc) = A.ga
global_upper(A::TestFunc) = A.gb
xsol(A::TestFunc) = A.x0
fsol(A::TestFunc) = A.f0

# Problems 4 AMPGO (http://infinity77.net/global_optimization/test_functions_1d.html).
ampgo_2 = TestFunc(x -> sin(x) + sin(10x/3);
                   id="AMPGO_2",
                   a = 4.3,  b = 6.2,  # for local optimization
                   ga = 2.5, gb = 7.5, # for global optimization
                   x0 = 5.1457352882823, f0 = -1.8995993491521)
ampgo_3 = TestFunc(x -> (begin;
                         s = zero(x);
                         # FIXME: The formula (in AMPGO web page) is for
                         #        `k ∈ 1:6` but the figure and the given
                         #        minimum if for `k ∈ 1:5`.
                         for k in 1:5; s -= k*sin((k + 1)*x + k); end
                         return s;
                         end);
                   id = "AMPGO_3",
                   a = -7.3, b = -6.1, # for local optimization
                   ga = -10, gb = 10,  # for global optimization
                   x0 = -6.7745761455604, f0 = -12.0312494421671)
ampgo_4 = TestFunc(x -> -(16x^2 - 24x + 5)*exp(-x);
                   id = "AMPGO_4", a = 1.9, b = 3.9,
                   x0 = 2.8680339892289, f0 = -3.8504507088002)
ampgo_5 = TestFunc(x -> (3x - oftype(x, 1.4))*sin(18x);
                   id = "AMPGO_5",
                   a = 0.8, b = 1.1,   # for local optimization
                   ga = 0.0, gb = 1.2, # for global optimization
                   x0 = 0.9660858038332, f0 = -1.4890725386896)
ampgo_6 = TestFunc(x -> -(x + sin(x))*exp(-x^2);
                   id = "AMPGO_6",
                   a = -0.6, b = 10,  # for local optimization
                   ga = -10, gb = 10, # for global optimization
                   x0 = 0.6795786601089, f0 = -0.8242393984761)
ampgo_7 = TestFunc(x -> sin(x) + sin(10x/3) + log(x) - oftype(x,0.84)*x + 3;
                   id = "AMPGO_7",
                   a = 4.2, b = 6.1,   # for local optimization
                   ga = 2.7, gb = 7.5, # for global optimization
                   x0 = 5.1997783686004, f0 = -1.6013075464944)
ampgo_8 = TestFunc(x -> (begin;
                         s = zero(x);
                         # FIXME: The formula (in AMPGO web page) is for
                         #        `k ∈ 1:6` but the figure and the given
                         #        minimum if for `k ∈ 1:5`.
                         for k in 1:5; s -= k*cos((k + 1)*x + k); end
                         return s;
                         end);
                   id = "AMPGO_8",
                   a = -7.7, b = -6.15, # for local optimization
                   ga = -10, gb = 10,  # for global optimization
                   x0 = -7.0835064075500, f0 = -14.5080079271950)
ampgo_9 = TestFunc(x -> sin(x) + sin(2x/3);
                   id = "AMPGO_9",
                   a = 13.6, b = 20.4,  # for local optimization
                   ga = 3.1, gb = 20.4, # for global optimization
                   x0 = 17.0391989476448, f0 = -1.9059611187158)
ampgo_10 = TestFunc(x -> -x*sin(x);
                    id = "AMPGO_10",
                    a = 5, b = 10,   # for local optimization
                    ga = 0, gb = 10, # for global optimization
                    x0 = 7.9786657125325, f0 = -7.9167273715878)
ampgo_11 = TestFunc(x -> 2cos(x) + cos(2x);
                    id = "AMPGO_11",
                    a = 0, b = 3,       # for local optimization
                    ga = -π/2, gb = 2π, # for global optimization
                    x0 = 2.0943950957161, f0 = -1.5)
ampgo_12 = TestFunc(x -> sin(x)^3 + cos(x)^3;
                    id = "AMPGO_12",
                    a = 1.6, b = 3.9, # for local optimization
                    ga = 0, gb = 2π, # for global optimization
                    x0 = π, f0 = -1)
ampgo_13 = TestFunc(x -> -x^oftype(x,2/3) - (1 - x^2)^oftype(x,1/3);
                    id = "AMPGO_13",
                    a = 0.001, b = 0.99,
                    x0 = 1/sqrt(2), f0 = -1.5874010519682)
ampgo_14 = TestFunc(x -> -exp(-x)*sin(2π*x);
                    id = "AMPGO_14",
                    a = 0, b = 0.7,
                    ga = 0, gb = 4,
                    x0 = 0.2248803847323, f0 = -0.7886853874087)
ampgo_15 = TestFunc(x -> (x^2 - 5x + 6)/(x^2 + 1);
                    id = "AMPGO_15",
                    a = -0.3, b = 5,
                    ga = -5, gb = 5,
                    x0 = 2.4142135734136, f0 = -0.0355339059327)
ampgo_18 = TestFunc(x -> x ≤ 3 ? (x - 2)^2 : 2log(x - 2) + 1;
                    id = "AMPGO_18", a = 0, b = 6, x0 = 2, f0 = 0)
ampgo_20 = TestFunc(x -> (sin(x) - x)*exp(-x^2);
                    id = "AMPGO_20",
                    a = -1.1, b = 6.2,
                    ga = -10, gb = 10,
                    x0 = 1.1951366409628, f0 = -0.0634905289364)
ampgo_21 = TestFunc(x -> (sin(x) + cos(2x))*x;
                    id = "AMPGO_21",
                    a = 3.1, b = 6.5,
                    ga = 0, gb = 10,
                    x0 = 4.7954086865801, f0 = -9.5083504406331)
ampgo_22 = TestFunc(x -> exp(-3x) - sin(x)^3;
                    id = "AMPGO_22",
                    a = 11.1, b = 17.2,
                    ga = 0, gb = 20,
                    x0 = 14.1371669411515, f0 = -1.0)

# Test functions from GSL (GNU Scientific Library).
gsl_fmin_1 = TestFunc(x -> x^4 - 1;
                      id = "GSL_FMIN_1", a = -3, b = 17, x0 = 0, f0 = 0)
gsl_fmin_2 = TestFunc(x -> sqrt(abs(x));
                      id = "GSL_FMIN_2", a = -2.0, b = 1.5, x0 = 0, f0 = 0)
gsl_fmin_3 = TestFunc(x -> x < 1 ? float(one(x)) : -exp(-x);
                      id = "GSL_FMIN_3", a = -2.0, b = 4.0, x0 = 1.0)
gsl_fmin_4 = TestFunc(x -> x - 30/(1 + 100_000*(x - oftype(x,0.8))^2);
                      id = "GSL_FMIN_4", a = 0.72, b = 2.0,
                      x0 = 0.7999998333408, f0 = -29.2000000833333)

end # module
