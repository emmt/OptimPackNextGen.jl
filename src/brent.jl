#
# brent.jl --
#
# Find a local root or a local minimum of an univariate function by Brent's
# methods described in:
#
# [1] Richard Brent, "Algorithms for minimization without derivatives,"
#     Prentice-Hall, inc. (1973).
#
#-------------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 1973, Richard Brent.
# Copyright (C) 2015-2023, Éric Thiébaut.
#

module Brent

export fzero, fmin, fmax, fminbrkt, fmaxbrkt

using Base: @pure
using Unitless

# Type for undefined argument/option.
const Undef = typeof(undef)

# goldstep = 1/φ^2 = 2 - φ ≈ 0.3812
import Base.MathConstants: φ
@pure goldstep(::Type{T}) where {T<:AbstractFloat} = one(T)/convert(T, φ)^2

"""
    Brent.fzero([T,] f, a, fa=f(a), b, fb=f(b)) -> (x, fx, lo, hi, nf)

applies the Van Wijngaarden–Dekker–Brent method for finding a zero of the
function `f(x)` in the interval `[a,b]`. Optional arguments `fa = f(a)` and/or
`fb = f(b)` may be specified to save computations, they may also be specified
as `undef` if the corresponding function value is not yet known. Optional
argument `T` is to specify the floating-point type used for computations (by
default, `T` is determined from the numeric type of `a`, `fa`, `b`, and `fb`).

`f(a)` and `f(b)` must have opposite signs (an exception is thrown if this does
not hold). The method returns a 5-tuple: a zero `x` in the given interval
`[a,b]` to within the tolerance `rtol*abs(x) + atol`, the corresponding
function value `f(x)`, `lo` and `hi` lower and upper bounds for the solution,
and the number `nf` of calls to `f`. Parameters `atol` and `rtol` can be
specified by keywords. If `rtol = eps(T)`, the machine relative precision, the
error is approximately bounded by:

    abs(x - z) ≤ 3*rtol*abs(z) + 2*atol

with `z` the exact solution. To avoid that rounding errors prevent convergence,
`rtol ≥ eps(T)` and `atol > 0` should hold. By default, `rtol = eps(T)` and
`atol` is `eps(T)*abs(b - a)`.

!!! warning
    If `fa` and/or `fb` are specified, they should have been computed with the
    floating-point type `T`. However and since the code takes care of type
    stability, what really matters is that the signs of `fa` and `fb` are
    correct.

!!! note
    The variable `x` and the function value `f(x)` must be real-valued but may
    be quantities with units as provided by the `Unitful` package.

The `fzero` method is based on Richard Brent's F77 code ZEROIN which itself is
a slightly modified translation of the Algol 60 procedure ZERO given in:

> Richard Brent, "Algorithms for minimization without derivatives",
> Prentice-Hall, inc. (1973).

"""
fzero(f, a::Number, b::Number; kwds...) = fzero(f, a, undef, b, undef; kwds...)

function fzero(f,
               a::Number, fa::Union{Number,Undef},
               b::Number, fb::Union{Number,Undef} = undef; kwds...)
    return fzero(fzero_float(a, fa, b, fb), f, a, fa, b, fb; kwds...)
end

fzero(::Type{T}, f, a::Number, b::Number; kwds...) where {T<:AbstractFloat} =
    fzero(T, f, a, undef, b, undef; kwds...)

function fzero(::Type{T}, f,
               a::Number, fa::Union{Number,Undef},
               b::Number, fb::Union{Number,Undef} = undef;
               kwds...) where {T<:AbstractFloat}
    # Convert bounds and get tolerances, then initialize search.
    Tx = convert_real_type(T, promote_typeof(a, b))
    a = convert(Tx, a)
    b = convert(Tx, b)
    atol, rtol = fzero_tolerances(a, b; kwds...)
    return _fzero_start(f, a, fa, b, fb, atol, rtol, 0)
end

# Initialize search with no function value.
function _fzero_start(f, a::Tx, fa::Undef, b::Tx, fb::Undef, atol::Tx, rtol::T,
                      eval::Int) where {Tx<:Number,T<:AbstractFloat}
    return _fzero_start(f, a, f(a), b, undef, atol, rtol, eval + 1)
end

# Initialize search with one function value.
function _fzero_start(f, a::Tx, fa::Number, b::Tx, fb::Undef, atol::Tx, rtol::T,
                      eval::Int) where {Tx<:Number,T<:AbstractFloat}
    Tf = convert_real_type(T, typeof(fa))
    fa = convert(Tf, fa)
    iszero(fa) && return (a, fa, a, a, eval)
    fb = convert(Tf, f(b))
    eval += 1
    iszero(fb) && return (b, fb, b, b, eval)
    return _fzero_search(f, a, fa, b, fb, atol, rtol, eval)
end
function _fzero_start(f, a::Tx, fa::Undef, b::Tx, fb::Number, atol::Tx, rtol::T,
                      eval::Int) where {Tx<:Number,T<:AbstractFloat}
    return _fzero_start(f, b, fb, a, fa, atol, rtol, eval)
end

# Initialize search with two function values.
function _fzero_start(f, a::Tx, fa::Number, b::Tx, fb::Number, atol::Tx, rtol::T,
                      eval::Int) where {Tx<:Number,T<:AbstractFloat}
    Tf = convert_real_type(T, promote_typeof(fa, fb))
    fa = convert(Tf, fa)
    iszero(fa) && return (a, fa, a, a, eval)
    fb = convert(Tf, fb)
    iszero(fb) && return (b, fb, b, b, eval)
    return _fzero_search(f, a, fa, b, fb, atol, rtol, eval)
end

# Brent's fzero method when f(a) and f(b) have been checked for early
# termination and when all parameters have correct numeric types.
function _fzero_search(f, a::Tx, fa::Tf, b::Tx, fb::Tf, atol::Tx, rtol::T,
                       eval::Int) where {T<:AbstractFloat,Tx<:Number,Tf<:Number}
    # Check the assumptions and the tolerance parameters.
    (fa > zero(fa)) == (fb > zero(fb)) && error("f(a) and f(b) must have different signs")

    # Initialize.
    c, fc = a, fa
    e = d = b - a

    # Loop to improve the interval bracketing the root.
    while true
        # Make sure B is the best point so far.
        if abs(fc) < abs(fb)
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa
        end

        # Compute tolerance.
        tol = rtol*abs(b) + atol
        # NOTE: In Brent's book, the tolerance is denoted `δ` and is given by:
        #
        #     δ = 2⋅ϵ⋅abs(x) + t
        #
        # with `ϵ = eps(T)/2` the relative machine precision halved for rounded
        # arithmetic for computations done with floating-point type `T` (see
        # Eq. (2.9) p. 51 in Brent's book) and `t > 0` chosen by the caller.
        # This corresponds to `atol = t` and `rtol = 2⋅ϵ`. The value of `ϵ`
        # should not be decreased below `eps(T)/2`, for then rounding errors
        # might prevent convergence. Hence `rtol ≥ eps(T)` is recommended. If
        # `ϵ = eps(T)/2`, that is `rtol = eps(T)`, the error is approximately
        # bounded by:
        #
        #     abs(x - z) ≤ 6⋅ϵ⋅abs(z) + 2⋅t = 3*rtol*abs(z) + 2*atol
        #
        # with `z` the exact solution (see Eq. (2.18) p. 52 in Brent's book).

        # Check for convergence.
        m = (c - b)/2
        abs(m) ≤ tol && return (b, fb, minmax(b, c)..., eval)

        # See if a bisection is forced.
        if abs(e) < tol || abs(fa) ≤ abs(fb)
            # Bounds decreasing too slowly, use bisection.
            d = e = m
        else
            s = fb/fa
            if a == c
                # Linear interpolation.
                p = 2*m*s
                q = one(s) - s # NOTE: one and oneunit are the same because s is unitless
            else
                # Inverse quadratic interpolation.
                q = fa/fc
                r = fb/fc
                p = (2*m*q*(q - r) - (b - a)*(r - one(r)))*s
                q = (q - one(q))*(r - one(r))*(s - one(s))
            end
            if p > zero(p)
                q = -q
            else
                p = -p
            end
            if 2*p < min(3*m*q - tol*abs(q), abs(e*q))
                # Take the interpolation point.
                e = d
                d = p/q
            else
                # Force a bisection.
                d = e = m
            end
        end
        a, fa = b, fb
        if abs(d) > tol
            b += d
        elseif m > zero(m)
            b += tol
        else
            b -= tol
        end
        fb = convert(Tf, f(b))
        eval += 1
        iszero(fb) && return (b, fb, b, b, eval)
        if (fb > zero(fb)) == (fc > zero(fc))
            # Drop point C (make it coincident with point A) and adjust bounds
            # of interval.
            c, fc = a, fa
            e = d = b - a
        end
    end
end

"""
    Brent.fmin([T,] f, a, b, args...; kwds...) -> (xm, fm, lo, hi, nf)

applies Brent's algorithm to find a local minimum of the function `f(x)` in the
interval `[a,b]`.

The result is the 5-tuple `(xm, fm, lo, hi, nf)` with `xm` the estimated value
for which `f` attains a local minimum value in `[a,b]`, `fm = f(xm)` the
function value at `xm`, `lo` and `hi` the bounds for the position of the local
minimum, and `nf` the number of function calls.

Optional argument `T` is the floating-point type to use for computations. If
`T` is unspecified, it is determined by the numeric type of the arguments `a`,
`b`, and `args...`.

To save computations, `args...` can consist in up to 3 points `x`, `w`, and `w`
in the interval `[a,b]` along with their function values:

    fmin([T,] f, a, b, x, f(x), [w, f(w), [v, f(v)]]; kwds...)

The given points need not be distinct nor ordered, they will be taken into
account to initialize the search.

The method used is a combination of golden section search and successive
parabolic interpolation. Convergence is never much slower than that for a
Fibonacci search. If `f` has a continuous second derivative which is positive
at the minimum (which is not at `a` or `b`), then convergence is superlinear,
and usually of the order of about `1.3247`.

Keywords `rtol` and `atol` can be used to specify a tolerance:

    tol = rtol*abs(x) + atol

for the solution. The function `f` is never evaluated at two points closer than
`tol`. The relative tolerance `rtol` should be no smaller than twice the
relative machine precision, and preferably not much less than the square root
of the relative machine precision.

If `f` is a unimodal function and the computed values of `f` are always
unimodal when separated by at least `sqrt(eps(T))*abs(x) + (atol/3)`, then `xm`
returned by `fmin` approximates the abscissa of the global minimum of `f` on
the interval `[a,b]` with an error less than `3*sqrt(eps(T))*abs(fm) + atol`.

If `f` is not unimodal, then `fmin` may approximate a local, but perhaps
non-global, minimum to the same accuracy.

This function is based on Richard Brent's FORTRAN 77 code FMIN which itself is
a slightly modified translation of the Algol 60 procedure LOCALMIN given in:

> Richard Brent, "Algorithms for minimization without derivatives,"
> Prentice-Hall, inc. (1973).

"""
@inline fmin(f, a::Number, b::Number, args::Number...; kwds...) =
    fmin(fmin_float(a, b, args...), f, a, b, args...; kwds...)

"""
    fmax([T,] f, a, b, args...; kwds...) -> (xm, fm, lo, hi, nf)

applies Brent's algorithm to find a local maximum of the function `f(x)` in the
interval `[a,b]`. See [`fmin`](@ref) for details.

"""
@inline fmax(f, a::Number, b::Number, args::Number...; kwds...) =
    fmax(fmin_float(a, b, args...), f, a, b, args...; kwds...)

for (func, lt) in ((:fmin, :(<)), (:fmax, :(>)))

    @eval function $func(::Type{T}, f, a::Number, b::Number; kwds...) where {T<:AbstractFloat}
        # Determine suitable type for the variables in the computations.
        Tx = convert_real_type(T, promote_typeof(a, b))

        # Convert input values.
        a = convert(Tx, a)
        b = convert(Tx, b)

        # Get tolerances.
        atol, rtol = fmin_tolerances(a, b; kwds...)

        # Order end points of search interval.
        if a > b
            a, b = b, a
        end

        # Initialize the search with a point in the interval.
        x = a + goldstep(T)*(b - a)
        fx = convert_real_type(T, f(x))

        # Run Brent's algorithm.
        return $(Symbol(:_,func))(f, a, b, x, fx, x, fx, x, fx, atol, rtol, 1)
    end

    @eval function $func(::Type{T}, f, a::Number, b::Number,
                         x::Number, fx::Number;
                         kwds...) where {T<:AbstractFloat}
        # Determine suitable types for the variables and for the function values in
        # the computations.
        Tx = convert_real_type(T, promote_typeof(a, b, x))
        Tf = convert_real_type(T, promote_typeof(fx))

        # Convert input values.
        a, b = convert(Tx, a), convert(Tx, b)
        x, fx = convert(Tx, x), convert(Tf, fx)

        # Get tolerances.
        atol, rtol = fmin_tolerances(a, b; kwds...)

        # Order end points of search interval and check initial x.
        if a > b
            a, b = b, a
        end
        (a ≤ x ≤ b) || bad_argument("given point `x` is not inside the search interval `[a,b]`")

        # Run Brent's algorithm.
        return $(Symbol(:_,func))(f, a, b, x, fx, x, fx, x, fx, atol, rtol)
    end

    @eval function $func(::Type{T}, f, a::Number, b::Number,
                         x::Number, fx::Number,
                         w::Number, fw::Number; kwds...) where {T<:AbstractFloat}
        # Determine suitable types for the variables and for the function values in
        # the computations.
        Tx = convert_real_type(T, promote_typeof(a, b, x, w))
        Tf = convert_real_type(T, promote_typeof(fx, fw))

        # Convert input values.
        a, b = convert(Tx, a), convert(Tx, b)
        x, fx = convert(Tx, x), convert(Tf, fx)
        w, fw = convert(Tx, w), convert(Tf, fw)

        # Get tolerances.
        atol, rtol = fmin_tolerances(a, b; kwds...)

        # Order end points of search interval and check initial x and w.
        if a > b
            a, b = b, a
        end
        (a ≤ x ≤ b) || bad_argument("given point `x` is not inside the search interval `[a,b]`")
        (a ≤ w ≤ b) || bad_argument("given point `w` is not inside the search interval `[a,b]`")

        # Reorder the points as assumed by Brent's algorithm before running it.
        if $lt(fw, fx)
            x, fx, w, fw = w, fw, x, fx
        end
        return $(Symbol(:_,func))(f, a, b, x, fx, w, fw, w, fw, atol, rtol)
    end

    @eval function $func(::Type{T}, f, a::Number, b::Number,
                         x::Number, fx::Number,
                         w::Number, fw::Number,
                         v::Number, fv::Number;
                         kwds...) where {T<:AbstractFloat}
        # Determine suitable types for the variables and for the function
        # values in the computations.
        Tx = convert_real_type(T, promote_typeof(a, b, x, w, v))
        Tf = convert_real_type(T, promote_typeof(fx, fw, fv))

        # Convert input values.
        a, b = convert(Tx, a), convert(Tx, b)
        x, fx = convert(Tx, x), convert(Tf, fx)
        w, fw = convert(Tx, w), convert(Tf, fw)
        v, fv = convert(Tx, v), convert(Tf, fv)

        # Get tolerances.
        atol, rtol = fmin_tolerances(a, b; kwds...)

        # Order end points of search interval and check initial x, w, and v.
        if a > b
            a, b = b, a
        end
        (a ≤ x ≤ b) || bad_argument("given point `x` is not inside the search interval `[a,b]`")
        (a ≤ w ≤ b) || bad_argument("given point `w` is not inside the search interval `[a,b]`")
        (a ≤ v ≤ b) || bad_argument("given point `v` is not inside the search interval `[a,b]`")

        # Reorder the points as assumed by Brent's algorithm before running it.
        if $lt(fw, fx)
            x, fx, w, fw = w, fw, x, fx
        end
        if $lt(fv, fx)
            x, fx, v, fv = v, fv, x, fx
        end
        if abs(x - v) < abs(x - w)
            v, fv, w, fw = w, fw, v, fv
        end
        return $(Symbol(:_,func))(f, a, b, x, fx, w, fw, v, fv, atol, rtol)
    end

end

"""
    _fmin(f, a, b, x, fx, w, fw, v, fv, atol, rtol, eval=0)

performs the main loop of Brent's algorithm assuming that all parameters are
properly set (as explained below). Tolerances `atol` and `rtol` may be (both)
specified as keywords.

Original Brent's algorithm assumes that the minimum is in the open interval
`(a,b)` with `a < b` and keeps track of the following variables:

- `x`, `fx = f(x)`: position and least function value found so far;
- `w`, `fw = f(w)`: previous values of `x` and `fx`;
- `v`, `fv = f(v)`: previous values of `w` and `fw`;
- `d`: computed step (new try is: `u = x + d`, unless `d` too small);
- `e`: the previous value of `d`, if a parabolic step is taken; the
  difference between the most distant current endpoint and `x`, if a
  golden step is taken.

Other variables need not be saved, notably:

- `u`, `fu = f(u)`: the next point to try and its function value.

Thus the main loop of Brent's algorithm can be entered with any `x`, `w`, `v`
(not necessarily distinct) which are in `[a,b]` and such that:

    fx = f(x)  ≤  fw = f(w)
    |x - w| ≤ |x - v|           (to avoid a tie)

other internal variables are:

    d = x - w
    e = w - v

"""
function _fmin(f, a::Tx, b::Tx,
               x::Tx, fx::Tf,
               w::Tx, fw::Tf,
               v::Tx, fv::Tf,
               atol::Tx, rtol::AbstractFloat,
               eval::Int = 0) where {Tx<:Number,Tf<:Number}
    # Constant for golden step.
    c = goldstep(real_type(Tx))

    # Initialize.
    d = x - w
    e = w - v

    while true

        # Compute mid-point and check the stopping criterion.
        m = (a + b)/2
        tol = rtol*abs(x) + atol
        tol2 = 2*tol
        if abs(x - m) ≤ tol2 - (b - a)/2
            return (x, fx, a, b, eval)
        end

        # Determine next step to take.
        take_golden_step = true
        if abs(e) > tol
            # Fit a parabola (make sure final Q ≥ 0).
            r = (x - w)*(fx - fv)
            q = (x - v)*(fx - fw)
            if q > r
                p = (x - w)*r - (x - v)*q
                q = 2*(q - r)
            else
                p = (x - v)*q - (x - w)*r
                q = 2*(r - q)
            end
            if 2*abs(p) < q*abs(e) && q*(a - x) < p < q*(b - x)
                # Take the parabolic interpolation step.
                take_golden_step = false
                e = d
                d = p/q
                s = x + d
                # F must not be evaluated too close to A or B.
                if s - a < tol2 || b - s < tol2
                    d = (x < m ? tol : -tol)
                end
            end
        end
        if take_golden_step
            # Take a golden-section step.
            e = (x < m ? b : a) - x
            d = c*e
        end

        # F must not be evaluated too close to X.
        if abs(d) ≥ tol
            u = x + d
        elseif d > zero(d)
            u = x + tol
        else
            u = x - tol
        end
        fu = convert(Tf, f(u))
        eval += 1

        # Update A, B, V, W, and X.
        if fu ≤ fx
            if u < x
                b = x
            else
                a = x
            end
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else
            if u < x
                a = u
            else
                b = u
            end
            if fu ≤ fw || w == x
                v, fv = w, fw
                w, fw = u, fu
            elseif fu ≤ fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
end

# Simple structure to change the sign of the result returned by a callable object.
struct ChangeSign{F}
    func::F
end
ChangeSign(f::ChangeSign) = f.func
@inline (obj::ChangeSign)(args...; kwds...) = -obj.func(args...; kwds...)

function _fmax(f, a::Tx, b::Tx,
               x::Tx, fx::Tf,
               w::Tx, fw::Tf,
               v::Tx, fv::Tf,
               atol::Tx, rtol::AbstractFloat,
               eval::Int = 0) where {Tx<:Number,Tf<:Number}
    (xm, fm, lo, hi, nf) = _fmin(ChangeSign(f), a, b, x, -fx, w, -fw, v, -fv, atol, rtol, eval)
    return (xm, -fm, lo, hi, nf)
end

"""
    fminbrkt(f, x, fx, w, fw, v, fv; atol=..., rtol=...)

runs Brent's algorithm to minimize function `f` given 3 points `x`, `w`, and
`v` bracketting the minimum and the corresponding function values `fx = f(x)`,
`fw = f(w)`, and `fv = f(v)` and such that `x ∈ [v,w]` and `fx ≤ min(fv, fw)`
hold.

""" fminbrkt

"""
    fmaxbrkt(f, x, fx, w, fw, v, fv; atol=..., rtol=...)

runs Brent's algorithm to maximize function `f` given 3 points `x`, `w`, and
`v` bracketting the maximum and the corresponding function values `fx = f(x)`,
`fw = f(w)`, and `fv = f(v)` and such that `x ∈ [v,w]` and `fx ≥ max(fv, fw)`
hold.

""" fmaxbrkt

for (min, le) in ((:min, :(≤)), (:max, :(≥)))

    @eval function $(Symbol(:f,min,:brkt))(f,
                                           x::Number, fx::Number,
                                           w::Number, fw::Number,
                                           v::Number, fv::Number; kwds...)
        # Determine types for computations.
        Tx1 = promote_typeof(x, w, v)
        Tf1 = promote_typeof(fx, fw, fv)
        T = float(promote_type(real_type(Tx1), real_type(Tf1)))
        Tx = convert_real_type(T, Tx1)
        Tf = convert_real_type(T, Tf1)

        # Convert values.
        x, fx = convert(Tx, x), convert(Tf, fx)
        w, fw = convert(Tx, w), convert(Tf, fw)
        v, fv = convert(Tx, v), convert(Tf, fv)

        # Determine search interval `[a,b]`, reorder bracket end points, and
        # check that `f(x) ≤ min(f(w),f(w))` for a minimum or that
        # `f(x) ≥ max(f(w),f(w))` for a maximum, and `x ∈ [a,b]`.
        a, b = minmax(w, v)
        (a ≤ x ≤ b && $le(fx, $min(fv, fw))) || bad_argument("illegal bracket")
        if abs(x - v) < abs(x - w)
            v, fv, w, fw = w, fw, v, fv
        end

        # Get tolerances.
        atol, rtol = fmin_tolerances(a, b; kwds...)

        # Call Brent's algorithm.
        return $(Symbol(:_f,min))(f, a, b, x, fx, w, fw, v, fv, atol, rtol)
    end

end

"""
    Brent.fzero_float(a, fa, b, fb) -> T

yields the default floating-point type for computations in `fzero` method. This
also checks the compatibilty of the types of the arguments.

"""
fzero_float(a::Number, fa::Undef, b::Number, fb::Undef) =
    floating_point_type(promote_typeof(a, b))
fzero_float(a::Number, fa::Number, b::Number, fb::Undef) =
    floating_point_type(promote_typeof(a, b), promote_typeof(fa))
fzero_float(a::Number, fa::Number, b::Number, fb::Number) =
    floating_point_type(promote_typeof(a, b), promote_typeof(fa, fb))
fzero_float(a::Number, fa::Undef, b::Number, fb::Number) =
    fzero_float(b, fb, a, fa)

"""
    Brent.fmin_float(a, b, [x, fx, [w, fw, [v, fv]]]) -> T

yields the default floating-point type for computations in `fmin` method. This
also checks the compatibilty of the types of the arguments.

"""
@inline fmin_float(a, b) = floating_point_type(promote_typeof(a, b))
@inline fmin_float(a, b, x, fx) = floating_point_type(
    promote_typeof(a, b, x), promote_typeof(fx))
@inline fmin_float(a, b, x, fx, w, fw) = floating_point_type(
    promote_typeof(a, b, x, w), promote_typeof(fx, fw))
@inline fmin_float(a, b, x, fx, w, fw, v, fv) = floating_point_type(
    promote_typeof(a, b, x, w, v), promote_typeof(fx, fw, fv))

"""
    Brent.fzero_tolerances(a, b; atol=undef, rtol=undef) -> (atol, rtol)

yields the absolute and relative tolerances for the solution sought by Brent's
`fzero` method applied to interval `[a,b]`. Arguments `a` and `b` must have the
same units and the same floating-point type as the solution sought by the
algorithm.

"""
function fzero_tolerances(a::Tx, b::Tx;
                          atol::Union{Number,Undef} = undef,
                          rtol::Union{Number,Undef} = undef) where {Tx<:Number}
    T = real_type(Tx)
    T <: AbstractFloat || bad_argument("bounds `a` and `b` must be floating-point numbers")
    if atol isa Number
        atol ≥ zero(atol) || bad_argument("absolute tolerance `atol = $atol` must be non-negative")
        atol = convert(Tx, atol)::Tx
    else
        atol = eps(T)*abs(b - a)
    end
    if rtol isa Number
        zero(rtol) < rtol < oneunit(rtol) ||
            bad_argument("relative tolerance `rtol = $rtol` must be in (0,1)")
        rtol = convert(T, rtol)::T
    else
        rtol = eps(T)
    end
    return atol, rtol
end

"""
    Brent.fmin_tolerances(a, b; atol=undef, rtol=undef) -> (atol, rtol)

yields the absolute and relative tolerances for the solution sought by Brent's
`fmin` method applied to interval `[a,b]`. Arguments `a` and `b` must have the
same units and the same floating-point type as the solution sought by the
algorithm.

"""
function fmin_tolerances(a::Tx, b::Tx;
                         atol::Union{Number,Undef} = undef,
                         rtol::Union{Number,Undef} = undef) where {Tx<:Number}
    T = real_type(Tx)
    T <: AbstractFloat ||
        bad_argument("bounds `a` and `b` must be floating-point numbers")
    if atol isa Number
        atol ≥ zero(atol) ||
            bad_argument("absolute tolerance `atol = $atol` must be non-negative")
        atol = convert(Tx, atol)::Tx
    else
        atol = eps(T)*abs(b - a)
    end
    if rtol isa Number
        rtol ≥ zero(rtol) ||
            bad_argument("relative tolerance `rtol = $rtol` must be non-negative")
        rtol < oneunit(rtol) ||
            bad_argument("relative tolerance `rtol = $rtol` must be less than $(oneunit(rtol))")
        (iszero(atol) & iszero(rtol)) &&
            bad_argument("absolute and relative tolerances must not be both zero")
        rtol = convert(T, rtol)::T

    else
        rtol = sqrt(eps(T))
    end
    return atol, rtol
end

"""
    Brent.promote_typeof(args...) -> T

yields the promoted type of the types of arguments `args...`.

"""
@inline promote_typeof(args...) = promote_type(map(typeof, args)...)

bad_argument(msg::ArgumentError.types[1]) = throw(ArgumentError(msg))
@noinline bad_argument(args...) = bad_argument(string(args...))

end # module
