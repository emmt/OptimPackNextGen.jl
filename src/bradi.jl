#
# bradi.jl --
#
# Find a global minimum of an univariate function with the "Bracket then Dig"
# method described in:
#
#     Ferréol Soulez, Éric Thiébaut, Michel Tallon, Isabelle Tallon-Bosc
#     and Paulo Garcia, "Optimal a posteriori fringe tracking in optical
#     interferometry", Proc. SPIE 9146, Optical and Infrared Interferometry
#     IV, 91462Y (July 24, 2014); doi:10.1117/12.2056590
#
#-------------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2014-2023, Éric Thiébaut.
#

module BraDi

using Unitless
using ..Brent: fminbrkt, bad_argument

"""
    BraDi.maximize([T,] f, x; kwds...) -> (xm, fm, nf)

finds the global maximum of the univariate function `f` over the interval
`[first(x),last(x)]` with `x` an abstract vector of values in monotonic order
such that it can be assumed that no more than a single local maximum lies in
any sub-interval `[x[i],x[i+2]]`. The result is a 3-tuple with `xm` the
position of the global maximum, `fm = f(xm)`, and `nf` the number of function
calls.

Optional argument `T` is the floating-point type used for the computations.
If unspecified, `T` is guessed from the type of the elements of `X`.

Keyword `period` may be used to specify the period of `f(x)` if it is a
periodic function.

Keywords `atol` and `rtol` may be used to specify the absolute and relative
tolerances for the precision of the solution (see [`Brent.fmin`](@ref)).

For example:

    BraDi.maximize(f, range(a, b; length=n))

performs a global search in the closed interval `[a,b]` which is sampled by `n`
equally spaced points.

This implements the BraDi ("Bracket" then "Dig") algorithm described in:

> Ferréol Soulez, Éric Thiébaut, Michel Tallon, Isabelle Tallon-Bosc and Paulo
> Garcia, "Optimal a posteriori fringe tracking in optical interferometry",
> Proc. SPIE 9146, Optical and Infrared Interferometry IV, 91462Y (July 24,
> 2014); doi:10.1117/12.2056590

See also [`BraDi.minimize`](@ref), [`Brent.fmin`](@ref), and
[`Step.minimize`](@ref).

"""
maximize(f, x::AbstractVector{<:Number}; kdws...) =
    maximize(default_float(x), f, x; kdws...)

# Simple structure to negate a callable object.
struct Negate{F}
    func::F
end

@inline (obj::Negate)(args...; kwds...) = -obj.func(args...; kwds...)

function maximize(::Type{T}, f, x::AbstractVector{<:Number};
                  kdws...) where {T<:AbstractFloat}
    (xm, fm, nf) = minimize(T, Negate(f), x; kdws...)
    return (xm, -fm, nf)
end

"""
    BraDi.minimize([T,] f, X; kwds...) -> (xm, fm, nf)

finds the global minimum of the univariate function `f` over the interval
`[first(x),last(x)]` with `x` an abstract vector of values in monotonic order
such that it can be assumed that no more than a single local minimum lies in
any sub-interval `[x[i],x[i+2]]`. The result is a 3-tuple with `xm` the
position of the global minimum, `fm = f(xm)`, and `nf` the number of function
calls.

Optional argument `T` is the floating-point type used for the computations.
If unspecified, `T` is guessed from the type of the elements of `X`.

Keyword `period` may be used to specify the period of `f(x)` if it is a
periodic function.

Keywords `atol` and `rtol` may be used to specify the absolute and relative
tolerances for the precision of the solution (see [`Brent.fmin`](@ref)).

For example:

    BraDi.minimize(f, range(a, b; length=n))

performs a global search in the closed interval `[a,b]` which is sampled by `n`
equally spaced points.

This implements the BraDi ("Bracket" then "Dig") algorithm described in:

> Ferréol Soulez, Éric Thiébaut, Michel Tallon, Isabelle Tallon-Bosc and Paulo
> Garcia, "Optimal a posteriori fringe tracking in optical interferometry",
> Proc. SPIE 9146, Optical and Infrared Interferometry IV, 91462Y (July 24,
> 2014); doi:10.1117/12.2056590

See also [`BraDi.maximize`](@ref), [`Brent.fmin`](@ref), and
[`Step.minimize`](@ref).

"""
minimize(f, x::AbstractVector{<:Number}; kdws...) =
    minimize(default_float(x), f, x; kdws...)

function minimize(::Type{T}, f, x::AbstractVector{<:Number};
                  period::Union{Number,Nothing} = nothing,
                  kwds...) where {T<:AbstractFloat}
    # Check x in increasing or decreasing.
    i_first, i_last = firstindex(x), lastindex(x)
    len = i_last - i_first + 1
    len ≥ 2 || bad_argument("insufficient number of coordinates")
    if len > 2
        flag = true
        if x[i_first] < x[i_last]
            @inbounds @simd for i in i_first:i_last-1
                flag &= (x[i] < x[i+1])
            end
        elseif span < zero(span)
            @inbounds @simd for i in i_first:i_last-1
                flag &= (x[i] > x[i+1])
            end
        end
        flag || bad_argument("values of `x` are not strictly decreasing or increasing")
    end

    # If the function is periodic, check the period and set the sign of the
    # period so that it can be used as and increment to move to next period.
    if period !== nothing
        (isfinite(period) && period != zero(period)) ||
            bad_argument("invalid period")
        span = x[i_last] - x[i_first]
        (abs(period) > abs(span)) ||
            bad_argument("period too small or spanned interval too large")
        period = copysign(period, span)
    end

    # Extract first point and compute corresponding function value to determine
    # types and then call private method with converted arguments.
    Tx = convert_real_type(T, eltype(x))
    x1 = convert(Tx, first(x))
    f1 = convert_real_type(T, f(x1))
    return _minimize(f, x, x1, f1, maybe_convert(Tx, period); kwds...)
end

# Aperiodic version.
function _minimize(f, x::AbstractVector, x1::Tx, f1::Tf, ::Nothing;
                   kwds...) where {Tx,Tf}
    # Initialization.
    i_first, i_last = firstindex(x), lastindex(x)
    eval = 1 # f(first(x))
    xbest = x1
    fbest = f1

    # Bracket and dig along given points.
    xb = xc = x1
    fb = fc = f1
    for i in i_first+1:i_last+1
        # Move to next triplet.
        xa, xb = xb, xc
        fa, fb = fb, fc
        if i ≤ i_last
            xc = convert(Tx, x[i])
            fc = convert(Tf, f(xc))
            eval += 1
            if fc < fbest
                xbest = xc
                fbest = fc
            end
        end
        if fa ≥ fb ≤ fc
            # A minimum is bracketed in `[xa,xc]`.
            xm, fm, lo, hi, nf = fminbrkt(f, xb, fb, xa, fa, xc, fc; kwds...)
            eval += nf
            if fm < fbest
                xbest = xm
                fbest = fm
            end
        end
    end
    return (xbest, fbest, eval)
end

# Periodic version.
function _minimize(f, x::AbstractVector, x1::Tx, f1::Tf, period::Tx;
                   kwds...) where {Tx,Tf}

    # Position of 2nd point and corresponding function value.
    i_first, i_last = firstindex(x), lastindex(x)
    x2 = convert(Tx, x[i_first+1])
    f2 = convert(Tf, f(x2))
    eval = 2

    # Best initial solution.
    if f1 < f2
        xbest = x1
        fbest = f1
    else
        xbest = x2
        fbest = f2
    end

    # Bracket and dig along given points plus 2 extra ones to overlap the next
    # period.
    xb, xc = x1, x2
    fb, fc = f1, f2
    for i in i_first+2:i_last+2
        # Move to next triplet (periodically). Points a, b, and c are distinct.
        xa, xb = xb, xc
        fa, fb = fb, fc
        if i ≤ i_last
            xc = convert(Tx, x[i])
            fc = convert(Tf, f(xc))
            eval += 1
            if fc < fbest
                xbest = xc
                fbest = fc
            end
        elseif i == i_last+1
            xc = x1 + period
            fc = f1
        else
            xc = x2 + period
            fc = f2
        end
        if fa ≥ fb ≤ fc
            # A minimum is bracketed in `[xa,xc]`.
            xm, fm, hi, lo, nf = fminbrkt(f, xb, fb, xa, fa, xc, fc; kwds...)
            eval += nf
            if fm < fbest
                xbest = xm
                fbest = fm
            end
        end
    end
    if abs(xbest - x1) ≥ abs(period)
        # Unwrap position of the solution.
        xbest -= period
    end
    return (xbest, fbest, eval)
end

default_float(x::AbstractVector) = floating_point_type(eltype(x))

maybe_convert(::Type{T}, x::T) where {T} = x
maybe_convert(::Type{T}, x) where {T} = convert(T, x)
maybe_convert(::Type{T}, x::Nothing) where {T} = nothing

end # module
