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
# Copyright (C) 2014-2020, Éric Thiébaut.
#

module BraDi

using ..Brent: fmin_atol, fmin_rtol, fminbrkt

"""
    BraDi.minimize([T=Float64,] f, x) -> (xbest, fbest)
    BraDi.maximize([T=Float64,] f, x) -> (xbest, fbest)

find the global minimum (resp. maximum) of an univariate function `f`.  The
argument `x` is an abstract vector of coordinates in monotonic order; `x[1]`
and `x[end]` are the endpoints of the global search interval and it is assumed
that no more than a single local minimum lies in any subinterval
`[x[i],x[i+2]]`.  The result is a tuple of 2-values: `xbest` the position of
the global minimum and `fbest` the function value at this position.

Optional argument `T` is the floating-point type used for the computations.

If specified, keywords `atol` and `rtol` set the absolute and relative
tolerances for the precision (see `Brent.fmin`).

For example:

    BraDi.minimize(f, range(a, b; length=n))

performs a global search in the closed interval `[a,b]` which is sampled by `n`
equally spaced points.

These functions implement the BraDi ("Bracket" then "Dig") algorithm described
in:

> Ferréol Soulez, Éric Thiébaut, Michel Tallon, Isabelle Tallon-Bosc and Paulo
> Garcia, "Optimal a posteriori fringe tracking in optical interferometry",
> Proc. SPIE 9146, Optical and Infrared Interferometry IV, 91462Y (July 24,
> 2014); doi:10.1117/12.2056590

See also `Brent.fmin` and `Step.minimize`.

"""
minimize(f, x::AbstractVector{<:Real}; kdws...) =
    minimize(Float64, f, x; kdws...)

maximize(f, x::AbstractVector{<:Real}; kdws...) =
    maximize(Float64, f, x; kdws...)

function minimize(::Type{T}, f, x::AbstractVector{<:Real};
                  period::Union{Real,Nothing} = nothing,
                  atol::Real = fmin_atol(T),
                  rtol::Real = fmin_rtol(T)) where {T<:AbstractFloat}
    _minimize(T, f, x, T(atol), T(rtol), _conv(T, period))
end

_conv(::Type{T}, x::T) where {T} = x
_conv(::Type{T}, x) where {T} = convert(T, x)
_conv(::Type{T}, x::Nothing) where {T} = nothing

function maximize(::Type{T}, f, x::AbstractVector{<:Real};
                  kdws...) where {T<:AbstractFloat}
    (xbest, fbest) = minimize(T, t -> -f(t), x; kdws...)
    return (xbest, -fbest)
end

@doc @doc(minimize) maximize

function _minimize(::Type{T}, f, x::AbstractVector{<:Real},
                   atol::T, rtol::T, ::Nothing) where {T<:AbstractFloat}
    # Check arguments.
    _check_coordinates(x)

    # Coordinate and functionn value of first point.
    i1 = firstindex(x)
    x1 = T(x[i1])
    f1 = T(f(x1))

    # Bracket and dig along given points.
    xbest = xb = xc = x1
    fbest = fb = fc = f1
    n = length(x)
    for k in 1 : n
        # Move to next triplet.
        xa, xb = xb, xc
        fa, fb = fb, fc
        if k < n
            xc = T(x[i1+k])
            fc = T(f(xc))
            if fc < fbest
                xbest = xc
                fbest = fc
            end
        end
        if fa ≥ fb ≤ fc
            # A minimum is bracketed in `[xa,xc]`.
            xm, fm = fminbrkt(f, xb, fb, xa, fa, xc, fc, atol, rtol)
            if fm < fbest
                xbest = xm
                fbest = fm
            end
        end
    end
    return (xbest, fbest)
end

function _minimize(::Type{T}, f, x::AbstractVector{<:Real},
                   atol::T, rtol::T, period::T) where {T<:AbstractFloat}
    # Check arguments.
    span = _check_coordinates(x)
    (isfinite(period) && period != 0) ||
        throw(ArgumentError("invalid period"))
    (abs(period) > abs(span)) ||
        throw(ArgumentError("period too small or spanned interval too large"))

    # Set the sign of the period so that it can be used as and increment to
    # move to next period.
    if span > 0
        period = +abs(period)
    else
        period = -abs(period)
    end

    # Coordinates and function values at the 2 first points.
    i1 = firstindex(x)
    x1 = T(x[i1])
    f1 = T(f(x1))
    x2 = T(x[i1+1])
    f2 = T(f(x2))

    # Bracket and dig along given points plus 2 extra ones to overlap the next
    # period.
    xb, fb = x1, f1
    xc, fc = x2, f2
    if fb < fb
        xbest, fbest = xb, fb
    else
        xbest, fbest = xc, fc
    end
    n = length(x)
    wa, wb = x1, x1 + period # endpoints of wrapping interval
    for k in 2 : n + 1
        # Move to next triplet (periodically).
        xa, xb = xb, xc
        fa, fb = fb, fc
        if k < n
            xc = T(x[i1+k])
            fc = T(f(xc))
            if fc < fbest
                xbest = xc
                fbest = fc
            end
        elseif k == n
            xc = x1 + period
            fc = f1
        else
            xc = x2 + period
            fc = f2
        end
        if fa ≥ fb ≤ fc
            # A minimum is bracketed in `[xa,xc]`.
            xm, fm = fminbrkt(f, xb, fb, xa, fa, xc, fc, atol, rtol)
            if fm < fbest
                # Update best solution so far, taking care of wrapping the coordinate
                # in the correct interval.
                if k < n || (wa < wb ? wa ≤ xm < wb : wb < xm ≤ wa)
                    xbest = xm
                else
                    xbest = xm - period
                end
                fbest = fm
            end
        end
    end
    return (xbest, fbest)
end

function _check_coordinates(x::AbstractVector{<:Real})
    len = length(x)
    len ≥ 2 || throw(ArgumentError("insufficient number of coordinates"))
    if len ≥ 2
        i_first = firstindex(x)
        i_last = lastindex(x)
        span = x[i_last] - x[i_first]
        if span > 0
            flag = true
            if len > 2
                @inbounds @simd for i in i_first:i_last-1
                    flag &= (x[i] < x[i+1])
                end
            end
            if flag
                return span
            end
        elseif span < 0
            flag = true
            if len > 2
                @inbounds @simd for i in i_first:i_last-1
                    flag &= (x[i] > x[i+1])
                end
            end
            if flag
                return span
            end
        end
    end
    throw(ArgumentError("coordinates are not strictly decreasing or increasing"))
end

end # module
