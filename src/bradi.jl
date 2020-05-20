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

finds the global minimum (resp. maximum) of an univariate function `f`.  The
argument `x` is an abstract vector of coordinates in monotonic order; `x[1]`
and `x[end]` are the endpoints of the global search interval and it is assumed
that no more than a single local minimum lies in any subinterval
`[x[i],x[i+2]]`.  The result is a tuple of 2-values: `xbest` the position of
the global minimum and `fbest` the function value at this position.

Optional argument `T` is the floating-point type used for the computations.

If specified, keywords `atol` and `rtol` set the absolute and relative
tolerances for the precision (see `Brent.fmin`).

For instance:

    BraDi.minimize(f, range(a,stop=b,length=n))

performs a global search in the closed interval `[a,b]` which is sampled by `n`
equalspaced points.

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
                  atol::Real = fmin_atol(T),
                  rtol::Real = fmin_rtol(T)) where {T<:AbstractFloat}
    _minimize(T, f, x, T(atol), T(rtol))
end

function maximize(::Type{T}, f, x::AbstractVector{<:Real};
                  kdws...) where {T<:AbstractFloat}
    (xbest, fbest) = minimize(T, t -> -f(t), x; kdws...)
    return (xbest, -fbest)
end

function _minimize(::Type{T}, f, x::AbstractVector{<:Real},
                   atol::T, rtol::T) where {T<:AbstractFloat}
    xbest = xa = xb = xc = T(x[1])
    fbest = fa = fb = fc = T(f(xc))
    n = length(x)
    for j in 2 : n + 1
        xa, xb = xb, xc
        fa, fb = fb, fc
        if j ≤ n
            xc = T(x[j])
            fc = T(f(xc))
            if fc < fbest
                xbest = xc
                fbest = fc
            end
        end
        if fa ≥ fb ≤ fc
            # A minimum has been bracketed in [XA,XC].
            xm, fm = fminbrkt(f, xb, fb, xa, fa, xc, fc, atol, rtol)
            if fm < fbest
                xbest = xm
                fbest = fm
            end
        end
    end
    return (xbest, fbest)
end

@doc @doc(minimize) maximize

end # module
