#
# bradi.jl --
#
# Find a global minimum of an univariate function by the BRADI method
# ("Bracket" then "Dig") described in:
#
# [1] Ferréol Soulez, Éric Thiébaut, Michel Tallon, Isabelle Tallon-Bosc
#     and Paulo Garcia, "Optimal a posteriori fringe tracking in optical
#     interferometry", Proc. SPIE 9146, Optical and Infrared Interferometry
#     IV, 91462Y (July 24, 2014); doi:10.1117/12.2056590
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2014-2018, Éric Thiébaut.
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module Bradi

import OptimPackNextGen.Brent

"""

    Bradi.minimize(f, x) -> (xbest, fbest)
    Bradi.maximize(f, x) -> (xbest, fbest)

finds the global minimum (resp. maximum) of an univariate function `f`.  The
argument `x` is an abstract vector of coordinates in monotonic order; `x[1]`
and `x[end]` are the endpoints of the global search interval and it is assumed
that no more than a single local minimum lies in any subinterval
`[x[i],x[i+2]]`.  The result is a tuple of 2-values: `xbest` the position of
the global minimum and `fbest` the function value at this position.

If specified, keywords `atol` and `rtol` set the absolute and relative
tolerances for the precision (see `Brent.fmin`).

For instance:

    Bradi.minimize(f, linspace(a,b,n))

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
function minimize(f, x::AbstractVector{T};
                  atol::Real = Brent.fmin_atol(T),
                  rtol::Real = Brent.fmin_rtol(T)) where {T<:AbstractFloat}
    minimize(f, x, T(atol), T(rtol))
end

function maximize(f, x::AbstractVector{T};
                  atol::Real = Brent.fmin_atol(T),
                  rtol::Real = Brent.fmin_rtol(T)) where {T<:AbstractFloat}
    (xb, fb) = minimize((x) -> -f(x), x, T(atol), T(rtol))
    return (xb, -fb)
end

function minimize(f, x::AbstractVector{T},
                  atol::T, rtol::T) where {T<:AbstractFloat}

    local xbest::T, xa::T, xb::T, xc::T
    local fbest::T, fa::T, fb::T, fc::T

    xbest = xa = xb = xc = x[1]
    fbest = fa = fb = fc = f(xc)
    n = length(x)
    for j in 2 : n + 1
        xa, xb = xb, xc
        fa, fb = fb, fc
        if j ≤ n
            xc = x[j]
            fc = f(xc)
            if fc < fbest
                xbest = xc
                fbest = fc
            end
        end
        if fa ≥ fb ≤ fc
            # A minimum has been bracketed in [XA,XC].
            xm, fm = Brent.fminbrkt(f, xb, fb, xa, fa, xc, fc, atol, rtol)
            if fm < fbest
                xbest = xm
                fbest = fm
            end
        end
    end
    return (xbest, fbest)
end

@doc @doc(minimize) maximize

end  # module
