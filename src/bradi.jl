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
using ..Brent: ChangeSign, fminbrkt, bad_argument

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

Specify keyword `periodic = true` if `f(x)` is a periodic function of period
`b - a` with `a = minimum(x)` and `b = maximum(x)`.

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

function maximize(::Type{T}, f, x::AbstractVector{<:Number};
                  kdws...) where {T<:AbstractFloat}
    (xm, fm, nf) = minimize(T, ChangeSign(f), x; kdws...)
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

Specify keyword `periodic = true` if `f(x)` is a periodic function of period
`b - a` with `a = minimum(x)` and `b = maximum(x)`.

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
                  periodic::Bool = false,
                  kwds...) where {T<:AbstractFloat}
    # Check that x is in strict increasing or decreasing order and that enough
    # points are given.
    ismonotonic(x) || bad_argument("values of `x` are not strictly decreasing or increasing")
    length(x) ≥ (periodic ? 3 : 2) || bad_argument("insufficient number of values in `x`")

    # Extract first point and compute corresponding function value to determine
    # types and then call private method with converted arguments.
    Tx = convert_real_type(T, eltype(x))
    x1 = convert(Tx, first(x))
    f1 = convert_real_type(T, f(x1))
    Tf = eltype(f1)
    eval = 1 # f(first(x))
    xbest = x1
    fbest = f1
    i_first, i_last = firstindex(x), lastindex(x)
    if periodic
        # Position of 2nd point and corresponding function value.
        x2 = convert(Tx, x[i_first+1])
        f2 = convert(Tf, f(x2))
        eval += 1
        if f2 < f1
            xbest = x2
            fbest = f2
        end
        xb, xc = x1, x2
        fb, fc = f1, f2
        k = i_last - 1 # f(x[n]) = f(x[1]), so no needs to call f for the last point
    else
        xb = xc = x1
        fb = fc = f1
        k = i_last
    end

    # Bracket and dig along given points.
    for i in i_first+eval:i_last+1
        # Move to next triplet. Points a, b, and c are distinct.
        xa, xb = xb, xc
        fa, fb = fb, fc
        if i ≤ k
            xc = convert(Tx, x[i])
            fc = convert(Tf, f(xc))
            eval += 1
            if fc < fbest
                xbest = xc
                fbest = fc
            end
        elseif periodic
            if i == i_last
                xc = x[i_last]
                fc = f1
            else
                xc = x[i_last] + (x2 - x1)
                fc = f2
            end
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
    if periodic
        # Unwrap position of the solution.
        a, b = minmax(x1, x[i_last])
        if xbest > b
            xbest = a + mod(xbest - b, b - a)
        end
    end
    return (xbest, fbest, eval)
end


ismonotonic(x::AbstractRange) = !iszero(step(x))

function ismonotonic(x::AbstractVector)
    flag = true
    i_first, i_last = firstindex(x), lastindex(x)
    if i_first < i_last
        if x[i_first] < x[i_last]
            @inbounds @simd for i in i_first:i_last-1
                flag &= (x[i] < x[i+1])
            end
        elseif x[i_first] > x[i_last]
            @inbounds @simd for i in i_first:i_last-1
                flag &= (x[i] > x[i+1])
            end
        else
            flag = false
        end
    end
    return flag
end

default_float(x::AbstractVector) = floating_point_type(eltype(x))

end # module
