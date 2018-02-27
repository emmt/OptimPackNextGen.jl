#
# step.jl --
#
# Implement the STEP (Select The Easiest Point) method for global univariate
# optimization [1].
#
# [1] S. Swarzberg, G. Seront & H. Bersini, "S.T.E.P.: the easiest way to
#     optimize a function," in IEEE World Congress on Computational
#     Intelligence, Proceedings of the First IEEE Conference on Evolutionary
#     Computation,vol. 1, pp. 519-524 (1994).
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2017, Éric Thiébaut.
#

# FIXME: fix estimation of precision (compared to Yorick version)
# FIXME: make sure no other minima exist when precision test is satisfied

module Step

# Use the same floating point type for scalars as in OptimPack.
import OptimPackNextGen.Float

using Compat

"""
# Cyclic singly linked list
"""
@compat mutable struct Item{T}
    next::Item{T}
    data::T
    (::Type{Item{T}}){T}(next::Item{T}, data::T) = new{T}(next, data)
    function (::Type{Item{T}}){T}(data::T)
        newitem = new{T}()
        newitem.next = newitem
        newitem.data = data
        return newitem
    end
end
Item{T}(data::T) = Item{T}(data)

"""
Append a new item after a given item of a cyclic singly linked list.
"""
function append!{T}(item::Item{T}, data::T)
    newitem = Item{T}(item.next, data)
    item.next = newitem
    return newitem
end

@compat mutable struct NodeData
    x::Float # position
    y::Float # function value
    q::Float # "quality" factor
end

# Default absolute and relative tolerances:
const TOL = (realmin(Float), sqrt(eps(Float)))

@inline sqrtdifmin(lvl,val) = sqrt(val - lvl)
@inline sqrtdifmax(lvl,val) = sqrt(lvl - val)

for (func, cmp, incr, wgt) in ((:minimize, <, -, :sqrtdifmin),
                               (:maximize, >, +, :sqrtdifmax))
    @eval begin

        function $func(f::Function, a::Float, b::Float;
                       maxeval::Int=100000,
                       tol::NTuple{2,Float}=TOL,
                       alpha::Float=0.0, beta::Float=0.0,
                       verb::Bool=false,
                       printer::Function=default_printer, output::IO=STDOUT)

            maxeval ≥ 2 || error("parameter `maxeval` must be at least 2")
            tol[1] ≥ 0 || error("absolute tolerance `tol[1]` must be nonnegative")
            0 ≤ tol[2] ≤ 1 || error("relative tolerance `tol[2]` must be in [0,1]")
            alpha ≥ 0 || error("parameter `alpha` must be nonnegative")
            beta ≥ 0 || error("parameter `beta` must be nonnegative")

            if a > b
                (a, b) = (b, a)
            elseif a == b
                return (a, f(a), zero(Float), 1)
            end

            fa = f(a)
            fb = f(b)
            list = Item(NodeData(a, fa, 0))
            append!(list, NodeData(b, fb, 0))
            evaluations::Int = 2

            if $cmp(fa, fb)
                xbest = a
                fbest = fa
            else
                xbest = b
                fbest = fb
            end
            xtol::Float = (b - a)/2
            rehash::Bool = true
            verb && printer(output, evaluations, xbest, fbest, xtol)
            c = xbest
            while true
                if xtol ≤ hypot(tol[1], xbest*tol[2])
                    break
                end
                if evaluations ≥ maxeval
                    warn("too many evaluations")
                    break
                end

                # Find where to split.
                split = list
                if rehash
                    qmin = Float(Inf)
                    t = hypot(alpha*fbest, beta)
                    level = $incr(fbest, t)
                    n2 = list
                    x2 = n2.data.x
                    w2 = $wgt(level, n2.data.y)
                    while true
                        n1 = n2
                        n2 = n2.next
                        x1 = x2
                        x2 = n2.data.x
                        x2 > x1 || break
                        w1 = w2
                        w2 = $wgt(level, n2.data.y)
                        q = (w1 + w2)/(x2 - x1)
                        n1.data.q = q
                        if q < qmin
                            qmin = q
                            split = n1
                        end
                    end
                    rehash = false
                else
                    qmin = split.data.q
                    n1 = list
                    x1 = n1.data.x
                    while true
                        n2 = n1.next
                        x2 = n2.data.x
                        x2 > x1 || break
                        q = n1.data.q
                        if q < qmin
                            qmin = q
                            split = n1
                        end
                        n1, x1 = n2, x2
                    end
                end

                # Split the chosen interval.
                evaluations += 1
                x0 = split.data.x
                x1 = split.next.data.x
                c = (x0 + x1)/2
                fc = f(c)
                if $cmp(fc, fbest)
                    xbest = c
                    fbest = fc
                    xtol = (x1 - x0)/2
                    rehash = true
                    verb && printer(output, evaluations, xbest, fbest, xtol)
                end
                if rehash
                    # All Q factors have to be recomputed.
                    q = zero(Float)
                else
                    # Compute the Q factors for the split interval and for the
                    # new one.
                    w1 = $wgt(level, split.data.y)
                    w2 = $wgt(level, fc)
                    w3 = $wgt(level, split.next.data.y)
                    e = (x1 - x0)/2
                    split.data.q = (w1 + w2)/e
                    q = (w2 + w3)/e
                end
                append!(split, NodeData(c, fc, q))
            end

            # Return best point found so far.
            (xbest, fbest, xtol, evaluations)
        end

        function $func(f::Function, a::Real, b::Real;
                       maxeval::Integer=10000,
                       tol::NTuple{2,Real}=TOL,
                       alpha::Real=0.0, beta::Real=0.0,
                       kwds...)
            $func(f, Float(a), Float(b);
                  maxeval = Int(maxeval),
                  tol = (Float(tol[1]), Float(tol[2])),
                  alpha = Float(alpha),
                  beta = Float(beta),
                  kwds...)
        end

    end
end

@doc """
# Find a global minimum or maximum

    Step.minimize(f, a, b) -> (xbest, fbest, xtol, n)
    Step.maximize(f, a, b) -> (xbest, fbest, xtol, n)

finds a global minimum (resp. maximum) of `f(x)` in the interval `[a,b]` and
returns its position `xbest`, the corresponding function value `fbest =
f(xbest)`, the uncertainty `xtol` and the number of function evaluations needed
to find it.

The algorithm is based on the STEP method described in:

> Swarzberg, S., Seront, G. & Bersini, H., "S.T.E.P.: the easiest way to
> optimize a function," in IEEE World Congress on Computational Intelligence,
> Proceedings of the First IEEE Conference on Evolutionary Computation, vol. 1,
> pp. 519-524 (1994).

The following optional keywords can be used:

* `printer` can be set with a user defined function to print iteration
  information, its signature is:

      printer(io::IO, iter::Integer, eval::Integer, rejects::Integer,
              f::Real, gnorm::Real, stp::Real)

  where `io` is the output stream, `iter` the iteration number (`iter = 0` for
  the starting point), `eval` is the number of calls to `fg!`, `rejects` is the
  number of times the computed direction was rejected, `f` and `gnorm` are the
  value of the function and norm of the gradient at the current point, `stp` is
  the length of the step to the current point.

* `output` specifies the output stream for printing information (`STDOUT` is
  used by default).

""" minimize

@doc @doc(minimize) maximize

function default_printer(eval::Int, xm::Float, fm::Float, prec::Float)
    default_printer(STDOUT, eval, xm, fm, prec)
end

function default_printer(io::IO, eval::Int, xm::Float, fm::Float, prec::Float)
    if eval < 3
        @printf(io, "# %s%s\n# %s%s\n",
                "EVALS              X                      F(X)        ",
                "       PREC",
                "------------------------------------------------------",
                "-------------")
    end
    @printf(io, "%7d  %23.15e  %23.15e  %10.2e\n", eval, xm, fm, prec)
end

# Simple parabola.  To be minimized over [-1,2].
testParabola(x) = x*x

# Brent's 5th function.  To be minimized over [-10,10].
testBrent5(x) = (x - sin(x))*exp(-x*x)

# Michalewicz's 1st function.  To be minimized over [-1,2].
testMichalewicz1(x) = x*sin(10.0*x)

# Michalewicz's 2nd function.  To be maximized over [0,pi].
function testMichalewicz2(x)
    s = 0.0
    a = sin(x)
    b = x*x/pi
    for i in 1:10
        s += a*(sin(b*i)^20)
    end
    return s
end

function runtests()
    println("\n# Simple parabola:")
    (xbest, fbest, xtol, n) = minimize(testParabola, -1, 2, verb=true,
                                       maxeval=100, alpha=0, beta=0)
    println("x = $xbest ± $xtol, f(x) = $fbest, n = $n")

    println("\n# Brent's 5th function:")
    (xbest, fbest, xtol, n) = minimize(testBrent5, -10, 10, verb=true,
                                       maxeval=1000)
    println("x = $xbest ± $xtol, f(x) = $fbest, n = $n")

    println("\n# Michalewicz's 1st function:")
    (xbest, fbest, xtol, n) = minimize(testMichalewicz1, -1, 2, verb=true,
                                       maxeval=1000)
    println("x = $xbest ± $xtol, f(x) = $fbest, n = $n")

    println("\n# Michalewicz's 2nd function:")
    (xbest, fbest, xtol, n) = maximize(testMichalewicz2, 0, pi, verb=true,
                                       maxeval=1000, tol=(1e-12,0))
    println("x = $xbest ± $xtol, f(x) = $fbest, n = $n")
end


end # module
