#
# step.jl --
#
# Implement the STEP (Select The Easiest Point) method for global univariate
# optimization [1].
#
# [1] S. Swarzberg, G. Seront & H. Bersini, "S.T.E.P.: the easiest way to
#     optimize a function," in IEEE World Congress on Computational
#     Intelligence, Proceedings of the First IEEE Conference on Evolutionary
#     Computation, vol. 1, pp. 519-524 (1994).
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2022, Éric Thiébaut.
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

# FIXME: use priority queues
# FIXME: fix estimation of precision (compared to Yorick version)
# FIXME: make sure no other minima exist when precision test is satisfied

module Step

# Use the same floating point type for scalars as in OptimPackNextGen.
import OptimPackNextGen.Float

using Printf

"""
    Node(x, fx, q[, next]) -> node

yields a node for the S.T.E.P. method at coordinate `x`, function value `fx =
f(x)`, and quality factor `q`. The nodes form a cyclic chained list, the next
node of the new node may optionaly be specified.

"""
mutable struct Node
    x::Float  # position
    fx::Float # function value
    q::Float  # "quality" factor of the interval
    next::Node
    Node(x::Number, fx::Number, q::Number, next::Node) = new(x, fx, q, next)
    function Node(x::Number, fx::Number, q::Number)
        node = new(x, fx, q)
        node.next = node
        return node
    end
end

"""
    append!(node, x, fx, q) -> next

appends a new node with parameters `x`, `fx = f(x)`, and `q` to `node` and
return it.

"""
function Base.append!(node::Node, x::Number, fx::Number, q::Number)
    next = Node(x, fx, q, node.next)
    node.next = next
    return next
end

# Default absolute and relative tolerances:
const TOL = (floatmin(Float), sqrt(eps(Float)))

@inline sqrtdifmin(lvl,val) = sqrt(val - lvl)
@inline sqrtdifmax(lvl,val) = sqrt(lvl - val)

for (func, cmp, incr, wgt) in ((:minimize, <, -, :sqrtdifmin),
                               (:maximize, >, +, :sqrtdifmax))
    @eval begin

        function $func(f::Function, a::Float, b::Float;
                       maxeval::Int = 100000,
                       tol::NTuple{2,Float} = TOL,
                       alpha::Float=0.0, beta::Float=0.0,
                       verb::Bool = false,
                       printer = default_printer, output::IO = stdout)

            maxeval ≥ 2 || error("parameter `maxeval` must be at least 2")
            tol[1] ≥ 0 || error("absolute tolerance `tol[1]` must be nonnegative")
            0 ≤ tol[2] ≤ 1 || error("relative tolerance `tol[2]` must be in [0,1]")
            alpha ≥ 0 || error("parameter `alpha` must be nonnegative")
            beta ≥ 0 || error("parameter `beta` must be nonnegative")

            # The code requires that a ≤ b.
            if a > b
                (a, b) = (b, a)
            end

            # Initial number of evaluations.
            eval = 0

            # Initial number of iterations.
            iter = 0

            # Evaluate function at ends of initial interval and determine which
            # is the best.
            fa = float(f(a))
            eval += 1
            if b == a
                fb = fa
                xbest = a
                fbest = fa
            else
                fb = float(f(b))
                eval += 1
                if $cmp(fa, fb)
                    xbest = a
                    fbest = fa
                else
                    xbest = b
                    fbest = fb
                end
            end
            prec = b - a
            ftrial = $incr(fbest, hypot(alpha*fbest, beta))
            verb && printer(output, iter, eval, xbest, fbest, prec)
            if prec ≤ hypot(tol[1], xbest*tol[2])
                status = :sufficient_precision
            elseif eval ≥ maxeval
                status = :too_many_evaluations
            else
                status = :continue
            end

            # Create initial chained list of nodes and manage to split initial
            # interval.
            list = Node(a, fa, NaN)
            append!(list, b, fb, NaN)
            split = list

            # Iteration number of the last full update of priorities.
            last_rehash = -1

            # Iterate until convergence or exceeding number of evaluations.
            while status === :continue
                # Split the chosen interval.
                c = (a + b)/2
                fc = float(f(c))
                eval += 1
                if $cmp(fc, fbest)
                    # Solution has improved. Memorize it and check for
                    # convergence.
                    iter += 1
                    xbest = c
                    fbest = fc
                    prec = (b - a)/2
                    ftrial = $incr(fbest, hypot(alpha*fbest, beta))
                    verb && printer(output, iter, eval, xbest, fbest, prec)
                    if prec ≤ hypot(tol[1], xbest*tol[2])
                        status = :sufficient_precision
                        break
                    end
                end
                if eval ≥ maxeval
                    status = :too_many_evaluations
                    break
                end

                # Insert node c in chained list, update priorities, and find
                # next sub-interval to split.
                append!(split, c, fc, NaN)
                qmin = typemax(list.q)
                if last_rehash < iter
                    # Do not rehash until next iteration.
                    last_rehash = iter

                    # Recompute all priorities.
                    this = list
                    w = $wgt(ftrial, this.fx)
                    while true
                        next = this.next
                        e = next.x - this.x
                        e > zero(e) || break
                        w′, w = w, $wgt(ftrial, next.fx)
                        q = (w + w′)/e
                        if q < qmin
                            qmin = q
                            split = this
                        end
                        this.q = q
                        this = next
                    end
                else
                    # Compute the priorities of the two new sub-intervals.
                    wa = $wgt(ftrial, fa)
                    wb = $wgt(ftrial, fb)
                    wc = $wgt(ftrial, fc)
                    e = (b - a)/2
                    split.q      = (wa + wc)/e
                    split.next.q = (wc + wb)/e

                    # Find next interval to split.
                    this = list
                    while true
                        next = this.next
                        this.x < next.x || break
                        if this.q < qmin
                            qmin = this.q
                            split = this
                        end
                        this = next
                    end
                end

                # Get ends of the chosen interval and corresponding function
                # values.
                a = split.x
                b = split.next.x
                fa = split.fx
                fb = split.next.fx
            end

            # Return best point found so far.
            (xbest, fbest, prec, eval, status)
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

"""
# Find a global minimum or maximum

    Step.minimize(f, a, b) -> (xbest, fbest, prec, n, st)
    Step.maximize(f, a, b) -> (xbest, fbest, prec, n, st)

finds a global minimum (resp. maximum) of `f(x)` in the interval `[a,b]` and
returns its position `xbest`, the corresponding function value `fbest =
f(xbest)`, the uncertainty `prec`, the number `n` of function evaluations
needed to find it, and a symbolic status `st`. The status `st` is
`:sufficient_precision` if a solution satisfying the convergence criterion has
been found; otherwise, `st` is `:too_many_evaluations` to indicate that the
required precision was not achieved after the maximum number of allowed
function calls

The algorithm is based on the STEP method described in:

> Swarzberg, S., Seront, G. & Bersini, H., "S.T.E.P.: the easiest way to
> optimize a function," in IEEE World Congress on Computational Intelligence,
> Proceedings of the First IEEE Conference on Evolutionary Computation, vol. 1,
> pp. 519-524 (1994).

The following optional keywords can be used:

* `printer` can be set with a user defined function to print iteration
  information, its signature is:

      printer(io::IO, iter::Int, eval::Int, xbest, fbest, prec)

  where `io` is the output stream, `iter` the iteration number (`iter = 0` for
  the starting point), `eval` is the number of calls to `f`, `xbest` is the
  best solution found so far, `fbest = f(xbest)` is the corresponding function
  value, and `prec` is an upper bound on the absolute precision for the
  solution.

* `output` specifies the output stream for printing information (`stdout` is
  used by default).

""" minimize

@doc @doc(minimize) maximize

function default_printer(iter::Int, eval::Int, xm::Float, fm::Float, prec::Float)
    default_printer(stdout, eval, xm, fm, prec)
end

function default_printer(io::IO, iter::Int, eval::Int,
                         xm::Float, fm::Float, prec::Float)
    if eval < 3
        @printf(io, "# %s%s\n# %s%s\n",
                "ITERS    EVALS              X         ",
                "             F(X)               PREC",
                "--------------------------------------",
                "--------------------------------------")
    end
    @printf(io, "%7d  %7d  %23.15e  %23.15e  %10.2e\n", iter, eval, xm, fm, prec)
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
    (xbest, fbest, prec, n, st) = minimize(testParabola, -1, 2, verb=true,
                                           maxeval=100, alpha=0, beta=0)
    println("x = $xbest ± $prec, f(x) = $fbest, ncalls = $n, status = $st")

    println("\n# Brent's 5th function:")
    (xbest, fbest, prec, n, st) = minimize(testBrent5, -10, 10, verb=true,
                                           maxeval=1000)
    println("x = $xbest ± $prec, f(x) = $fbest, ncalls = $n, status = $st")

    println("\n# Michalewicz's 1st function:")
    (xbest, fbest, prec, n, st) = minimize(testMichalewicz1, -1, 2, verb=true,
                                           maxeval=1000)
    println("x = $xbest ± $prec, f(x) = $fbest, ncalls = $n, status = $st")

    println("\n# Michalewicz's 2nd function:")
    (xbest, fbest, prec, n, st) = maximize(testMichalewicz2, 0, pi, verb=true,
                                           maxeval=1000, tol=(1e-12,0))
    println("x = $xbest ± $prec, f(x) = $fbest, ncalls = $n, status = $st")
end

end # module
