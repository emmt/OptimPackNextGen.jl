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

module Step

using Printf

"""
    Step.minimize(f, a, b; kwds...) -> (xm, fm, prec, n, st)

attempts to find a global minimum of `f(x)` in the interval `[a,b]` and returns
its position `xm`, `fm = f(xm)`, the uncertainty `prec`, the number `n` of
function evaluations, and a symbolic status `st`.

See [`Step.search`](@ref) for possible values of `st` and for allowed keywords.
See [`Step.maximize`](@ref) for searching for a global maximum.

"""
minimize(f, a, b; kwds...) = search(:min, f, a, b; kwds...)

"""
    Step.maximize(f, a, b; kwds...) -> (xm, fm, prec, n, st)

attempts to find a global maximum of `f(x)` in the interval `[a,b]` and returns
its position `xm`, `fm = f(xm)`, the uncertainty `prec`, the number `n` of
function evaluations, and a symbolic status `st`.

See [`Step.search`](@ref) for possible values of `st` and for allowed keywords.
See [`Step.minimize`](@ref) for searching for a global minimum.

"""
maximize(f, a, b; kwds...) = search(:max, f, a, b; kwds...)

"""
    Step.search(obj::Symbol, f, a, b; kwds...) -> (xm, fm, prec, n, st)

attempts to find a global extremum of the univariate function `f(x)` in the
interval `[a,b]`. Objective `obj` can be `:min` (resp. `:max`) to search for
 a global minimum (resp. maximum).

The result is a 5-tuple with `xm` the position of the extremum, `fm = f(xm)`
the corresponding function value, `prec` an upper bound on the absolute
accuracy of the solution, `n` the number of function evaluations, and a
symbolic termination status `st` (which is `:sufficient_precision` if a
solution satisfying the convergence criterion has been found or
`:too_many_evaluations` if the required precision was not achieved after the
maximum number of allowed function calls).

The algorithm is based on the STEP method described in:

> Swarzberg, S., Seront, G. & Bersini, H., "S.T.E.P.: the easiest way to
> optimize a function," in IEEE World Congress on Computational Intelligence,
> Proceedings of the First IEEE Conference on Evolutionary Computation, vol. 1,
> pp. 519-524 (1994).

The following optional keywords can be specified:

* `atol` and `rtol` are absolute and relative tolerances for the accuracy
  of the solution. The algorithm stops when:

      prec ≤ max(atol, rtol*abs(xm))

  By default, the absolute tolerance `atol` is set to the smallest positive
  normal number representable by the floating-point type of `x`, while the
  relative tolerance `rtol` is set to the square root of the relative precision
  of the floating-point type of `x`.

* `maxeval` specifies the maximum number of function evaluations. By default,
  `maxeval = 10_000`.

* `aboost` and `rboost` are absolute and relative boost parameters for the
  function value that the algorithm is trying to reach and which is defined as:

      fboost = fm ± max(aboost, rboost*abs(fm))

  where `fm` is the best function value so far and `±` is `+` when seeking for
  a maximum and -` when seeking for a minimum. By default, these parameters are
  both set to zero as (hence `fboost = fm`) because these settings are the most
  efficient in practice (despite what is claimed in the paper describing the
  S.T.E.P. method).

* `printer` can be set with a user defined function to print iteration
  information, its signature is:

      printer(io::IO, iter::Int, eval::Int, xm, fm, prec)

  with `io` the output stream, `iter` the iteration number (`iter = 0` for the
  starting point), `eval` the number of calls to `f`, `xm` the best solution
  found so far, `fm = f(xm)` the corresponding function value, and `prec` an
  upper bound on the absolute accuracy of the solution. If the variable `x`
  and/or the function value `f(x)` have units, these are discarded by the
  default printer (`xm` and `prec` however always have the same units).

* `output` specifies the output stream for printing information (`stdout` is
  used by default).

See also [`Step.minimize`](@ref) and [`Step.maximize`](@ref).

"""
function search(obj::Symbol, f, a, b; kwds...)
    # Convert ends of intial interval to a common floating-point type.
    Tx = float(promote_type(typeof(a), typeof(b)))
    a_ = convert(Tx, a)::Tx
    b_ = convert(Tx, b)::Tx

    # The low-level code requires that a ≤ b.
    if a_ > b_
        (a_, b_) = (b_, a_)
    end

    # Compute function value at leftmost ends of initial interval. This is to
    # determine the type of function values and `qnan` a NaN with the same type
    # as the "difficulty" of a sub-interval.
    fa = float(f(a_))
    sgn = one(fa)
    if obj === :max
        # The low-level code seeks for a minimum.
        sgn = -sgn
        fa = -fa
    elseif obj !== :min
        error("objective must be `:min` or `:max`")
    end
    qnan = sqrt(zero(fa))/zero(Tx)

    # Call the algorithm with correctly typed parameters.
    return search_(sgn, f, a_, b_, fa, qnan; kwds...)
end

"""
    Node(x, fx, q[, next]) -> node

yields a node for the S.T.E.P. method at coordinate `x`, function value `fx =
f(x)`, and "difficulty" `q`. The nodes form a cyclic chained list, the next
node of the new node may optionaly be specified.

"""
mutable struct Node{Tx,Tf,Tq}
    x::Tx  # position
    fx::Tf # function value
    q::Tq  # "quality" factor of the interval
    next::Node{Tx,Tf,Tq}
    function Node{Tx,Tf,Tq}(x, fx, q, next::Node{Tx,Tf,Tq}) where {Tx,Tf,Tq}
        return new{Tx,Tf,Tq}(x, fx, q, next)
    end
    function Node{Tx,Tf,Tq}(x, fx, q) where {Tx,Tf,Tq}
        node = new{Tx,Tf,Tq}(x, fx, q)
        node.next = node
        return node
    end
end

"""
    append!(node, x, fx, q) -> next

appends a new node with parameters `x`, `fx = f(x)`, and `q` to `node` and
return it.

"""
function Base.append!(node::Node{Tx,Tf,Tq}, x, fx, q) where {Tx,Tf,Tq}
    next = Node{Tx,Tf,Tq}(x, fx, q, node.next)
    node.next = next
    return next
end

# The following methods yield the default tolerances for variables of type Tx
# and function values of type Tf. Tx and Tf may have units but must be
# floating-point types.
#
# NOTE: See the inline documentation for `isapprox` about the difficulty to
#       define a universal absolute tolerance.
#
# NOTE: We use one(T) to get rid of units and nextfloat(zero(T)) instead of
#       floatmin(T) which does not support unitful quantities.
default_atol(Tx::Type) = nextfloat(zero(Tx))
default_rtol(Tx::Type) = sqrt(eps(one(Tx)))
default_aboost(Tf::Type) = zero(Tf)
default_rboost(Tf::Type) = zero(one(Tf))

for func in (:default_atol, :default_rtol, :default_aboost, :default_rboost)
    @eval $func(arg) = $func(typeof(arg))
end

# First low level helper to check settings and convert keywords types.
function search_(sgn, f, a::Tx, b::Tx, fa::Tf, qnan::Tq;
                 maxeval::Integer = 10_000,
                 atol = default_atol(Tx),
                 rtol = default_rtol(Tx),
                 rboost = default_rboost(Tf),
                 aboost = default_aboost(Tf),
                 kwds...) where {Tx,Tf,Tq}
    # Check keywords.
    maxeval ≥ 2 || error("parameter `maxeval` must be at least 2")
    atol ≥ zero(atol) || error("absolute tolerance `atol` must be nonnegative")
    zero(rtol) ≤ rtol ≤ one(rtol) || error("relative tolerance `rtol` must be in [0,1]")
    rboost ≥ zero(rboost) || error("parameter `rboost` must be nonnegative")
    aboost ≥ zero(aboost) || error("parameter `aboost` must be nonnegative")

    # Call next helper with keywords converted to their correct types.
    return search__(sgn, f, a, b, fa, qnan;
                    maxeval = convert(Int, maxeval),
                    atol = convert(Tx, atol),
                    rtol = convert(typeof(one(Tx)), rtol),
                    aboost = convert(Tf, aboost),
                    rboost = convert(typeof(one(Tf)), rboost),
                    kwds...)
end

function search__(sgn, f, a::Tx, b::Tx, fa::Tf, qnan::Tq;
                  maxeval::Int, atol::Tx, rtol, aboost::Tf, rboost,
                  verb::Bool = false,
                  printer = default_printer,
                  output::IO = stdout) where {Tx,Tf,Tq}
    # Initial number of iterations and of evaluations. Note that f(a) has
    # already been computed.
    iter = 0
    eval = 1

    # Evaluate function at rightmost ends of initial interval.
    if b == a
        fb = fa
    else
        fb = sgn*float(f(b))
        eval += 1
    end

    # Initial best solution.
    if fa < fb
        xbest = a
        fbest = fa
    else
        xbest = b
        fbest = fb
    end
    prec = b - a
    fboost = fbest - max(aboost, rboost*abs(fbest))
    verb && printer(output, iter, eval, xbest, sgn*fbest, prec)
    if prec ≤ max(atol, rtol*abs(xbest))
        status = :sufficient_precision
    elseif eval ≥ maxeval
        status = :too_many_evaluations
    else
        status = :continue
    end

    # If the initial interval is larger than the required precision,
    # iteratively split the "easiest" sub-interval until the precision
    # criterion is satisfied or the number of evaluations exceeds the limit.
    if status === :continue
        # Create initial chained list of nodes and manage to split the initial
        # interval.
        list = Node{Tx,Tf,Tq}(a, fa, qnan)
        append!(list, b, fb, qnan)
        split = list

        # Iteration number of the last full update of the "difficulties" of the
        # sub-intervals.
        last_rehash = -1

        # Iterate until convergence or exceeding number of evaluations.
        while true
            # Split the chosen interval.
            c = (a + b)/2
            fc = sgn*float(f(c))
            eval += 1
            if fc < fbest
                # Solution has improved. Memorize it and check for convergence.
                iter += 1
                xbest = c
                fbest = fc
                prec = (b - a)/2
                fboost = fbest - max(aboost, rboost*abs(fbest))
                verb && printer(output, iter, eval, xbest, sgn*fbest, prec)
                if prec ≤ max(atol, rtol*abs(xbest))
                    status = :sufficient_precision
                    break
                end
            end
            if eval ≥ maxeval
                status = :too_many_evaluations
                break
            end

            # Insert node c in chained list, update "difficulties", and find
            # next sub-interval to split.
            append!(split, c, fc, qnan)
            qmin = typemax(list.q)
            if last_rehash < iter
                # Do not rehash until next iteration.
                last_rehash = iter

                # Recompute all "difficulties".
                this = list
                w = sqrt(this.fx - fboost)
                while true
                    next = this.next
                    e = next.x - this.x
                    e > zero(e) || break
                    w′, w = w, sqrt(next.fx - fboost)
                    q = (w + w′)/e
                    if q < qmin
                        qmin = q
                        split = this
                    end
                    this.q = q
                    this = next
                end
            else
                # Compute the "difficulties" of the two new sub-intervals.
                wa = sqrt(fa - fboost)
                wb = sqrt(fb - fboost)
                wc = sqrt(fc - fboost)
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
    end

    # Return best point found so far.
    (xbest, sgn*fbest, prec, eval, status)
end

function default_printer(io::IO, iter::Int, eval::Int,
                         xm::Tx, fm::Tf, prec::Tx) where {Tx,Tf}
    if eval < 3
        @printf(io, "# %s%s\n# %s%s\n",
                "ITERS    EVALS              X         ",
                "             F(X)               PREC",
                "--------------------------------------",
                "--------------------------------------")
    end
    @printf(io, "%7d  %7d  %23.15e  %23.15e  %10.2e\n", iter, eval,
            xm/oneunit(Tx), fm/oneunit(Tf), prec/oneunit(Tx))
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
                                           maxeval=100, atol=1e-12, rtol=0)
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
                                           maxeval=1000, atol=1e-12, rtol=0)
    println("x = $xbest ± $prec, f(x) = $fbest, ncalls = $n, status = $st")
end

end # module
