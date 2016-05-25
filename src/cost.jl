#
# cost.jl --
#
# Cost function interface for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

import Base.call

"""
## Cost functions in TiPi

In many cases, solving an inverse problem amounts to minimizing a cost
function.  Abstract type `AbstractCost` is the parent type of any cost function
instance.  The following methods are available for any cost function instance
`f`:

    f(x)    or    cost(f, x)

yield the cost function `f` for the variables `x`.  A multiplier `alpha` can
be specified:

    cost(alpha, f, x)

which yields `alpha*cost(f, x)`.  To perform the optimization, the gradient of
the cost function with respect to the variables may be required:

    cost!(f, x, g, ovr)

stores the gradient of the cost function `f` at `x` in `g` (which has the same
type as `x`, more speciffically `g` was allocated by `g = vcreate(x)`).
Argument `ovr` is a boolean indicating whether to override `g` with the
gradient of `f(x)` or to increment the values of `g` with the gradient of
`f(x)`.  A multiplier `alpha` can also be specified:

    cost!(alpha, f, x, g, ovr)

yields `alpha*cost(f, x)` and stores the gradient of `alpha*f(x)` in `g` (or
increment the values of `g` with the gradient of `alpha*f(x)`).

Developpers only have to provide the two following methods for any concrete
cost function type:

    cost{C,V}(alpha::TiPi.Float, f::C, x::V)
    cost!{C,V}(alpha::TiPi.Float, f::C, x::V, g::V, ovr::Bool)

where `C` is the type of the cost function and `V` is the type of the
variables.  The multiplier is explicitly specified for efficiency reasons.  In
order to be overloaded, these methods have to be imported first.  Typically, a
specific cost function is implemented by:

    import TiPi: Float, AbstractCost, cost, cost!

    type C <: AbstractCost
        ... # any parameters needed by the cost function
    end

    function cost(alpha::Float, f::C, x::V)
        alpha == zero(Float) && return zero(Float)
        fx = ... # compute f(x)
        return alpha*Float(fx)
    end

    function cost!(alpha::Float, f::C, x::V, g::V, ovr::Bool=false)
        if alpha == zero(Float)
            ovr && vfill!(g, 0)
            return zero(Float)
        end
        fx = ... # compute f(x)
        gx = ... # compute the gradient of f(x)
        if ovr
            vscale!(g, alpha, gx)
        else
            vupdate!(g, alpha, gx)
        end
        return alpha*Float(fx)
    end

It is expected that `cost()` and `cost!()` both return a real of type
`TiPi.Float` (which is currently an alias to `Cdouble`).  Default methods for
types derived from `AbstractCost` are implemented to provide the following
shortcuts:

    cost(f, x) = cost(1, f, x)
    f(x) = cost(1, f, x)

and to convert any `Real` multiplier to `TiPi.Float`.

"""
abstract AbstractCost

call(f::AbstractCost, x) = cost(one(Float), f, x)
cost(f::AbstractCost, x) = cost(one(Float), f, x)
cost(alpha::Real, f::AbstractCost, x) = cost(Float(alpha), f, x) :: Float
#function cost(alpha::Float, f::AbstractCost, x)
#    error("cost() method not implemented")
#end

function cost!{T}(f::AbstractCost, x::T, g::T, ovr::Bool)
    cost!(one(Float), f, x, g, ovr)
end
function cost!{T}(alpha::Real, f::AbstractCost, x::T, g::T, ovr::Bool)
    cost!(Float(alpha), f, x, g, ovr) ::Float
end

function prox!{T}(alpha::Float, f::AbstractCost, x::T, xp::T)
   error("proximal operator not implemented for $(typeof(f))")
end

function prox(alpha::Real, f::AbstractCost, x)
   xp = vcreate(x)
   prox!(Float(alpha), f, x, xp)
   return xp
end

prox!{T}(f::AbstractCost, x::T, xp::T) = prox!(one(Float), f, x, xp)
prox{T}(f::AbstractCost, x::T) = prox(one(Float), f, x)

##############################
# Maximum a posteriori (MAP) #
##############################

type MAPCost{L<:AbstractCost,R<:AbstractCost} <: AbstractCost
    lkl::L     # parameters of the likelihood term
    mu::Float  # regularization weight
    rgl::R     # parameters of the regularization term
    function MAPCost{L,R}(lkl::L, mu::Float, rgl::R)
        @assert mu ≥ 0
        new(lkl, mu, rgl)
    end
end

function MAPCost{L<:AbstractCost,R<:AbstractCost}(lkl::L, mu::Real, rgl::R)
    MAPCost{L,R}(lkl, Float(mu), rgl)
end

function cost{L,R}(alpha::Float, f::MAPCost{L,R}, x)
    alpha == zero(Float) && return zero(Float)
    return (cost(alpha,      f.lkl, x) +
            cost(alpha*f.mu, f.rgl, x))
end

function cost!{L,R,T}(alpha::Float, f::MAPCost{L,R}, x::T, g::T,
                      ovr::Bool=false)
    if alpha == zero(Float)
        ovr && vfill!(g, 0)
        return zero(Float)
    end
    return Float(cost!(alpha,      f.lkl, x, g, ovr) +
                 cost!(alpha*f.mu, f.rgl, x, g, false))
end

#------------------------------------------------------------------------------
# QUADRATIC COST

immutable QuadraticCost{E,F} <: AbstractCost
    A::LinearOperator{E,F}
    b::Union{E,Void}
    W::SelfAdjointOperator{E}
    bias::Bool
    function QuadraticCost{E,F}(A::LinearOperator{E,F}, ::Void,
                                W::SelfAdjointOperator{E})
        new(A, nothing, W, false)
    end
    function QuadraticCost{E,F}(A::LinearOperator{E,F}, b::E,
                                W::SelfAdjointOperator{E})
        new(A, b, W, true)
    end
end

QuadraticCost() = QuadraticCost(Any)

function QuadraticCost{E}(::Type{E})
    I = Identity(E)
    QuadraticCost{E,E}(I, nothing, I)
end

function QuadraticCost{E,F}(A::LinearOperator{E,F}, b::E,
                            W::SelfAdjointOperator{E})
    QuadraticCost{E,F}(A, b, W)
end

function QuadraticCost{E}(::Void, b::E, W::SelfAdjointOperator{E})
    QuadraticCost{E,E}(Identity(E), b, W)
end

function QuadraticCost{E,F}(A::LinearOperator{E,F}, ::Void,
                            W::SelfAdjointOperator{E})
    QuadraticCost{E,F}(A, nothing, W)
end

function QuadraticCost{E,F}(A::LinearOperator{E,F}, b::E, ::Void)
    QuadraticCost{E,F}(A, b, Identity(E))
end

function QuadraticCost{E,F}(A::LinearOperator{E,F}, ::Void, ::Void)
    QuadraticCost{E,F}(A, nothing, Identity(E))
end

function QuadraticCost{E}(::Void, ::Void, W::SelfAdjointOperator{E})
    QuadraticCost{E,E}(Identity(E), nothing, W)
end

function QuadraticCost{E}(::Void, b::E, ::Void)
    QuadraticCost{E,E}(Identity(E), b, Identity(E))
end

"""
     residuals(f, x)

yields `r = A*x - b` for the quadratic cost `f`.
"""
function residuals{E,F}(q::QuadraticCost{E,F}, x::F)
    r = q.A*x
    if q.bias
        if is(r, x)
            r = vcreate(x)
            vcombine!(r, 1, x, -1, q.b)
        else
            vupdate!(r, -1, q.b)
        end
    end
    return r
end

function cost{E,F}(alpha::Float, q::QuadraticCost{E,F}, x::F)
    alpha == zero(Float) && return zero(Float)
    r = residuals(q, x)
    return Float((alpha/2)*vdot(r, q.W*r))
end

function cost!{E,F}(alpha::Float, q::QuadraticCost{E,F}, x::F, g, ovr::Bool)
    if alpha == zero(Float)
        ovr && vfill!(g, 0)
        return zero(Float)
    end
    r = residuals(q, x)
    Wr = q.W*r
    if ovr
        apply_adjoint!(g, q.A, Wr)
        if alpha != one(Float)
            vscale!(g, alpha)
        end
    else
        vupdate!(g, alpha, q.A'*Wr)
    end
    return Float((alpha/2)*vdot(r, Wr))
end

"""
A general formulation of a quadratic cost is:

    f(x) = (1/2)*(A*x - b)'*W*(A*x - b)

with `A` a linear operator, `b` a "vector" and `W` a weighting operator.  The
gradient is given by:

    g(x) = A'*W*(A*x - b)

""" QuadraticCost

#------------------------------------------------------------------------------
# CHECKING OF GRADIENTS

function check_gradient{T}(f::AbstractCost, x::T; keywords...)
    g = vcreate(x)
    f0 = cost!(f, x, g, true)
    check_gradient(x -> cost(f, x), x, g; keywords...)
end

function check_gradient{T}(f::AbstractCost, x::T, g::T; keywords...)
    check_gradient(x -> cost(f, x), x, g; keywords...)
end

function check_gradient{T,N}(f::Function, x::Array{T,N}, g::Array{T,N};
                             number::Integer=10,
                             xtol::NTuple{2,Real}=(1e-8,1e-5),
                             dir::Real=+1, verb::Bool=false)

    @assert minimum(xtol) > 0
    @assert number > 0

    n = length(x)
    number = min(n, number)
    if number < n
        sel = rand(1:n, number)
    else
        sel = 1:n
    end
    xatol = T(xtol[1])
    xrtol = T(xtol[2])
    if verb
        @printf("GRADIENT CHECK WITH: xtol=(%.1e,%1e),  number=%d\n",
                xatol, xrtol, number)
    end
    g1 = Array(T, number)
    if dir != 0
        # Forward or backward differences.
        s = T(sign(dir))
        f0 = f(x)
        for j in 1:number
            i = sel[j]
            xi = x[i]
            h = s*max(xatol, xrtol*abs(xi))
            x[i] = xi + h
            g1[j] = (f(x) - f0)/h
            x[i] = xi
            if verb
                @printf("  %9d: %15.7e / %15.7e\n", i, g[i], g1[j])
            end
        end
    else
        # Centered differences.
        for j in 1:number
            println(j)
            i = sel[j]
            xi = x[i]
            h = max(xatol, xrtol*abs(xi))
            x[i] = xi + h
            f1 = f(x)
            x[i] = xi - h
            f2 = f(x)
            x[i] = xi
            g1[j] = (f1 - f2)/(h + h)
            if verb
                @printf("  %9d: %15.7e / %15.7e\n", i, g[i], g1[j])
            end
        end
    end

    # Compute statistics.
    sumabserr = zero(T)
    sumrelerr = zero(T)
    maxabserr = zero(T)
    maxrelerr = zero(T)
    for j in 1:number
        i = sel[j]
        a = g[i]
        b = g1[j]
        abserr = abs(a - b)
        relerr = (abserr == zero(T) ? zero(T)
                  : (abserr + abserr)/(abs(a) + abs(b)))
        sumabserr += abserr
        sumrelerr += relerr
        maxabserr = max(maxabserr, abserr)
        maxrelerr = max(maxrelerr, relerr)
    end
    avgabserr = T(1/number)*sumabserr
    avgrelerr = T(1/number)*sumrelerr
    if verb
        @printf("ABSOLUTE ERROR: average = %.1e,  maximum = %.1e\n",
                avgabserr, maxabserr)
        @printf("RELATIVE ERROR: average = %.1e,  maximum = %.1e\n",
                avgrelerr, maxrelerr)
    end
    avgabserr, maxabserr, avgrelerr, maxrelerr
end

"""

    check_gradient(f, x, g)

Compare gradient function with gradient estimated by finite differences.  `f`
is the function, `x` are the variables and `g` is the gradient of `f(x)` at
`x`.  The returned value is the 4-value tuple:

    (avgabserr, maxabserr, avgrelerr, maxrelerr)

with the average and maximal absolute and relative errors.

If `f` is a cost function (its type inherits from `AbstractCost`), the
gradient needs not be specified:

    check_gradient(f, x)

The number of gradient values to check may be specified by keyword
`number` (the subset of parameters is randomly chosen).  The default is
to compute the finite difference gradient for all variables.

The absolute and relative finite difference step size can be specified by
keyword `xtol`.  By default, `xtol = (1e-8, 1e-5)`.

Keyword `dir` can be used to specify which kind of finite differences to use to
estimate the gradient: "forward" (if `dir > 0`), "backward" (if `dir < 0`) or
"centered" (if `dir = 0`).  The default is to use centered finite differences
which are more precise (of order `h^3` with `h` the step size) but twice more
expensive to compute.

If keyword `verb` is true, the result is printed.

""" check_gradient

#------------------------------------------------------------------------------
