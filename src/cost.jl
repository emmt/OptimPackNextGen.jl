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

call(f::AbstractCost, x) = cost(Float(1), f, x)
cost(f::AbstractCost, x) = cost(Float(1), f, x)
cost(alpha::Real, f::AbstractCost, x) = cost(Float(alpha), f, x) :: Float
#function cost(alpha::Float, f::AbstractCost, x)
#    error("cost() method not implemented")
#end

function cost!{T}(f::AbstractCost, x::T, g::T, ovr::Bool)
    cost!(1, f, x, g, ovr)
end
function cost!{T}(alpha::Real, f::AbstractCost, x::T, g::T, ovr::Bool)
    cost!(Float(alpha), f, x, g, ovr) ::Float
end

function prox!{T}(alpha::Float, f::AbstractCost, x::T, xp::T)
   error("proximal operator not implemented for $(typeof(f))")
end

function prox(alpha::Real, f::AbstractCost, x)
   xp = vcreate(x)
   prox!(alpha, f, x, xp)
   return xp
end

prox!{T}(f::AbstractCost, x::T, xp::T) = prox!(1, f, x, xp)
prox{T}(f::AbstractCost, x::T) = prox(1, f, x)

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
