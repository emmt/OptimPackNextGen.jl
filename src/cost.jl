#
# cost.jl --
#
# Cost function interface for TiPi.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

import Base.call

abstract AbstractCost

cost(obj::AbstractCost, x) = cost(1, obj, x)
call(obj::AbstractCost, x) = cost(obj, x)
call(obj::AbstractCost, alpha::Real, x) = cost(alpha, obj, x)

cost!{T}(obj::AbstractCost, x::T, g::T, clr::Bool) = cost!(1, obj, x, g, clr)
call{T}(obj::AbstractCost, x::T, g::T, clr::Bool) = cost!(obj, x, g, clr)
call{T}(obj::AbstractCost, alpha::Real, x::T, g::T, clr::Bool) = cost!(alpha, obj, x, g, clr)

function prox!{T}(alpha::Real, param::AbstractCost, x::T, xp::T)
   error("proximal operator not implemented for $(typeof(param))")
end

function prox(alpha::Real, param::AbstractCost, x)
   xp = similar(x)
   prox!(alpha, param, x, xp)
   return xp
end

prox!{T}(obj::AbstractCost, x::T, xp::T) = prox!(1, obj, x, xp)
prox{T}(obj::AbstractCost, x::T) = prox(1, obj, x)

##############################
# Maximum a posteriori (MAP) #
##############################

type MAPCost{L<:AbstractCost,R<:AbstractCost} <: AbstractCost
    mu::Cdouble  # regularization weight
    lkl::L       # parameters of the likelihood term
    rgl::R       # parameters of the regularization term
end

function cost{T}(alpha::Real, param::MAPCost, x::T)
    alpha == 0 && return 0.0
    return (cost(alpha,          param.lkl, x) +
            cost(alpha*param.mu, param.rgl, x))
end

function cost!{T}(alpha::Real, param::MAPCost, x::T, gx::T,
                  clr::Bool=false)
    if alpha == 0
        clr && fill!(gx, 0)
        return 0.0
    else
        return (cost(alpha,          param.lkl, x, gx, clr) +
                cost(alpha*param.mu, param.rgl, x, gx, false))
    end
end
