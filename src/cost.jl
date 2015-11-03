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

cost{T<:AbstractCost}(obj::T, x) = cost(1, obj, x)
call{T<:AbstractCost}(obj::T, x) = cost(obj, x)
call{T<:AbstractCost}(obj::T, alpha::Real, x) = cost(alpha, obj, x)

cost!{T<:AbstractCost,S}(obj::T, x::S, g::S, clr::Bool) = cost!(1, obj, x, g, clr)
call{T<:AbstractCost,S}(obj::T, x::S, g::S, clr::Bool) = cost!(obj, x, g, clr)
call{T<:AbstractCost,S}(obj::T, alpha::Real, x::S, g::S, clr::Bool) = cost!(alpha, obj, x, g, clr)



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

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
