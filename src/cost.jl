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

abstract CostParam

##############################
# Maximum a posteriori (MAP) #
##############################

type MAPCostParam <: CostParam
   mu::Cdouble     # regularization weight
   lkl::CostParam  # parameters of the likelihood term
   rgl::CostParam  # parameters of the regularization term
end

function cost{T}(alpha::Real, param::MAPCostParam, x::T)
    alpha == 0 && return 0.0
    cost(alpha, param.lkl, x) + cost(alpha*param.mu, param.rgl, x)
end

function cost!{T}(alpha::Real, param::MAPCostParam, x::T, gx::T, clr::Bool=false)
    if alpha == 0
	clr && gx[:] = 0
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
