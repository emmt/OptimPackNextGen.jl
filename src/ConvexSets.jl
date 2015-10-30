#
# ConvexSets.jl -
#
# Implement convex sets in Julia.
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl which is licensed under the MIT "Expat" License:
#
# Copyright (C) 2015, Éric Thiébaut.
#
#------------------------------------------------------------------------------

module ConvexSets

export project_variables!,  project_gradient!,  project_direction!
export ConvexSet, BoxedSet

# This abstract type is the base of all convex set types.
abstract ConvexSet

"""
A `BoxedSet` is a convex set with separable bounds on the variables:
`lower[i] ≤ x[i] ≤ upper[i]` where `-∞ ≤ lower[i] ≤ upper[i] ≤ +∞`.
Depending on the finiteness of the bounds and of their dependency on index
`i`, boxed sets come in different flavors.
"""
abstract BoxedSet{T} <: ConvexSet

type ScalarLowerBound{T} <: BoxedSet{T}
    lower::T
end

type ScalarUpperBound{T} <: BoxedSet{T}
    upper::T
end

type ScalarBounds{T} <: BoxedSet{T}
    lower::T
    upper::T
end

#------------------------------------------------------------------------------
# DEFAULT IMPLEMENTATIONS

# Default implementation for projecting a search direction.
function project_direction!{S<:ConvexSet,T}(dst::T, dom::S, x::T, d::T)
    negate!(dst, d)
    project_gradient!(dst, dom, x, dst)
    negate!(dst, dst)
end

# Default implementation for computing the step bounds.
function step_bounds{S<:ConvexSet,T}(dom::S, x::T, d::T)
    (minimum_step(dom, x, d),
     maximum_step(dom, x, d))
end

function shortcut_step{S<:ConvexSet,T}(dom::S, alpha::Real, x::T, d::T)
    min(alpha, maximum_step(dom, x, d))
end

#------------------------------------------------------------------------------
# SCALAR LOWER BOUND

function project_variables!{T,N}(dst::Array{T,N}, dom::ScalarLowerBound{T},
                                 x::Array{T,N})
    @assert(size(x) == size(dst))
    xmin::T = dom.lower
    for i in 1:length(x)
        dst[i] = max(x[i], xmin)
    end
end

function project_gradient!{T,N}(dst::Array{T,N}, dom::ScalarLowerBound{T},
                                x::Array{T,N}, g::Array{T,N})
    @assert(size(x) == size(dst))
    @assert(size(g) == size(dst))
    xmin::T = dom.lower
    @simd for i in 1:length(dst)
        @inbounds dst[i] = (g[i] < zero(T) || x[i] > xmin ? g[i] : zero(T))
    end
end

function project_direction!{T,N}(dst::Array{T,N}, dom::ScalarLowerBound{T},
                                 x::Array{T,N}, d::Array{T,N})
    @assert(size(x) == size(dst))
    @assert(size(d) == size(dst))
    xmin::T = dom.lower
    @simd for i in 1:length(dst)
        @inbounds dst[i] = (d[i] > zero(T) || x[i] > xmin ? d[i] : zero(T))
    end
end

function maximum_step{T,N}(dom::ScalarLowerBound{T},
                           x::Array{T,N}, d::Array{T,N})
    @assert(size(d) == size(x))
    amax::T = zero(T)
    xmin::T = dom.lower
    for i in 1:length(x)
        if d[i] >= zero(T)
            # inferior bound can only be encountered for a strictly
            # negative direction
            return inf(T)
        elseif x[i] + amax*d[i] > xmin
            # the bound is not reached, the step step can be enlarged
            amax = (xmin - x[i])/d[i]
        end
    end
    amax
end

function minimum_step{T,N}(dom::ScalarLowerBound{T},
                           x::Array{T,N}, d::Array{T,N})
    @assert(size(d) == size(x))
    unlimited::Bool = false
    amin::T = inf(T)
    xmin::T = dom.lower
    for i in 1:length(x)
        if d[i] < zero(T) && (unlimited || x[i] + amin*d[i] < xmin)
            amin = (xmin - x[i])/d[i]
            unlimited = (amin < inf(T))
        end
    end
    amin
end

#------------------------------------------------------------------------------
# SCALAR UPPER BOUND

function project_variables!{T,N}(dst::Array{T,N}, dom::ScalarUpperBound{T},
                                 x::Array{T,N})
    @assert(size(x) == size(dst))
    xmax::T = dom.upper
    for i in 1:length(x)
        dst[i] = min(x[i], xmax)
    end
end

function project_gradient!{T,N}(dst::Array{T,N}, dom::ScalarUpperBound{T},
                                x::Array{T,N}, g::Array{T,N})
    @assert(size(x) == size(dst))
    @assert(size(g) == size(dst))
    xmax::T = dom.upper
    @simd for i in 1:length(dst)
        @inbounds dst[i] = (g[i] > zero(T) || x[i] < xmax ? g[i] : zero(T))
    end
end

function project_direction!{T,N}(dst::Array{T,N}, dom::ScalarUpperBound{T},
                                 x::Array{T,N}, d::Array{T,N})
    @assert(size(x) == size(dst))
    @assert(size(d) == size(dst))
    xmax::T = dom.upper
    @simd for i in 1:length(dst)
        @inbounds dst[i] = (d[i] < zero(T) || x[i] < xmax ? d[i] : zero(T))
    end
end

function maximum_step{T,N}(dom::ScalarUpperBound{T},
                           x::Array{T,N}, d::Array{T,N})
    @assert(size(d) == size(x))
    amax::T = zero(T)
    xmax::T = dom.upper
    for i in 1:length(x)
        if d[i] <= zero(T)
            # superior bound can only be encountered for a strictly
            # positive direction
            return inf(T)
        elseif x[i] + amax*d[i] < xmax
            # the bound is not reached, the step step can be enlarged
            amax = (xmax - x[i])/d[i]
        end
    end
    amax
end

function minimum_step{T,N}(dom::ScalarUpperBound{T},
                           x::Array{T,N}, d::Array{T,N})
    @assert(size(d) == size(x))
    unlimited::Bool = false
    amin::T = inf(T)
    xmax::T = dom.upper
    for i in 1:length(x)
        if d[i] > zero(T) && (unlimited || x[i] + amin*d[i] > xmax)
            amin = (xmax - x[i])/d[i]
            unlimited = (amin < inf(T))
        end
    end
    amin
end

end # module

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
