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

import Base: convert, sign
import ..Algebra: project_variables!, project_direction!

export project_variables!,  project_gradient!,  project_direction!
export minimum_step, maximum_step, shortcut_step
export ConvexSet, AbstractBoundedSet, ScalarLowerBound, ScalarUpperBound,
       ScalarBounds, BoxedSet
export Ascent, Descent


"""

In an optimization problem, the search direction `d` can be either a `Descent`
or an `Ascent` direction, depending on how the variables `x` of the problem are
changed along the line search:

    x = x0 + α d      for an descent direction
    x = x0 - α d      for an ascent direction

where `d` is the search direction, 'x0` are the current variables and `α > 0`
is the step length.  The orientation of the search direction can also be
specified by a strictly positive value for a `Descent` direction or a strictly
negative value for an `Ascent` direction.
"""
abstract Orientation
immutable Ascent  <: Orientation; end
immutable Descent <: Orientation; end
convert{T<:Orientation}(::Type{T}, ::Any) = error("invalid orientation")
convert{T<:Orientation}(::Type{T}, ::Union{Ascent,Type{Ascent}}) = Ascent
convert{T<:Orientation}(::Type{T}, ::Union{Descent,Type{Descent}}) = Descent
convert{T<:Orientation}(::Type{T}, s::Real) = (s < 0 ? Ascent :
                                               s > 0 ? Descent :
                                               error("invalid orientation"))
sign(::Union{Ascent,Type{Ascent}}) = -1
sign(::Union{Descent,Type{Descent}}) = 1

# This abstract type is the base of all convex set types.
abstract ConvexSet

"""
An `AbstractBoundedSet` is a convex set with bounds on the variables:
`lower[i] ≤ x[i] ≤ upper[i]` where `-∞ ≤ lower[i] ≤ upper[i] ≤ +∞`.
Depending on the finiteness of the bounds and of their dependency on index
`i`, boxed sets come in different flavors.
"""
abstract AbstractBoundedSet{T} <: ConvexSet

type ScalarLowerBound{T} <: AbstractBoundedSet{T}
    lower::T
    function ScalarLowerBound(xl::T)
        isnan(xl) && error("invalid lower bound")
        new(xl)
    end
end
ScalarLowerBound{T<:AbstractFloat}(xl::T) = ScalarLowerBound{T}(xl)

type ScalarUpperBound{T} <: AbstractBoundedSet{T}
    upper::T
    function ScalarUpperBound(xu::T)
        isnan(xu) && error("invalid upper bound")
        new(xu)
    end
end
ScalarUpperBound{T<:AbstractFloat}(xu::T) = ScalarUpperBound{T}(xu)

type ScalarBounds{T} <: AbstractBoundedSet{T}
    lower::T
    upper::T
    function ScalarBounds(xl::T, xu::T)
        isnan(xl) && error("invalid lower bound")
        isnan(xu) && error("invalid upper bound")
        xl > xu   && error("incompatible bounds")
        new(xl, xu)
    end
end
function ScalarBounds{R<:Real,S<:Real}(xl::R, xu::S)
    T = promote_type(R, S)
    ScalarBounds{T}(convert(T, xl), convert(T, xu))
end

type BoxedSet{T,N} <: AbstractBoundedSet{T}
    lower::Array{T,N}
    upper::Array{T,N}
    function BoxedSet(xl::Array{T,N}, xu::Array{T,N})
        @assert(size(xl) == size(xu))
        @inbounds begin
            @simd for i in 1:length(xl)
                if isnan(xl[i])
                    error("invalid lower bound")
                end
                if isnan(xu[i])
                    error("invalid upper bound")
                end
                if xl[i] > xu[i]
                    error("incompatible bounds")
                end
            end
        end
        new(xl, xu)
    end
end

#------------------------------------------------------------------------------
# DEFAULT IMPLEMENTATIONS
#
# The various sub-types of `ConvexSet` shall only implement method
# `project_direction!` with a `orient` parameter to indicate whether the last
# argument is an ascent or a descent direction.

function project_gradient!{T}(dst::T, dom::ConvexSet, x::T, g::T)
    project_direction!(dst, dom, x, Ascent, g)
end

function project_direction!{T}(dst::T, dom::ConvexSet, x::T, d::T)
    project_direction!(dst, dom, x, Descent, d)
end

function project_direction!{T}(dst::T, dom::ConvexSet, x::T, orient::Real, d::T)
    project_direction!!(dst, dom, x, convert(Orientation, orient), g)
end

# Default implementation for computing the step bounds.
function step_bounds{S<:ConvexSet,T}(dom::S, x::T, d::T)
    (minimum_step(dom, x, d),
     maximum_step(dom, x, d))
end

function shortcut_step{S<:ConvexSet,T}(alpha::Real, dom::S, x::T, d::T)
    min(alpha, maximum_step(dom, x, d))
end

#------------------------------------------------------------------------------
# SCALAR LOWER BOUND

function project_variables!{T,N}(dst::Array{T,N}, dom::ScalarLowerBound{T},
                                 x::Array{T,N})
    @assert(size(dst) == size(x))
    const xl::T = dom.lower
    for i in 1:length(x)
        @inbounds dst[i] = max(x[i], xl)
    end
end

function project_direction!{T,N}(dst::Array{T,N},
                                 dom::ScalarLowerBound{T},
                                 x::Array{T,N},
                                 orient::Union{Type{Ascent},Type{Descent}},
                                 d::Array{T,N})
    @assert(size(dst) == size(x))
    @assert(size(d)   == size(x))
    const xl::T = dom.lower
    if orient == Descent
        # Make a descent direction feasible.
        @simd for i in 1:length(x)
            @inbounds dst[i] = (d[i] > 0 || x[i] > xl) ? d[i] : 0
        end
    else
        # Make an ascent direction feasible.
        @simd for i in 1:length(x)
            @inbounds dst[i] = (d[i] < 0 || x[i] > xl) ? d[i] : 0
        end
    end
end

"""
Compute smallest step which will bring all variables "out of bounds".

See Also: minimum_step
"""
function maximum_step{T,N}(dom::ScalarLowerBound{T},
                           x::Array{T,N}, d::Array{T,N})
    @assert(size(d) == size(x))
    amax::T = zero(T)
    xl::T = dom.lower
    for i in 1:length(x)
        if d[i] > zero(T)
            # inferior bound can only be encountered for a strictly
            # negative direction
            return convert(T, Inf)
        elseif d[i] < zero(T) && x[i] + amax*d[i] > xl
            # the bound is not reached, the step step can be enlarged
            amax = (xl - x[i])/d[i]
        end
    end
    amax
end

"""
Compute smallest step which will bring at least one variable "out of bounds".

See Also: maximum_step
"""
function minimum_step{T,N}(dom::ScalarLowerBound{T},
                           x::Array{T,N}, d::Array{T,N})
    @assert(size(d) == size(x))
    unlimited::Bool = false
    amin::T = convert(T, Inf)
    xl::T = dom.lower
    for i in 1:length(x)
        if d[i] < zero(T) && (unlimited || x[i] + amin*d[i] < xl)
            amin = (xl - x[i])/d[i]
            unlimited = (amin < convert(T, Inf))
        end
    end
    amin
end

#------------------------------------------------------------------------------
# SCALAR UPPER BOUND

function project_variables!{T,N}(dst::Array{T,N}, dom::ScalarUpperBound{T},
                                 x::Array{T,N})
    @assert(size(x) == size(dst))
    const xu::T = dom.upper
    for i in 1:length(x)
        @inbounds dst[i] = min(x[i], xu)
    end
end

function project_direction!{T,N}(dst::Array{T,N},
                                 dom::ScalarUpperBound{T},
                                 x::Array{T,N},
                                 orient::Union{Type{Ascent},Type{Descent}},
                                 d::Array{T,N})
    @assert(size(dst) == size(x))
    @assert(size(d)   == size(x))
    const xu::T = dom.upper
    if orient == Descent
        # Make a descent direction feasible.
        @simd for i in 1:length(x)
            @inbounds dst[i] = (d[i] < 0 || x[i] < xu) ? d[i] : 0
        end
    else
        # Make an ascent direction feasible.
        @simd for i in 1:length(x)
            @inbounds dst[i] = (d[i] > 0 || x[i] < xu) ? d[i] : 0
        end
    end
end

function maximum_step{T,N}(dom::ScalarUpperBound{T},
                           x::Array{T,N}, d::Array{T,N})
    @assert(size(d) == size(x))
    amax::T = zero(T)
    xu::T = dom.upper
    for i in 1:length(x)
        if d[i] < zero(T)
            # superior bound can only be encountered for a strictly
            # positive direction
            return convert(T, Inf)
        elseif d[i] > zero(T) && x[i] + amax*d[i] < xu
            # the bound is not reached, the step step can be enlarged
            amax = (xu - x[i])/d[i]
        end
    end
    amax
end

function minimum_step{T,N}(dom::ScalarUpperBound{T},
                           x::Array{T,N}, d::Array{T,N})
    @assert(size(d) == size(x))
    unlimited::Bool = false
    amin::T = convert(T, Inf)
    xu::T = dom.upper
    for i in 1:length(x)
        if d[i] > zero(T) && (unlimited || x[i] + amin*d[i] > xu)
            amin = (xu - x[i])/d[i]
            unlimited = (amin < convert(T, Inf))
        end
    end
    amin
end

#------------------------------------------------------------------------------
# BOXED SET

function project_variables!{T,N}(dst::Array{T,N}, dom::BoxedSet{T,N},
                                 x::Array{T,N})
    xl = dom.lower
    xu = dom.upper
    @assert(size(xl) == size(x))
    @assert(size(xu) == size(x))
    @inbounds begin
        @simd for i in 1:length(x)
            dst[i] = max(xl[i], min(x[i], xu[i]))
        end
    end
end

function project_direction!{T,N}(dst::Array{T,N},
                                 dom::BoxedSet{T},
                                 x::Array{T,N},
                                 orient::Union{Type{Ascent},Type{Descent}},
                                 d::Array{T,N})
    xl = dom.lower
    xu = dom.upper
    @assert(size(xl)  == size(x))
    @assert(size(xu)  == size(x))
    @assert(size(dst) == size(x))
    @assert(size(d)   == size(x))
    @inbounds begin
        if orient == Descent
            # Make a descent direction feasible.
            @simd for i in 1:length(x)
                s = d[i]
                dst[i] = (s >= 0 ? (x[i] < xu[i]) : (x[i] > xl[i])) ? s : 0
            end
        else
            # Make an ascent direction feasible.
            @simd for i in 1:length(x)
                s = d[i]
                dst[i] = (s <= 0 ? (x[i] < xu[i]) : (x[i] > xl[i])) ? s : 0
            end
        end
    end
end

"""
### Compute step bounds for line search

The step `stpmin` to the closest bound, the step `wolfemin` to the closest not
yet reached bound and the step `stpmax` to the farthest bound are computed by
the call:
```
    (stpmin, wolfemin, stpmax) = stepbounds(dom, x, s, d)
```
where `dom` is a boxed convex set, `x` are the current variables and
`sign(s)*d` is the search direction.

In orther words, `stpmin` is the smallest step which will bring at least one
variable "out of bounds" and `stpmax` is the smallest step which will bring all
variables "out of bounds".
"""
function stepbounds{T,N}(dom::BoxedSet{T,N},
                         x::Array{T,N},
                         orient::Union{Type{Ascent},Type{Descent}},
                         d::Array{T,N})
    xl = dom.lower
    xu = dom.upper
    @assert(size(xl) == size(x))
    @assert(size(xu) == size(x))
    @assert(size(d)  == size(x))

    a1 = convert(T, Inf)
    a2 = convert(T, Inf)
    a3 = convert(T, 0)
    s = convert(T, sign(orient))
    for i in 1:length(x)
        p = s*dx[i]
        if p > 0
            # Compute the step length to reach the upper bound.
            a = (xu[i] - x[i])/p
            a1 = min(a1, a)
            if a > 0 # FIXME: was a1 > 0
                a2 = min(a2, a)
            end
            a3 = max(a3, a)
        elseif p < 0
            # Compute the step length to reach the lower bound.
            a = (xl[i] - x[i])/p
            a1 = min(a1, a)
            if a > 0 # FIXME: was a1 > 0
                a2 = min(a2, a)
            end
            a3 = max(a3, a)
        end
    end
    return (a1, a2, a3) # boundmin, wolfemin, boundmax
end

end # module
