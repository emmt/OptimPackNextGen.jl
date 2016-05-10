#
# algebra.jl --
#
# Implement basic operations for *vectors*.  Here arrays of any rank are
# considered as *vectors*, the only requirements are that, when combining
# *vectors*, they have the same type and dimensions.  These methods are
# intended to be used for numerical optimization and thus, for now,
# elements must be real (not complex).
#
#------------------------------------------------------------------------------
#
# This file is part of TiPi.jl licensed under the MIT "Expat" License.
#
# Copyright (C) 2015-2016, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

module Algebra

export inner, norm1, norm2, normInf, swap!, update!, combine!,
       project_variables!, project_direction!, step_limits, get_free_variables

# Use the same floating point type for scalars as in TiPi.
import ..Float

"""
### Euclidean norm

The Euclidean (L2) norm of `v` can be computed by:
```
    norm2(v)
```
"""
function norm2{T<:AbstractFloat,N}(v::Array{T,N})
    s::T = zero(T)
    @simd for i in 1:length(v)
        @inbounds s += v[i]*v[i]
    end
    return Float(sqrt(s))
end

"""
### L1 norm

The L1 norm of `v` can be computed by:
```
    norm1(v)
```
"""
function norm1{T<:AbstractFloat,N}(v::Array{T,N})
    s::T = zero(T)
    @simd for i in 1:length(v)
        @inbounds s += abs(v[i])
    end
    return Float(s)
end

"""
### Infinite norm

The infinite norm of `v` can be computed by:
```
    normInf(v)
```
"""
function normInf{T<:AbstractFloat,N}(v::Array{T,N})
    s::T = zero(T)
    @simd for i in 1:length(v)
        @inbounds s = max(s, abs(v[i]))
    end
    return Float(s)
end

"""
### Compute scalar product

The call:
```
    inner(x, y)
```
computes the inner product (a.k.a. scalar product) between `x` and `y` (which
must have the same size).  The triple inner product between `w`, `x` and `y`
can be computed by:
```
    inner(w, x, y)
```
Finally:
```
    inner(sel, x, y)
```
computes the sum of the product of the elements of `x` and `y` whose indices
are given by the `sel` argument.
"""
function inner{T<:AbstractFloat,N}(x::Array{T,N}, y::Array{T,N})
    @assert(size(x) == size(y))
    s::T = 0
    @simd for i in 1:length(x)
        @inbounds s += x[i]*y[i]
    end
    return Float(s)
end

function inner{T<:AbstractFloat,N}(w::Array{T,N}, x::Array{T,N}, y::Array{T,N})
    @assert(size(x) == size(w))
    @assert(size(y) == size(w))
    s::T = 0
    @simd for i in 1:length(w)
        @inbounds s += w[i]*x[i]*y[i]
    end
    return Float(s)
end

function inner{T<:AbstractFloat,N}(sel::Vector{Int}, x::Array{T,N}, y::Array{T,N})
    @assert(size(y) == size(x))
    s::T = 0
    const n = length(x)
    @simd for i in 1:length(sel)
        j = sel[i]
        1 <= j <= n || throw(BoundsError())
        @inbounds s += x[j]*y[j]
    end
    return Float(s)
end

"""
### Exchange contents

The call:
```
    swap!(x, y)
```
exchanges the contents of `x` and `y` (which must have the same size).
"""
function swap!{T,N}(x::Array{T,N}, y::Array{T,N})
    @assert(size(x) == size(y))
    temp::T
    @inbounds begin
        @simd for i in 1:length(x)
            temp = x[i]
            x[i] = y[i]
            y[i] = temp
        end
    end
end

"""
### Increment an array by a scaled step

The call:
```
    update!(dst, alpha, x)
```
increments the components of the destination *vector* `dst` by those of
`alpha*x`.  The code is optimized for some specific values of the multiplier
`alpha`.  For instance, if `alpha` is zero, then `dst` left unchanged without
using `x`.

Another possibility is:
```
    update!(dst, sel, alpha, x)
```
with `sel` a selection of indices to which apply the operation.

"""
function update!{T<:AbstractFloat,N}(dst::Array{T,N},
                                     a::T, x::Array{T,N})
    @assert(size(x) == size(dst))
    const n = length(dst)
    @inbounds begin
        if a == one(T)
            @simd for i in 1:n
                dst[i] += x[i]
            end
        elseif a == -one(T)
            @simd for i in 1:n
                dst[i] -= x[i]
            end
        elseif a != zero(T)
            @simd for i in 1:n
                dst[i] += a*x[i]
            end
        end
    end
end

function update!{T<:AbstractFloat,N}(dst::Array{T,N}, sel::Vector{Int},
                                     a::T, x::Array{T,N})
    @assert(size(x) == size(dst))
    const m = length(sel)
    const n = length(dst)
    if a == one(T)
        @simd for i in 1:m
            j = sel[i]
            1 <= j <= n || throw(BoundsError())
            @inbounds dst[j] += x[j]
        end
    elseif a == -one(T)
        @simd for i in 1:m
            j = sel[i]
            1 <= j <= n || throw(BoundsError())
            @inbounds dst[j] -= x[j]
        end
    elseif a != zero(T)
        @simd for i in 1:m
            j = sel[i]
            1 <= j <= n || throw(BoundsError())
            @inbounds dst[j] += a*x[j]
        end
    end
end

"""
### Linear combination of arrays

The calls:
```
    combine!(dst, alpha, x)
    combine!(dst, alpha, x, beta, y)
```
stores the linear combinations `alpha*x` and `alpha*x + beta*y` into the
destination array `dst`.  The code is optimized for some specific values of the
coefficients `alpha` and `beta`.  For instance, if `alpha` (resp. `beta`) is
zero, then the contents of `x` (resp. `y`) is not used.

The source array(s) and the destination an be the same.  For instance, the two
following lines of code produce the same result:
```
    combine!(dst, 1, dst, alpha, x)
    update!(dst, alpha, x)
```
"""

function combine!{T<:Real,N}(dst::Array{T,N}, a::T, x::Array{T,N})
    @assert(size(x) == size(dst))
    const n = length(dst)
    @inbounds begin
        if a == zero(T)
            @simd for i in 1:n
                dst[i] = a
            end
        elseif a == one(T)
            @simd for i in 1:n
                dst[i] = x[i]
            end
        elseif a == -one(T)
            @simd for i in 1:n
                dst[i] = -x[i]
            end
        else
            @simd for i in 1:n
                dst[i] = a*x[i]
            end
        end
    end
end

function combine!{T<:AbstractFloat,N}(dst::Array{T,N},
                                      a::T, x::Array{T,N},
                                      b::T, y::Array{T,N})
    @assert(size(x) == size(dst))
    @assert(size(y) == size(dst))
    const n = length(dst)
    @inbounds begin
        if a == zero(T)
            combine!(dst, b, y)
        elseif b == zero(T)
            combine!(dst, a, x)
        elseif a == one(T)
            if b == one(T)
                @simd for i in 1:n
                    dst[i] = x[i] + y[i]
                end
            elseif b == -one(T)
                @simd for i in 1:n
                    dst[i] = x[i] - y[i]
                end
            else
                @simd for i in 1:n
                    dst[i] = x[i] + b*y[i]
                end
            end
        elseif a == -one(T)
            if b == one(T)
                @simd for i in 1:n
                    dst[i] = y[i] - x[i]
                end
            elseif b == -one(T)
                @simd for i in 1:n
                    dst[i] = -x[i] - y[i]
                end
            else
                @simd for i in 1:n
                    dst[i] = b*y[i] - x[i]
                end
            end
        else
            if b == one(T)
                @simd for i in 1:n
                    dst[i] = a*x[i] + y[i]
                end
            elseif b == -one(T)
                @simd for i in 1:n
                    dst[i] = a*x[i] - y[i]
                end
            else
                @simd for i in 1:n
                    dst[i] = a*x[i] + b*y[i]
                end
            end
        end
    end
end

function update!{T<:AbstractFloat,N}(dst::Array{T,N},
                                     alpha::Real, x::Array{T,N})
    update!(dst, T(alpha), x)
end

function combine!{T<:Real,N}(dst::Array{T,N},
                             alpha::Real, x::Array{T,N})
    combine!(dst, T(alpha), x)
end

function combine!{T<:Real,N}(dst::Array{T,N},
                             alpha::Real, x::Array{T,N},
                             beta::Real,  y::Array{T,N})
    combine!(dst, T(alpha), x, T(beta), y)
end

#-------------------------------------------------------------------------------
# PROJECTING VARIABLES

@inline clamp{T<:Real}(x::T, xl::T, xu::T) = max(xl, min(x, xu))
@inline clamp{T<:Real}(x::T, ::Type{Void}, xu::T) = min(x, xu)
@inline clamp{T<:Real}(x::T, xl::T, ::Type{Void}) = max(xl, x)

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       xl::Real, xu::Real,
                                       x::Array{T,N})
    project_variables!(dst, convert(T, xl), convert(T, xu), x)
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       xl::T, xu::T,
                                       x::Array{T,N})
    @assert size(dst) == size(x)
    @assert xl ≤ xu # this also check for NaN
    bounded_below = xl > convert(T, -Inf)
    bounded_above = xu < convert(T, +Inf)
    if bounded_below
        if bounded_above
            @simd for i in 1:length(x)
                @inbounds dst[i] = clamp(x[i], xl, xu)
            end
        else
            @simd for i in 1:length(x)
                @inbounds dst[i] = clamp(x[i], xl, nothing)
            end
        end
    else
        if bounded_above
            @simd for i in 1:length(x)
                @inbounds dst[i] = clamp(x[i], nothing, xu)
            end
        elseif dst != x
            copy!(dst, x)
        end
    end
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       xl::Array{T,N}, xu::Real,
                                       x::Array{T,N})
    project_variables!(dst, xl, convert(T, xu), x)
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       xl::Array{T,N}, xu::T,
                                       x::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(xl)  == size(x)
    bounded_above = xu < convert(T, +Inf)
    if bounded_above
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], xl[i], xu)
        end
    else
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], xl[i], nothing)
        end
    end
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       xl::Real, xu::Array{T,N},
                                       x::Array{T,N})
    project_variables!(dst, convert(T, xl), xu, x)
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       xl::T, xu::Array{T,N},
                                       x::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(xu)  == size(x)
    bounded_below = xl > convert(T, -Inf)
    if bounded_below
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], xl, xu[i])
        end
    else
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], nothing, xu[i])
        end
    end
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       xl::Array{T,N}, xu::Array{T,N},
                                       x::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(xl)  == size(x)
    @assert size(xu)  == size(x)
    @simd for i in 1:length(x)
        @inbounds dst[i] = clamp(x[i], xl[i], xu[i])
    end
end

#------------------------------------------------------------------------------
# PROJECTING DIRECTION

abstract Orientation
immutable Forward  <: Orientation; end
immutable Backward <: Orientation; end

convert{T<:Orientation}(::Type{T}, s::Real) = (s > 0 ? Forward :
                                               s < 0 ? Backward :
                                               error("invalid orientation"))
sign(::Union{Forward,Type{Forward}}) = +1
sign(::Union{Backward,Type{Backward}}) = -1


function orientation(T::DataType, x)
    s = sign(x)
    s != 0 || error("invalid orientation")
    convert(T, s)
end

Orientation(::Union{Forward,Type{Forward}}) = Forward
Orientation(::Union{Backward,Type{Backward}}) = Backward
Orientation(x) = convert(Orientation, x)

@inline function project_forward{T<:Real}(x::T, d::T, xl::T, xu::T)
    (d > zero(T) ? x < xu : x > xl) ? d : zero(T)
end

@inline function project_forward{T<:Real}(x::T, d::T, ::Type{Void}, xu::T)
    (d < zero(T) || x < xu) ? d : zero(T)
end

@inline function project_forward{T<:Real}(x::T, d::T, xl::T, ::Type{Void})
    (d > zero(T) || x > xl) ? d : zero(T)
end

@inline function project_backward{T<:Real}(x::T, d::T, xl::T, xu::T)
    (d < zero(T) ? x < xu : x > xl) ? d : zero(T)
end

@inline function project_backward{T<:Real}(x::T, d::T, ::Type{Void}, xu::T)
    (d > zero(T) || x < xu) ? d : zero(T)
end

@inline function project_backward{T<:Real}(x::T, d::T, xl::T, ::Type{Void})
    (d < zero(T) || x > xl) ? d : zero(T)
end

function project_gradient!{T<:Real,N}(dst::Array{T,N},
                                      xl::Union{Real,Array{T,N}},
                                      xu::Union{Real,Array{T,N}},
                                      x::Array{T,N},
                                      d::Array{T,N})
    project_direction!(dst, xl, xu, x, -1, d)
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       xl::Real, xu::Real,
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    project_direction!(dst, convert(T, xl), convert(T, xu), x, orient, d)
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       xl::T, xu::T,
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(d)   == size(x)
    @assert xl ≤ xu # this also check for NaN
    const forward = Orientation(orient) == Forward
    const bounded_above = xu < convert(T, +Inf)
    const bounded_below = xl > convert(T, -Inf)
    @inbounds begin
        if bounded_below && bounded_above
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], xl, xu)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], xl, xu)
                end
            end
        elseif bounded_below
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], xl, nothing)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], xl, nothing)
                end
            end
        elseif bounded_above
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], nothing, xu)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], nothing, xu)
                end
            end
        else
            if dst != d
                copy!(dst, d)
            end
        end
    end
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       xl::Array{T,N}, xu::Real,
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    project_direction!(dst, xl, convert(T, xu), x, orient, d)
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       xl::Array{T,N}, xu::T,
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(xl)  == size(x)
    @assert size(d)   == size(x)
    const forward = Orientation(orient) == Forward
    const bounded_above = xu < convert(T, +Inf)
    @inbounds begin
        if bounded_above
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], xl[i], xu)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], xl[i], xu)
                end
            end
        else
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], xl[i], nothing)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], xl[i], nothing)
                end
            end
        end
    end
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       xl::Real, xu::Array{T,N},
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    project_direction!(dst, convert(T, xl), xu, x, orient, d)
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       xl::T, xu::Array{T,N},
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(xu)  == size(x)
    @assert size(d)   == size(x)
    const forward = Orientation(orient) == Forward
    const bounded_below = xl > convert(T, -Inf)
    @inbounds begin
        if bounded_below
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], xl, xu[i])
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], xl, xu[i])
                end
            end
        else
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], nothing, xu[i])
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], nothing, xu[i])
                end
            end
        end
    end
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       xl::Array{T,N}, xu::Array{T,N},
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(xl)  == size(x)
    @assert size(xu)  == size(x)
    @assert size(d)   == size(x)
    const forward = Orientation(orient) == Forward
    @inbounds begin
        if forward
            @simd for i in 1:length(x)
                dst[i] = project_forward(x[i], d[i], xl[i], xu[i])
            end
        else
            @simd for i in 1:length(x)
                dst[i] = project_backward(x[i], d[i], xl[i], xu[i])
            end
        end
    end
end

#------------------------------------------------------------------------------
# COMPUTING STEP LIMITS

"""
### Compute step limits for line search

When there are separable bound constraints on the variables, the step `smin` to
the closest not yet reached bound and the step `smax` to the farthest bound are
computed by the call:
```
    (smin, smax) = step_limits(xl, xu, x, s, d)
```
where `xl` is the lower bound, `xu` is the upper bound, `x` are the current
variables and `sign(s)*d` is the search direction.

In orther words, `smin` is the smallest step which will bring at least one more
variable "out of bounds" and `smax` is the smallest step which will bring all
variables "out of bounds".  As a consequence, `0 < smin` and `0 ≤ smax`.
"""
function step_limits{T<:Real,N}(xl::Real, xu::Real, x::Array{T,N},
                                orient, d::Array{T,N})
    step_limits(convert(T, xl), convert(T, xu), x, orient, d)
end

function step_limits{T<:Real,N}(xl::T, xu::T, x::Array{T,N},
                                orient, d::Array{T,N})
    @assert size(d) == size(x)
    @assert xl ≤ xu # this also check for NaN
    const ZERO = zero(T)
    const s = orientation(T, orient)
    const bounded_below = xl > convert(T, -Inf)
    const bounded_above = xu < convert(T, +Inf)
    smin = convert(T, Inf)
    smax = ZERO
    @inbounds begin
        if bounded_below && bounded_above
            @simd for i in 1:length(x)
                p = s*d[i]
                if p != ZERO
                    a = (p > ZERO ? xu - x[i] : xl - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        elseif bounded_below
            @simd for i in 1:length(x)
                p = s*d[i]
                if p < ZERO
                    a = (xl - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        elseif bounded_above
            @simd for i in 1:length(x)
                p = s*d[i]
                if p > ZERO
                    a = (xu - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        end
    end
    return (smin, smax)
end

function step_limits{T<:Real,N}(xl::Array{T,N}, xu::Real, x::Array{T,N},
                                orient, d::Array{T,N})
    step_limits(xl, convert(T, xu), x, orient, d)
end

function step_limits{T<:Real,N}(xl::Array{T,N}, xu::T,
                                x::Array{T,N},
                                orient, d::Array{T,N})
    @assert size(xl) == size(x)
    @assert size(d)  == size(x)
    const ZERO = zero(T)
    const s = orientation(T, orient)
    const bounded_above = xu < convert(T, +Inf)
    smin = convert(T, Inf)
    smax = ZERO
    @inbounds begin
        if bounded_above
            @simd for i in 1:length(x)
                p = s*d[i]
                if p != ZERO
                    a = (p > ZERO ? xu - x[i] : xl[i] - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        else
            @simd for i in 1:length(x)
                p = s*d[i]
                if p < ZERO
                    a = (xl[i] - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        end
    end
    return (smin, smax)
end

function step_limits{T<:Real,N}(xl::Real, xu::Array{T,N}, x::Array{T,N},
                                orient, d::Array{T,N})
    step_limits(convert(T, xl), xu, x, orient, d)
end

function step_limits{T<:Real,N}(xl::T, xu::Array{T,N}, x::Array{T,N},
                                orient, d::Array{T,N})
    @assert size(xu) == size(x)
    @assert size(d)  == size(x)
    const ZERO = zero(T)
    const s = orientation(T, orient)
    const bounded_below = xl > convert(T, -Inf)
    smin = convert(T, Inf)
    smax = ZERO
    @inbounds begin
        if bounded_below
            @simd for i in 1:length(x)
                p = s*d[i]
                if p != ZERO
                    a = (p > ZERO ? xu[i] - x[i] : xl - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        else
            @simd for i in 1:length(x)
                p = s*d[i]
                if p > ZERO
                    a = (xu[i] - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        end
    end
    return (smin, smax)
end

function step_limits{T<:Real,N}(xl::Array{T,N}, xu::Array{T,N}, x::Array{T,N},
                                orient, d::Array{T,N})
    @assert size(xl) == size(x)
    @assert size(xu) == size(x)
    @assert size(d)  == size(x)
    const ZERO = zero(T)
    const s = orientation(T, orient)
    smin = convert(T, Inf)
    smax = ZERO
    @inbounds begin
        @simd for i in 1:length(x)
            p = s*d[i]
            if p != ZERO
                # Step length to reach the upper/lower bound:
                a = (p > ZERO ? xu[i] - x[i] : xl[i] - x[i])/p
                if 0 < a < smin
                    smin = a
                end
                if a > smax
                    smax = a
                end
            end
        end
    end
    return (smin, smax)
end

#------------------------------------------------------------------------------
# GETTING FREE VARIABLES

@inline function may_move_forward{T<:Real}(x::T, d::T, xl::T, ::Type{Void})
    d > zero(T) || (d != zero(T) && x > xl)
end

@inline function may_move_backward{T<:Real}(x::T, d::T, xl::T, ::Type{Void})
    d < zero(T) || (d != zero(T) && x > xl)
end

@inline function may_move_forward{T<:Real}(x::T, d::T, ::Type{Void}, xu::T)
    d < zero(T) || (d != zero(T) && x < xu)
end

@inline function may_move_backward{T<:Real}(x::T, d::T, ::Type{Void}, xu::T)
    d > zero(T) || (d != zero(T) && x < xu)
end

@inline function may_move_forward{T<:Real}(x::T, d::T, xl::T, xu::T)
    d != zero(T) && (d < zero(T) ? x > xl : x < xu)
end

@inline function may_move_backward{T<:Real}(x::T, d::T, xl::T, xu::T)
    d != zero(T) && (d > zero(T) ? x > xl : x < xu)
end

"""

Get free variables when following a direction.

"""

function get_free_variables{T<:Real,N}(xl::Real, xu::Real, x::Array{T,N},
                                       orient, d::Array{T,N})
    get_free_variables(convert(T, xl), convert(T, xu), x, orient, d)
end

function get_free_variables{T<:Real,N}(xl::T, xu::T, x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(d)  == size(x)
    @assert xl ≤ xu # this also check for NaN
    const forward = Orientation(orient) == Forward
    const bounded_below = xl > convert(T, -Inf)
    const bounded_above = xu < convert(T, +Inf)
    const n = length(x)
    sel = Array(Int, n)
    j = 0
    @inbounds begin
        if bounded_below && bounded_above
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], xl, xu)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], xl, xu)
                        j += 1
                        sel[j] = i
                    end
                end
            end
        elseif bounded_below
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], xl, nothing)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], xl, nothing)
                        j += 1
                        sel[j] = i
                    end
                end
            end
        elseif bounded_above
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], nothing, xu)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], nothing, xu)
                        j += 1
                        sel[j] = i
                    end
                end
            end
        else
            @simd for i in 1:n
                sel[i] = i
            end
            j = n
        end
    end
    return (j == n ? sel : (j > 0 ? sel[1:j] : Array(Int, 0)))
end

function get_free_variables{T<:Real,N}(xl::Array{T,N}, xu::Real, x::Array{T,N},
                                       orient, d::Array{T,N})
    get_free_variables(xl, convert(T, xu), x, orient, d)
end

function get_free_variables{T<:Real,N}(xl::Array{T,N}, xu::T, x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(d)  == size(x)
    @assert size(xl) == size(x)
    const forward = Orientation(orient) == Forward
    const bounded_above = xu < convert(T, +Inf)
    const n = length(x)
    sel = Array(Int, n)
    j = 0
    @inbounds begin
        if bounded_above
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], xl[i], xu)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], xl[i], xu)
                        j += 1
                        sel[j] = i
                    end
                end
            end
        else
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], xl[i], nothing)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], xl[i], nothing)
                        j += 1
                        sel[j] = i
                    end
                end
            end
        end
    end
    return (j == n ? sel : (j > 0 ? sel[1:j] : Array(Int, 0)))
end

function get_free_variables{T<:Real,N}(xl::Real, xu::Real, x::Array{T,N},
                                       orient, d::Array{T,N})
    get_free_variables(convert(T, xl), convert(T, xu), x, orient, d)
end

function get_free_variables{T<:Real,N}(xl::T, xu::Array{T,N}, x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(d)  == size(x)
    @assert size(xu) == size(x)
    const forward = Orientation(orient) == Forward
    const bounded_below = xl > convert(T, -Inf)
    const n = length(x)
    sel = Array(Int, n)
    j = 0
    @inbounds begin
        if bounded_below
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], xl, xu[i])
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], xl, xu[i])
                        j += 1
                        sel[j] = i
                    end
                end
            end
        else
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], nothing, xu[i])
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], nothing, xu[i])
                        j += 1
                        sel[j] = i
                    end
                end
            end
        end
    end
    return (j == n ? sel : (j > 0 ? sel[1:j] : Array(Int, 0)))
end

function get_free_variables{T<:Real,N}(xl::Array{T,N}, xu::Array{T,N}, x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(d)  == size(x)
    @assert size(xl) == size(x)
    @assert size(xu) == size(x)
    const forward = Orientation(orient) == Forward
    const n = length(x)
    sel = Array(Int, n)
    j = 0
    @inbounds begin
        if forward
            @simd for i in 1:n
                if may_move_forward(x[i], d[i], xl[i], xu[i])
                    j += 1
                    sel[j] = i
                end
            end
        else
            @simd for i in 1:n
                if may_move_backward(x[i], d[i], xl[i], xu[i])
                    j += 1
                    sel[j] = i
                end
            end
        end
    end
    return (j == n ? sel : (j > 0 ? sel[1:j] : Array(Int, 0)))
end

end # module
