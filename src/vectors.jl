#
# vectors.jl --
#
# Implement basic operations for *vectors*.  Here arrays of any rank are
# considered as *vectors*, the only requirements are that, when combining
# *vectors*, they have the same type and dimensions.  These methods are
# intended to be used for numerical optimization and thus, for now,
# elements must be real (not complex).
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

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

    inner(x, y)

computes the inner product (a.k.a. scalar or dot product) between `x` and `y`
(which must have the same size).  The triple inner product between `w`, `x` and
`y` can be computed by:

    inner(w, x, y)

Finally:

    inner(sel, x, y)

computes the sum of the product of the elements of `x` and `y` whose indices
are given by the `sel` argument.
"""
function inner{T<:AbstractFloat,N}(x::Array{T,N}, y::Array{T,N})
    @assert size(x) == size(y)
    s::T = 0
    @simd for i in 1:length(x)
        @inbounds s += x[i]*y[i]
    end
    return Float(s)
end

function inner{T<:AbstractFloat,N}(w::Array{T,N}, x::Array{T,N}, y::Array{T,N})
    @assert size(x) == size(w)
    @assert size(y) == size(w)
    s::T = 0
    @simd for i in 1:length(w)
        @inbounds s += w[i]*x[i]*y[i]
    end
    return Float(s)
end

function inner{T<:AbstractFloat,N}(sel::Vector{Int}, x::Array{T,N}, y::Array{T,N})
    @assert size(y) == size(x)
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
### Create a new variable instance

The call:

    vcreate(x)

creates a new variable instance similar to `x`.
"""
vcreate{T,N}(x::Array{T,N}) = Array(T, size(x))

"""
### Copy contents

The call:

    vcopy!(dst, src)

copies the contents of `src` into `dst` (which must have the same type and
size).  Nothing is done if `src` and `dst` are the same object.
"""
function vcopy!{T,N}(dst::Array{T,N}, src::Array{T,N})
    if !is(dst, src)
        @assert size(src) == size(dst)
        @inbounds begin
            @simd for i in 1:length(dst)
                dst[i] = src[i]
            end
        end
    end
end

"""
### Exchange contents

The call:

    vswap!(x, y)

exchanges the contents of `x` and `y` (which must have the same type and size).
Nothing is done if `src` and `dst` are the same object.
"""
function vswap!{T,N}(x::Array{T,N}, y::Array{T,N})
    if !is(x, y)
        @assert size(x) == size(y)
        @inbounds begin
            @simd for i in 1:length(x)
                temp = x[i]
                x[i] = y[i]
                y[i] = temp
            end
        end
    end
end

"""
### In-place scaling

The call:

    vscale!(x, alpha)

multiplies the contents of `x` by the scalar `alpha`.
"""
function vscale!{T<:AbstractFloat,N}(x::Array{T,N}, alpha::T)
    @inbounds begin
        if alpha == zero(T)
            @simd for i in 1:length(x)
                x[i] = alpha
            end
        elseif alpha == -one(T)
            @simd for i in 1:length(x)
                x[i] = -x[i]
            end
        elseif alpha != one(T)
            @simd for i in 1:length(x)
                x[i] *= alpha
            end
        end
    end
end

vscale!{T<:AbstractFloat,N}(x::Array{T,N}, alpha::Real) = vscale!(x, T(alpha))

"""
### Fill with a value

The call:

    vfill!(x, alpha)

sets all elements of `x` with the scalar value `alpha`.
"""
function vfill!{T<:AbstractFloat,N}(x::Array{T,N}, alpha::T)
    @inbounds begin
        @simd for i in 1:length(x)
            x[i] = alpha
        end
    end
end

vfill!{T<:AbstractFloat,N}(x::Array{T,N}, alpha::Real) = vfill!(x, T(alpha))

"""
### Increment an array by a scaled step

The call:

    vupdate!(dst, alpha, x)

increments the components of the destination *vector* `dst` by those of
`alpha*x`.  The code is optimized for some specific values of the multiplier
`alpha`.  For instance, if `alpha` is zero, then `dst` left unchanged without
using `x`.

Another possibility is:

    vupdate!(dst, sel, alpha, x)

with `sel` a selection of indices to which apply the operation.  Note that if
an indice is repeated, the operation will be performed several times at this
location.
"""
function vupdate!{T<:AbstractFloat,N}(dst::Array{T,N},
                                      a::T, x::Array{T,N})
    @assert size(dst) == size(x)
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

function vupdate!{T<:AbstractFloat,N}(dst::Array{T,N}, sel::Vector{Int},
                                      a::T, x::Array{T,N})
    @assert size(dst) == size(x)
    const n = length(dst)
    if a == one(T)
        @simd for i in 1:length(sel)
            j = sel[i]
            1 <= j <= n || throw(BoundsError())
            @inbounds dst[j] += x[j]
        end
    elseif a == -one(T)
        @simd for i in 1:length(sel)
            j = sel[i]
            1 <= j <= n || throw(BoundsError())
            @inbounds dst[j] -= x[j]
        end
    elseif a != zero(T)
        @simd for i in 1:length(sel)
            j = sel[i]
            1 <= j <= n || throw(BoundsError())
            @inbounds dst[j] += a*x[j]
        end
    end
end

function vupdate!{T<:AbstractFloat,N}(dst::Array{T,N},
                                      alpha::Real, x::Array{T,N})
    vupdate!(dst, T(alpha), x)
end

function vupdate!{T<:AbstractFloat,N}(dst::Array{T,N}, sel::Vector{Int},
                                      alpha::Real, x::Array{T,N})
    vupdate!(dst, sel, T(alpha), x)
end

"""
### Elementwise multiplication

The call:

    vproduct!(dst, x, y)

stores the elementwise multiplication of `x` by `y` in `dst`.

Another possibility is:

    vproduct!(dst, sel, x, y)

with `sel` a selection of indices to which apply the operation.
"""
function vproduct!{T<:AbstractFloat,N}(dst::Array{T,N},
                                       x::Array{T,N}, y::Array{T,N})
    @assert size(x) == size(dst)
    @assert size(y) == size(dst)
    @simd for i in 1:length(dst)
        @inbounds dst[i] = x[i]*y[i]
    end
end

function vproduct!{T<:AbstractFloat,N}(dst::Array{T,N}, sel::Vector{Int},
                                       x::Array{T,N}, y::Array{T,N})
    @assert size(x) == size(dst)
    @assert size(y) == size(dst)
    const n = length(dst)
    @simd for i in 1:length(sel)
        j = sel[i]
        1 <= j <= n || throw(BoundsError())
        @inbounds dst[j] = x[j]*y[j]
    end
end

"""
### Linear combination of arrays

The calls:

    vcombine!(dst, alpha, x)
    vcombine!(dst, alpha, x, beta, y)

stores the linear combinations `alpha*x` and `alpha*x + beta*y` into the
destination array `dst`.  The code is optimized for some specific values of the
coefficients `alpha` and `beta`.  For instance, if `alpha` (resp. `beta`) is
zero, then the contents of `x` (resp. `y`) is not used.

The source(s) and the destination can be the same.  For instance, the two
following lines of code produce the same result:

    vcombine!(dst, 1, dst, alpha, x)
    vupdate!(dst, alpha, x)

"""

function vcombine!{T<:AbstractFloat,N}(dst::Array{T,N}, a::T, x::Array{T,N})
    @assert size(x) == size(dst)
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

function vcombine!{T<:AbstractFloat,N}(dst::Array{T,N},
                                       a::T, x::Array{T,N},
                                       b::T, y::Array{T,N})
    @assert size(x) == size(dst)
    @assert size(y) == size(dst)
    const n = length(dst)
    @inbounds begin
        if a == zero(T)
            vcombine!(dst, b, y)
        elseif b == zero(T)
            vcombine!(dst, a, x)
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

function vcombine!{T<:AbstractFloat,N}(dst::Array{T,N},
                                       alpha::Real, x::Array{T,N})
    vcombine!(dst, T(alpha), x)
end

function vcombine!{T<:AbstractFloat,N}(dst::Array{T,N},
                                       alpha::Real, x::Array{T,N},
                                       beta::Real,  y::Array{T,N})
    vcombine!(dst, T(alpha), x, T(beta), y)
end

#-------------------------------------------------------------------------------
# PROJECTING VARIABLES

@inline clamp{T<:Real}(x::T, lo::T, hi::T) = max(lo, min(x, hi))
@inline clamp{T<:Real}(x::T, ::Void, hi::T) = min(x, hi)
@inline clamp{T<:Real}(x::T, lo::T, ::Void) = max(lo, x)

"""
    project_variables!(dst, lo, hi, x)

stores in `dst` the projection of the variables `x` in the box whose lower
bound is `lo` and upper bound is `hi`.

This is the same as `dst = clamp(x, lo, hi)` except that the result is
preallocated and that the operation is *much* faster (by a factor of 2-3).
"""
function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       lo::Real, hi::Real,
                                       x::Array{T,N})
    project_variables!(dst, T(lo), T(hi), x)
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       lo::T, hi::T,
                                       x::Array{T,N})
    @assert size(dst) == size(x)
    @assert lo ≤ hi # this also check for NaN
    const bounded_below = lo > T(-Inf)
    const bounded_above = hi < T(+Inf)
    if bounded_below && bounded_above
            @simd for i in 1:length(x)
                @inbounds dst[i] = clamp(x[i], lo, hi)
            end
    elseif bounded_below
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], lo, nothing)
        end
    elseif bounded_above
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], nothing, hi)
        end
    elseif !is(dst, x)
        vcopy!(dst, x)
    end
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       lo::Array{T,N}, hi::Real,
                                       x::Array{T,N})
    project_variables!(dst, lo, T(hi), x)
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       lo::Array{T,N}, hi::T,
                                       x::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(lo)  == size(x)
    const bounded_above = hi < T(+Inf)
    if bounded_above
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], lo[i], hi)
        end
    else
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], lo[i], nothing)
        end
    end
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       lo::Real, hi::Array{T,N},
                                       x::Array{T,N})
    project_variables!(dst, T(lo), hi, x)
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       lo::T, hi::Array{T,N},
                                       x::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(hi)  == size(x)
    const bounded_below = lo > T(-Inf)
    if bounded_below
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], lo, hi[i])
        end
    else
        @simd for i in 1:length(x)
            @inbounds dst[i] = clamp(x[i], nothing, hi[i])
        end
    end
end

function project_variables!{T<:Real,N}(dst::Array{T,N},
                                       lo::Array{T,N}, hi::Array{T,N},
                                       x::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(lo)  == size(x)
    @assert size(hi)  == size(x)
    @simd for i in 1:length(x)
        @inbounds dst[i] = clamp(x[i], lo[i], hi[i])
    end
end

#------------------------------------------------------------------------------
# PROJECTING DIRECTION

abstract Orientation
immutable Forward  <: Orientation; end
immutable Backward <: Orientation; end

convert{T<:Orientation}(::Type{T}, ::Union{Forward,Type{Forward}}) = Forward
convert{T<:Orientation}(::Type{T}, ::Union{Backward,Type{Backward}}) = Backward
convert{T<:Orientation}(::Type{T}, s::Real) = (s > 0 ? Forward :
                                               s < 0 ? Backward :
                                               error("invalid orientation"))
sign(::Union{Forward,Type{Forward}}) = +1
sign(::Union{Backward,Type{Backward}}) = -1

Orientation(x) = convert(Orientation, x)
orientation(T::DataType, x) = Orientation(x) == Forward ? +one(T) : -one(T)

@inline function project_forward{T<:Real}(x::T, d::T, lo::T, hi::T)
    (d > zero(T) ? x < hi : x > lo) ? d : zero(T)
end

@inline function project_backward{T<:Real}(x::T, d::T, lo::T, hi::T)
    (d < zero(T) ? x < hi : x > lo) ? d : zero(T)
end

@inline function project_forward{T<:Real}(x::T, d::T, ::Void, hi::T)
    (d < zero(T) || x < hi) ? d : zero(T)
end

@inline function project_backward{T<:Real}(x::T, d::T, ::Void, hi::T)
    (d > zero(T) || x < hi) ? d : zero(T)
end

@inline function project_forward{T<:Real}(x::T, d::T, lo::T, ::Void)
    (d > zero(T) || x > lo) ? d : zero(T)
end

@inline function project_backward{T<:Real}(x::T, d::T, lo::T, ::Void)
    (d < zero(T) || x > lo) ? d : zero(T)
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       lo::T, hi::T,
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(d)   == size(x)
    @assert lo ≤ hi # this also check for NaN
    const forward = Orientation(orient) == Forward
    const bounded_above = hi < T(+Inf)
    const bounded_below = lo > T(-Inf)
    @inbounds begin
        if bounded_below && bounded_above
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], lo, hi)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], lo, hi)
                end
            end
        elseif bounded_below
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], lo, nothing)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], lo, nothing)
                end
            end
        elseif bounded_above
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], nothing, hi)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], nothing, hi)
                end
            end
        elseif !is(dst, d)
            vcopy!(dst, d)
        end
    end
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       lo::Array{T,N}, hi::T,
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(lo)  == size(x)
    @assert size(d)   == size(x)
    const forward = Orientation(orient) == Forward
    const bounded_above = hi < T(+Inf)
    @inbounds begin
        if bounded_above
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], lo[i], hi)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], lo[i], hi)
                end
            end
        else
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], lo[i], nothing)
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], lo[i], nothing)
                end
            end
        end
    end
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       lo::T, hi::Array{T,N},
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(hi)  == size(x)
    @assert size(d)   == size(x)
    const forward = Orientation(orient) == Forward
    const bounded_below = lo > T(-Inf)
    @inbounds begin
        if bounded_below
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], lo, hi[i])
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], lo, hi[i])
                end
            end
        else
            if forward
                @simd for i in 1:length(x)
                    dst[i] = project_forward(x[i], d[i], nothing, hi[i])
                end
            else
                @simd for i in 1:length(x)
                    dst[i] = project_backward(x[i], d[i], nothing, hi[i])
                end
            end
        end
    end
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       lo::Array{T,N}, hi::Array{T,N},
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(dst) == size(x)
    @assert size(lo)  == size(x)
    @assert size(hi)  == size(x)
    @assert size(d)   == size(x)
    const forward = Orientation(orient) == Forward
    @inbounds begin
        if forward
            @simd for i in 1:length(x)
                dst[i] = project_forward(x[i], d[i], lo[i], hi[i])
            end
        else
            @simd for i in 1:length(x)
                dst[i] = project_backward(x[i], d[i], lo[i], hi[i])
            end
        end
    end
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       lo::Real, hi::Real,
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    project_direction!(dst, T(lo), T(hi), x, orient, d)
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       lo::Array{T,N}, hi::Real,
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    project_direction!(dst, lo, T(hi), x, orient, d)
end

function project_direction!{T<:Real,N}(dst::Array{T,N},
                                       lo::Real, hi::Array{T,N},
                                       x::Array{T,N},
                                       orient, d::Array{T,N})
    project_direction!(dst, T(lo), hi, x, orient, d)
end

function project_gradient!{T<:Real,N}(dst::Array{T,N},
                                      lo::Union{Real,Array{T,N}},
                                      hi::Union{Real,Array{T,N}},
                                      x::Array{T,N},
                                      d::Array{T,N})
    project_direction!(dst, lo, hi, x, -1, d)
end

#------------------------------------------------------------------------------
# COMPUTING STEP LIMITS

"""
### Compute step limits for line search

When there are separable bound constraints on the variables, the step `smin` to
the closest not yet reached bound and the step `smax` to the farthest bound are
computed by the call:
```
    (smin, smax) = step_limits(lo, hi, x, s, d)
```
where `lo` is the lower bound, `hi` is the upper bound, `x` are the current
variables and `sign(s)*d` is the search direction.

In orther words, `smin` is the smallest step which will bring at least one more
variable "out of bounds" and `smax` is the smallest step which will bring all
variables "out of bounds".  As a consequence, `0 < smin` and `0 ≤ smax`.
"""
function step_limits{T<:Real,N}(lo::T, hi::T, x::Array{T,N},
                                orient, d::Array{T,N})
    @assert size(d) == size(x)
    @assert lo ≤ hi # this also check for NaN
    const ZERO = zero(T)
    const INFINITY = T(Inf)
    const s = orientation(T, orient)
    const bounded_below = lo > -INFINITY
    const bounded_above = hi < +INFINITY
    smin = INFINITY
    @inbounds begin
        if bounded_below && bounded_above
            smax = ZERO
            @simd for i in 1:length(x)
                p = s*d[i]
                if p != ZERO
                    a = (p > ZERO ? hi - x[i] : lo - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        elseif bounded_below
            smax = ZERO
            @simd for i in 1:length(x)
                p = s*d[i]
                if p > ZERO
                    smax = INFINITY
                elseif p < ZERO
                    a = (lo - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        elseif bounded_above
            smax = ZERO
            @simd for i in 1:length(x)
                p = s*d[i]
                if p > ZERO
                    a = (hi - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                elseif p < ZERO
                    smax = INFINITY
                end
            end
        else
            smax = INFINITY
        end
    end
    return (smin, smax)
end

function step_limits{T<:Real,N}(lo::Array{T,N}, hi::T,
                                x::Array{T,N},
                                orient, d::Array{T,N})
    @assert size(lo) == size(x)
    @assert size(d)  == size(x)
    const ZERO = zero(T)
    const INFINITY = T(Inf)
    const s = orientation(T, orient)
    const bounded_above = hi < +INFINITY
    smin = INFINITY
    smax = ZERO
    @inbounds begin
        if bounded_above
            @simd for i in 1:length(x)
                p = s*d[i]
                if p != ZERO
                    a = (p > ZERO ? hi - x[i] : lo[i] - x[i])/p
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
                    smax = INFINITY
                elseif p < ZERO
                    a = (lo[i] - x[i])/p
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

function step_limits{T<:Real,N}(lo::T, hi::Array{T,N}, x::Array{T,N},
                                orient, d::Array{T,N})
    @assert size(hi) == size(x)
    @assert size(d)  == size(x)
    const ZERO = zero(T)
    const INFINITY = T(Inf)
    const s = orientation(T, orient)
    const bounded_below = lo > -INFINITY
    smin = INFINITY
    smax = ZERO
    @inbounds begin
        if bounded_below
            @simd for i in 1:length(x)
                p = s*d[i]
                if p != ZERO
                    a = (p > ZERO ? hi[i] - x[i] : lo - x[i])/p
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
                    a = (hi[i] - x[i])/p
                    if 0 < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                elseif p < ZERO
                    smax = INFINITY
                end
            end
        end
    end
    return (smin, smax)
end

function step_limits{T<:Real,N}(lo::Array{T,N}, hi::Array{T,N}, x::Array{T,N},
                                orient, d::Array{T,N})
    @assert size(lo) == size(x)
    @assert size(hi) == size(x)
    @assert size(d)  == size(x)
    const ZERO = zero(T)
    const s = orientation(T, orient)
    smin = T(Inf)
    smax = ZERO
    @inbounds begin
        @simd for i in 1:length(x)
            p = s*d[i]
            if p != ZERO
                # Step length to reach the upper/lower bound:
                a = (p > ZERO ? hi[i] - x[i] : lo[i] - x[i])/p
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

function step_limits{T<:Real,N}(lo::Real, hi::Real, x::Array{T,N},
                                orient, d::Array{T,N})
    step_limits(T(lo), T(hi), x, orient, d)
end

function step_limits{T<:Real,N}(lo::Array{T,N}, hi::Real, x::Array{T,N},
                                orient, d::Array{T,N})
    step_limits(lo, T(hi), x, orient, d)
end

function step_limits{T<:Real,N}(lo::Real, hi::Array{T,N}, x::Array{T,N},
                                orient, d::Array{T,N})
    step_limits(T(lo), hi, x, orient, d)
end

#------------------------------------------------------------------------------
# GETTING FREE VARIABLES

@inline function may_move_forward{T<:Real}(x::T, d::T, lo::T, ::Void)
    d > zero(T) || (d != zero(T) && x > lo)
end

@inline function may_move_backward{T<:Real}(x::T, d::T, lo::T, ::Void)
    d < zero(T) || (d != zero(T) && x > lo)
end

@inline function may_move_forward{T<:Real}(x::T, d::T, ::Void, hi::T)
    d < zero(T) || (d != zero(T) && x < hi)
end

@inline function may_move_backward{T<:Real}(x::T, d::T, ::Void, hi::T)
    d > zero(T) || (d != zero(T) && x < hi)
end

@inline function may_move_forward{T<:Real}(x::T, d::T, lo::T, hi::T)
    d != zero(T) && (d < zero(T) ? x > lo : x < hi)
end

@inline function may_move_backward{T<:Real}(x::T, d::T, lo::T, hi::T)
    d != zero(T) && (d > zero(T) ? x > lo : x < hi)
end

"""
## Get free variables when following a direction

    sel =  get_free_variables(lo, hi, x, orient, d)

yields the list of components of the variables `x` which are allowed to vary
along the search direction `sign(orient)*d` under box constraints with `lo` and
`hi` the lower and upper bounds.

If the projected gradient `p` of the objective function is available, the free
variables can be obtained by:

    sel =  get_free_variables(p)

where the projected gradient `p` has been computed as:

    project_direction!(p, lo, hi, x, -1, g)

with `g` the gradient of the objective function at `x`.

"""

function get_free_variables{T<:Real,N}(p::Array{T,N})
    const ZERO = zero(T)
    const n = length(p)
    sel = Array(Int, n)
    j = 0
    @inbounds begin
        @simd for i in 1:n
            if p[i] != ZERO
                j += 1
                sel[j] = i
            end
        end
    end
    return (j == n ? sel : (j > 0 ? sel[1:j] : Array(Int, 0)))
end

function get_free_variables{T<:Real,N}(lo::T, hi::T, x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(d)  == size(x)
    @assert lo ≤ hi # this also check for NaN
    const forward = Orientation(orient) == Forward
    const bounded_below = lo > T(-Inf)
    const bounded_above = hi < T(+Inf)
    const n = length(x)
    sel = Array(Int, n)
    j = 0
    @inbounds begin
        if bounded_below && bounded_above
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], lo, hi)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], lo, hi)
                        j += 1
                        sel[j] = i
                    end
                end
            end
        elseif bounded_below
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], lo, nothing)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], lo, nothing)
                        j += 1
                        sel[j] = i
                    end
                end
            end
        elseif bounded_above
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], nothing, hi)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], nothing, hi)
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

function get_free_variables{T<:Real,N}(lo::Array{T,N}, hi::T, x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(d)  == size(x)
    @assert size(lo) == size(x)
    const forward = Orientation(orient) == Forward
    const bounded_above = hi < T(+Inf)
    const n = length(x)
    sel = Array(Int, n)
    j = 0
    @inbounds begin
        if bounded_above
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], lo[i], hi)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], lo[i], hi)
                        j += 1
                        sel[j] = i
                    end
                end
            end
        else
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], lo[i], nothing)
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], lo[i], nothing)
                        j += 1
                        sel[j] = i
                    end
                end
            end
        end
    end
    return (j == n ? sel : (j > 0 ? sel[1:j] : Array(Int, 0)))
end

function get_free_variables{T<:Real,N}(lo::T, hi::Array{T,N}, x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(d)  == size(x)
    @assert size(hi) == size(x)
    const forward = Orientation(orient) == Forward
    const bounded_below = lo > T(-Inf)
    const n = length(x)
    sel = Array(Int, n)
    j = 0
    @inbounds begin
        if bounded_below
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], lo, hi[i])
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], lo, hi[i])
                        j += 1
                        sel[j] = i
                    end
                end
            end
        else
            if forward
                @simd for i in 1:n
                    if may_move_forward(x[i], d[i], nothing, hi[i])
                        j += 1
                        sel[j] = i
                    end
                end
            else
                @simd for i in 1:n
                    if may_move_backward(x[i], d[i], nothing, hi[i])
                        j += 1
                        sel[j] = i
                    end
                end
            end
        end
    end
    return (j == n ? sel : (j > 0 ? sel[1:j] : Array(Int, 0)))
end

function get_free_variables{T<:Real,N}(lo::Array{T,N}, hi::Array{T,N}, x::Array{T,N},
                                       orient, d::Array{T,N})
    @assert size(d)  == size(x)
    @assert size(lo) == size(x)
    @assert size(hi) == size(x)
    const forward = Orientation(orient) == Forward
    const n = length(x)
    sel = Array(Int, n)
    j = 0
    @inbounds begin
        if forward
            @simd for i in 1:n
                if may_move_forward(x[i], d[i], lo[i], hi[i])
                    j += 1
                    sel[j] = i
                end
            end
        else
            @simd for i in 1:n
                if may_move_backward(x[i], d[i], lo[i], hi[i])
                    j += 1
                    sel[j] = i
                end
            end
        end
    end
    return (j == n ? sel : (j > 0 ? sel[1:j] : Array(Int, 0)))
end

function get_free_variables{T<:Real,N}(lo::Real, hi::Real, x::Array{T,N},
                                       orient, d::Array{T,N})
    get_free_variables(T(lo), T(hi), x, orient, d)
end

function get_free_variables{T<:Real,N}(lo::Array{T,N}, hi::Real, x::Array{T,N},
                                       orient, d::Array{T,N})
    get_free_variables(lo, T(hi), x, orient, d)
end

function get_free_variables{T<:Real,N}(lo::Real, hi::Array{T,N}, x::Array{T,N},
                                       orient, d::Array{T,N})
    get_free_variables(T(lo), hi, x, orient, d)
end

#------------------------------------------------------------------------------
