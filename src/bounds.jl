#
# bounds.jl --
#
# Implement operations involving simple bound constraints.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2018, Éric Thiébaut.
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module SimpleBounds

export
    clip,
    get_free_variables!,
    get_free_variables,
    project_direction!,
    project_variables!,
    step_limits

"""
```julia
clip(x, lo, hi)
```

yields `x` subject to simple bound constraints.  That is, returns `x` if `lo ≤
x ≤ hi`, `lo` if 'x < lo' and `hi` if `x > hi`.  This method is similar to
`clamp` except that bounds may be `nothing` to indicate that there is no limit
for the corresponding bound.

See also [`clamp`](@ref).

"""
@inline clip(x::T, lo::T,  hi::T) where {T<:Real} = max(lo, min(x, hi))
@inline clip(x::T, ::Void, hi::T) where {T<:Real} = min(x, hi)
@inline clip(x::T, lo::T, ::Void) where {T<:Real} = max(lo, x)

#-------------------------------------------------------------------------------
# PROJECTING VARIABLES

"""
```julia
project_variables!(dst, src, lo, hi) -> dst
```

overwrites destination `dst` the projection of the source variables `src` in the
box whose lower bound is `lo` and upper bound is `hi`.  The destination `dst`
is returned.

This is the same as `dst = clip(src, lo, hi)` except that the result is
preallocated and that the operation is *much* faster (by a factor of 2-3).

"""
function project_variables!(dst::DenseArray{T,N},
                            src::DenseArray{T,N},
                            lo::Real,
                            hi::Real) where {T<:AbstractFloat,N}
    project_variables!(dst, src, T(lo), T(hi))
end

function project_variables!(dst::DenseArray{T,N},
                            src::DenseArray{T,N},
                            lo::T,
                            hi::T) where {T<:AbstractFloat,N}
    @assert size(dst) == size(src)
    @assert lo ≤ hi # this also check for NaN
    const bounded_below = (lo > T(-Inf))
    const bounded_above = (hi < T(+Inf))
    if bounded_below && bounded_above
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = clip(src[i], lo, hi)
        end
    elseif bounded_below
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = clip(src[i], lo, nothing)
        end
    elseif bounded_above
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = clip(src[i], nothing, hi)
        end
    elseif !is(dst, src)
        vcopy!(dst, src)
    end
    return dst
end

function project_variables!(dst::DenseArray{T,N},
                            src::DenseArray{T,N},
                            lo::DenseArray{T,N},
                            hi::Real) where {T<:AbstractFloat,N}
    project_variables!(dst, src, lo, T(hi))
end

function project_variables!(dst::DenseArray{T,N},
                            src::DenseArray{T,N},
                            lo::DenseArray{T,N},
                            hi::T) where {T<:AbstractFloat,N}
    @assert size(dst) == size(src) == size(lo)
    if hi < T(+Inf)
        @inbounds @simd for i in eachindex(dst, src, lo)
            dst[i] = clip(src[i], lo[i], hi)
        end
    else
        @inbounds @simd for i in eachindex(dst, src, lo)
            dst[i] = clip(src[i], lo[i], nothing)
        end
    end
    return dst
end

function project_variables!(dst::DenseArray{T,N},
                            src::DenseArray{T,N},
                            lo::Real,
                            hi::DenseArray{T,N}) where {T<:AbstractFloat,N}
    project_variables!(dst, src, T(lo), hi)
end

function project_variables!(dst::DenseArray{T,N},
                            src::DenseArray{T,N},
                            lo::T,
                            hi::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(dst) == size(src) == size(hi)
    if lo > T(-Inf)
        @inbounds @simd for i in eachindex(dst, src, hi)
            dst[i] = clip(src[i], lo, hi[i])
        end
    else
        @inbounds @simd for i in eachindex(dst, src, hi)
            dst[i] = clip(src[i], nothing, hi[i])
        end
    end
    return dst
end

function project_variables!(dst::DenseArray{T,N},
                            src::DenseArray{T,N},
                            lo::DenseArray{T,N},
                            hi::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(dst) == size(src) == size(lo) == size(hi)
    @inbounds  @simd for i in eachindex(dst, src, lo, hi)
        dst[i] = clip(src[i], lo[i], hi[i])
    end
    return dst
end

#------------------------------------------------------------------------------
# PROJECTING DIRECTION

# Orientation is indicated by a singleton.
abstract type Orientation end
struct Forward  <: Orientation; end
struct Backward <: Orientation; end

const FORWARD = Forward()
const BACKWARD = Backward()

Base.convert(::Type{Orientation}, ::Union{Forward,Type{Forward}}) = FORWARD
Base.convert(::Type{Orientation}, ::Union{Backward,Type{Backward}}) = BACKWARD
Base.convert(::Type{Orientation}, s::Real) = (s > 0 ? FORWARD :
                                              s < 0 ? BACKWARD :
                                              error("invalid orientation"))
Base.sign(::Union{Forward,Type{Forward}}) = +1
Base.sign(::Union{Backward,Type{Backward}}) = -1

Orientation(x) = convert(Orientation, x)
orientation(T::DataType, x) = Orientation(x) === FORWARD ? +one(T) : -one(T)

@inline projdir(x::T, lo::T, hi::T, ::Forward, d::T) where {T<:AbstractFloat} =
    (d > zero(T) ? x < hi : x > lo) ? d : zero(T)

@inline projdir(x::T, lo::T, hi::T, ::Backward, d::T) where {T<:AbstractFloat} =
    (d < zero(T) ? x < hi : x > lo) ? d : zero(T)

@inline projdir(x::T, ::Void, hi::T, ::Forward, d::T) where {T<:AbstractFloat} =
    (d < zero(T) || x < hi) ? d : zero(T)

@inline projdir(x::T, ::Void, hi::T, ::Backward, d::T) where {T<:AbstractFloat} =
    (d > zero(T) || x < hi) ? d : zero(T)

@inline projdir(x::T, lo::T, ::Void, ::Forward, d::T) where {T<:AbstractFloat} =
    (d > zero(T) || x > lo) ? d : zero(T)

@inline projdir(x::T, lo::T, ::Void, ::Backward, d::T) where {T<:AbstractFloat} =
    (d < zero(T) || x > lo) ? d : zero(T)

function project_direction!(dst::DenseArray{T,N},
                            x::DenseArray{T,N},
                            lo::T,
                            hi::T,
                            o::Orientation,
                            d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(dst) == size(x) == size(d)
    @assert lo ≤ hi # this also check for NaN
    const bounded_above = (hi < T(+Inf))
    const bounded_below = (lo > T(-Inf))
    if bounded_below && bounded_above
        @inbounds @simd for i in eachindex(dst, x, d)
            dst[i] = projdir(x[i], lo, hi, o, d[i])
        end
    elseif bounded_below
        @inbounds @simd for i in eachindex(dst, x, d)
            dst[i] = projdir(x[i], lo, nothing, o, d[i])
        end
    elseif bounded_above
        @inbounds @simd for i in eachindex(dst, x, d)
            dst[i] = projdir(x[i], nothing, hi, o, d[i])
        end
    elseif !is(dst, d)
        vcopy!(dst, d)
    end
    return dst
end

function project_direction!(dst::DenseArray{T,N},
                            x::DenseArray{T,N},
                            lo::DenseArray{T,N},
                            hi::T,
                            o::Orientation,
                            d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(dst) == size(x) == size(d) == size(lo)
    if hi < T(+Inf)
        @inbounds @simd for i in eachindex(dst, x, d, lo)
            dst[i] = projdir(x[i], lo[i], hi, o, d[i])
        end
    else
        @inbounds @simd for i in eachindex(dst, x, d, lo)
            dst[i] = projdir(x[i], lo[i], nothing, o, d[i])
        end
    end
    return dst
end

function project_direction!(dst::DenseArray{T,N},
                            x::DenseArray{T,N},
                            lo::T,
                            hi::DenseArray{T,N},
                            o::Orientation,
                            d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(dst) == size(x) == size(d) == size(hi)
    if lo > T(-Inf)
        @inbounds @simd for i in eachindex(dst, x, d, hi)
            dst[i] = projdir(x[i], lo, hi[i], o, d[i])
        end
    else
        @inbounds @simd for i in eachindex(dst, x, d, hi)
            dst[i] = projdir(x[i], nothing, hi[i], o, d[i])
        end
    end
    return dst
end

function project_direction!(dst::DenseArray{T,N},
                            x::DenseArray{T,N},
                            lo::DenseArray{T,N},
                            hi::DenseArray{T,N},
                            o::Orientation,
                            d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(dst) == size(x) == size(d) == size(lo) == size(hi)
    @inbounds @simd for i in eachindex(dst, x, d, lo, hi)
        dst[i] = projdir(x[i], lo[i], hi[i], o, d[i])
    end
    return dst
end

function project_direction!(dst::DenseArray{T,N},
                            x::DenseArray{T,N},
                            lo::Real,
                            hi::Real,
                            orient,
                            d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    project_direction!(dst, x, T(lo), T(hi), Orientation(orient), d)
end

function project_direction!(dst::DenseArray{T,N},
                            x::DenseArray{T,N},
                            lo::DenseArray{T,N},
                            hi::Real,
                            orient,
                            d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    project_direction!(dst, x, lo, T(hi), Orientation(orient), d)
end

function project_direction!(dst::DenseArray{T,N},
                            x::DenseArray{T,N},
                            lo::Real,
                            hi::DenseArray{T,N},
                            orient,
                            d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    project_direction!(dst, x, T(lo), hi, Orientation(orient), d)
end

function project_gradient!(dst::DenseArray{T,N},
                           x::DenseArray{T,N},
                           lo::Union{Real,DenseArray{T,N}},
                           hi::Union{Real,DenseArray{T,N}},
                           d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    project_direction!(dst, x, lo, hi, BACKWARD, d)
end

#------------------------------------------------------------------------------
# COMPUTING STEP LIMITS

"""
### Compute step limits for line search

When there are separable bound constraints on the variables, the step `smin` to
the closest not yet reached bound and the step `smax` to the farthest bound are
computed by the call:
```
    (smin, smax) = step_limits(x, lo, hi, s, d)
```
where `lo` is the lower bound, `hi` is the upper bound, `x` are the current
variables and `sign(s)*d` is the search direction.

In orther words, `smin` is the smallest step which will bring at least one more
variable "out of bounds" and `smax` is the smallest step which will bring all
variables "out of bounds".  As a consequence, `0 < smin` and `0 ≤ smax`.
"""
function step_limits(x::AbstractArray{T,N},
                     lo::T,
                     hi::T,
                     orient,
                     d::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(x) == size(d)
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
            @simd for i in eachindex(x, d)
                p = s*d[i]
                if p != ZERO
                    a = (p > ZERO ? hi - x[i] : lo - x[i])/p
                    if ZERO < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        elseif bounded_below
            smax = ZERO
            @simd for i in eachindex(x, d)
                p = s*d[i]
                if p > ZERO
                    smax = INFINITY
                elseif p < ZERO
                    a = (lo - x[i])/p
                    if ZERO < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        elseif bounded_above
            smax = ZERO
            @simd for i in eachindex(x, d)
                p = s*d[i]
                if p > ZERO
                    a = (hi - x[i])/p
                    if ZERO < a < smin
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

function step_limits(x::AbstractArray{T,N},
                     lo::AbstractArray{T,N},
                     hi::T,
                     orient,
                     d::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(x) == size(lo) == size(d)
    const ZERO = zero(T)
    const INFINITY = T(Inf)
    const s = orientation(T, orient)
    const bounded_above = hi < +INFINITY
    smin = INFINITY
    smax = ZERO
    @inbounds begin
        if bounded_above
            @simd for i in eachindex(x, lo, d)
                p = s*d[i]
                if p != ZERO
                    a = (p > ZERO ? hi - x[i] : lo[i] - x[i])/p
                    if ZERO < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        else
            @simd for i in eachindex(x, lo, d)
                p = s*d[i]
                if p > ZERO
                    smax = INFINITY
                elseif p < ZERO
                    a = (lo[i] - x[i])/p
                    if ZERO < a < smin
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

function step_limits(x::AbstractArray{T,N},
                     lo::T,
                     hi::AbstractArray{T,N},
                     orient,
                     d::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(x) == size(hi) == size(d)
    const ZERO = zero(T)
    const INFINITY = T(Inf)
    const s = orientation(T, orient)
    const bounded_below = lo > -INFINITY
    smin = INFINITY
    smax = ZERO
    @inbounds begin
        if bounded_below
            @simd for i in eachindex(x, hi, d)
                p = s*d[i]
                if p != ZERO
                    a = (p > ZERO ? hi[i] - x[i] : lo - x[i])/p
                    if ZERO < a < smin
                        smin = a
                    end
                    if a > smax
                        smax = a
                    end
                end
            end
        else
            @simd for i in eachindex(x, hi, d)
                p = s*d[i]
                if p > ZERO
                    a = (hi[i] - x[i])/p
                    if ZERO < a < smin
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

function step_limits(x::AbstractArray{T,N},
                     lo::AbstractArray{T,N},
                     hi::AbstractArray{T,N},
                     orient,
                     d::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(x) == size(lo) == size(hi) == size(d)
    const ZERO = zero(T)
    const s = orientation(T, orient)
    smin = T(Inf)
    smax = ZERO
    @inbounds begin
        @simd for i in eachindex(x, lo, hi, d)
            p = s*d[i]
            if p != ZERO
                # Step length to reach the upper/lower bound:
                a = (p > ZERO ? hi[i] - x[i] : lo[i] - x[i])/p
                if ZERO < a < smin
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

function step_limits(x::AbstractArray{T,N},
                     lo::Real,
                     hi::Real,
                     orient,
                     d::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    step_limits(x, T(lo), T(hi), orient, d)
end

function step_limits(x::AbstractArray{T,N},
                     lo::AbstractArray{T,N},
                     hi::Real,
                     orient,
                     d::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    step_limits(x, lo, T(hi), orient, d)
end

function step_limits(x::AbstractArray{T,N},
                     lo::Real,
                     hi::AbstractArray{T,N},
                     orient,
                     d::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    step_limits(x, T(lo), hi, orient, d)
end

#------------------------------------------------------------------------------
# GETTING FREE VARIABLES

@inline may_move(x::T, lo::T, ::Void, ::Forward, d::T) where {T<:AbstractFloat} =
    d > zero(T) || (d != zero(T) && x > lo)

@inline may_move(x::T, lo::T, ::Void, ::Backward, d::T) where {T<:AbstractFloat} =
    d < zero(T) || (d != zero(T) && x > lo)

@inline may_move(x::T, ::Void, hi::T, ::Forward, d::T) where {T<:AbstractFloat} =
    d < zero(T) || (d != zero(T) && x < hi)

@inline may_move(x::T, ::Void, hi::T, ::Backward, d::T) where {T<:AbstractFloat} =
    d > zero(T) || (d != zero(T) && x < hi)

@inline may_move(x::T, lo::T, hi::T, ::Forward, d::T) where {T<:AbstractFloat} =
    d != zero(T) && (d < zero(T) ? x > lo : x < hi)

@inline may_move(x::T, lo::T, hi::T, ::Backward, d::T) where {T<:AbstractFloat} =
    d != zero(T) && (d > zero(T) ? x > lo : x < hi)

"""
## Get free variables when following a direction

    sel = get_free_variables(x, lo, hi, orient, d)

yields the list of components of the variables `x` which are allowed to vary
along the search direction `sign(orient)*d` under box constraints with `lo` and
`hi` the lower and upper bounds.

If the projected gradient `gp` of the objective function is available, the free
variables can be obtained by:

    sel = get_free_variables(gp)

where the projected gradient `gp` has been computed as:

    project_direction!(gp, x, lo, hi, -1, g)

with `g` the gradient of the objective function at `x`.

There are in-place versions:

    get_free_variables!(sel, x, lo, hi, orient, d) -> sel
    get_free_variables!(sel, gp)

"""
function get_free_variables!(sel::Vector{Int},
                             gp::DenseArray{T,N}) where {T<:AbstractFloat,N}
    const ZERO = zero(T)
    const n = length(gp)
    resize!(sel, n)
    j = 0
    @inbounds @simd for i in 1:n
        if gp[i] != ZERO
            j += 1
            sel[j] = i
        end
    end
    return (j == n ? sel : resize!(sel, j))
end

function get_free_variables!(sel::Vector{Int},
                             x::DenseArray{T,N},
                             lo::T,
                             hi::T,
                             o::Orientation,
                             d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(x) == size(d)
    @assert lo ≤ hi # this also check for NaN
    const bounded_below = (lo > T(-Inf))
    const bounded_above = (hi < T(+Inf))
    const n = length(x)
    resize!(sel, n)
    j = 0
    if bounded_below && bounded_above
        @inbounds @simd for i in 1:n
            if may_move(x[i], lo, hi, o, d[i])
                j += 1
                sel[j] = i
            end
        end
    elseif bounded_below
        @inbounds @simd for i in 1:n
            if may_move(x[i], lo, nothing, o, d[i])
                j += 1
                sel[j] = i
            end
        end
    elseif bounded_above
        @inbounds @simd for i in 1:n
            if may_move(x[i], nothing, hi, o, d[i])
                j += 1
                sel[j] = i
            end
        end
    else
        @inbounds @simd for i in 1:n
            sel[i] = i
        end
        j = n
    end
    return (j == n ? sel : resize!(sel, j))
end

function get_free_variables!(sel::Vector{Int},
                             x::DenseArray{T,N},
                             lo::DenseArray{T,N},
                             hi::T,
                             o::Orientation,
                             d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(x) == size(d) == size(lo)
    const n = length(x)
    resize!(sel, n)
    j = 0
    if hi < T(+Inf)
        @inbounds @simd for i in 1:n
            if may_move(x[i], lo[i], hi, o, d[i])
                j += 1
                sel[j] = i
            end
        end
    else
        @inbounds @simd for i in 1:n
            if may_move(x[i], lo[i], nothing, o, d[i])
                j += 1
                sel[j] = i
            end
        end
    end
    return (j == n ? sel : resize!(sel, j))
end

function get_free_variables!(sel::Vector{Int},
                             x::DenseArray{T,N},
                             lo::T,
                             hi::DenseArray{T,N},
                             o::Orientation,
                             d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(x) == size(d) == size(hi)
    const n = length(x)
    resize!(sel, n)
    j = 0
    if lo > T(-Inf)
        @inbounds @simd for i in 1:n
            if may_move(x[i], lo, hi[i], o, d[i])
                j += 1
                    sel[j] = i
            end
        end
    else
        @inbounds @simd for i in 1:n
            if may_move(x[i], nothing, hi[i], o, d[i])
                j += 1
                sel[j] = i
            end
        end
    end
    return (j == n ? sel : resize!(sel, j))
end

function get_free_variables!(sel::Vector{Int},
                             x::DenseArray{T,N},
                             lo::DenseArray{T,N},
                             hi::DenseArray{T,N},
                             o::Orientation,
                             d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    @assert size(x) == size(d) == size(lo) == size(hi)
    const n = length(x)
    resize!(sel, n)
    j = 0
    @inbounds @simd for i in 1:n
        if may_move(x[i], lo[i], hi[i], o, d[i])
            j += 1
            sel[j] = i
        end
    end
    return (j == n ? sel : resize!(sel, j))
end

function get_free_variables!(sel::Vector{Int},
                             x::DenseArray{T,N},
                             lo::Real,
                             hi::Real,
                             orient,
                             d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    get_free_variables!(sel, x, T(lo), T(hi), Orientation(orient), d)
end

function get_free_variables!(sel::Vector{Int},
                             x::DenseArray{T,N},
                             lo::DenseArray{T,N},
                             hi::Real,
                             orient,
                             d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    get_free_variables!(sel, x, lo, T(hi), Orientation(orient), d)
end

function get_free_variables!(sel::Vector{Int},
                             x::DenseArray{T,N},
                             lo::Real,
                             hi::DenseArray{T,N},
                             orient,
                             d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    get_free_variables!(sel, x, T(lo), hi, Orientation(orient), d)
end

function get_free_variables(gp::DenseArray{T,N}) where {T<:AbstractFloat,N}
    get_free_variables!(newvariablelengthvector(Int, length(gp)), gp)
end

function get_free_variables(x::DenseArray{T,N},
                            lo::Union{Real,DenseArray{T,N}},
                            hi::Union{Real,DenseArray{T,N}},
                            orient,
                            d::DenseArray{T,N}) where {T<:AbstractFloat,N}
    get_free_variables!(newvariablelengthvector(Int, length(x)),
                        x, lo, hi, orient, d)
end

@doc @doc(get_free_variables!) get_free_variables

function newvariablelengthvector(::Type{T}, n::Integer) where {T}
    vec = Array{T}(n)
    sizehint!(vec, n)
    vec
end

#------------------------------------------------------------------------------

end # module
