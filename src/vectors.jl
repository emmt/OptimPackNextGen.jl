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
# Copyright (C) 2015-2017, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

"""
    vnorm2(v)

yields the Euclidean (L2) norm of `v`.  Also see `vnorm1` and `vnorminf`.

"""
function vnorm2{T<:AbstractFloat,N}(v::AbstractArray{T,N}) :: Float
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += v[i]*v[i]
    end
    return Float(sqrt(s))
end

vnorm2{V}(x::V) = Float(sqrt(vdot(x, y)))

"""
    vnorm1(v)

yields the L1 norm of `v`, that is the sum of the absolute values of its
elements.  Also see `vnorm2` and `vnorminf`.

"""
function vnorm1{T<:AbstractFloat,N}(v::AbstractArray{T,N}) :: Float
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += abs(v[i])
    end
    return Float(s)
end

"""
    vnorminf(v)

yields the infinite norm of `v`, that is the maximum absolute value of its
elements.  Also see `vnorm2` and `vnorm1`.

"""
function vnorminf{T<:AbstractFloat,N}(v::AbstractArray{T,N}) :: Float
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s = max(s, abs(v[i]))
    end
    return Float(s)
end

#------------------------------------------------------------------------------

"""
### Inner product

The call:

    vdot(x, y)

yields the inner product (a.k.a. scalar or dot product) between `x` and `y`
(which must have the same size).  The triple inner product between `w`, `x` and
`y` can be computed by:

    vdot(w, x, y)

Finally:

    vdot(sel, x, y)

computes the sum of the product of the elements of `x` and `y` whose indices
are given by the `sel` argument.

If the arguments are complex, they are considered as vectors of pairs of reals
and the result is:

    vdot(x, y) = x[1].re*y[1].re + x[1].im*y[1].im +
                 x[2].re*y[2].re + x[2].im*y[2].im + ...

which is the real part of the usual definition.

"""
function vdot{T<:AbstractFloat,N}(x::AbstractArray{T,N},
                                  y::AbstractArray{T,N}) :: Float
    @assert size(x) == size(y)
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        @inbounds s += x[i]*y[i]
    end
    return Float(s)
end

# FIXME: use v0.6 feature to enforce arguments to be the same type of arrays.

function vdot{T<:AbstractFloat,N}(w::AbstractArray{T,N},
                                  x::AbstractArray{T,N},
                                  y::AbstractArray{T,N}) :: Float
    @assert size(w) == size(x)== size(y)
    local s::T = 0
    @inbounds @simd for i in eachindex(w, x, y)
        s += w[i]*x[i]*y[i]
    end
    return Float(s)
end

function vdot{T<:AbstractFloat,N}(sel::AbstractVector{Int},
                                  x::DenseArray{T,N},
                                  y::DenseArray{T,N}) :: Float
    @assert size(y) == size(x)
    local s::T = zero(T)
    const n = length(x)
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
        1 ≤ j ≤ n || throw(BoundsError())
        s += x[j]*y[j]
    end
    return Float(s)
end

function vdot{T<:AbstractFloat,N}(x::AbstractArray{Complex{T},N},
                                  y::AbstractArray{Complex{T},N}) :: Float
    @assert size(x) == size(y)
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        s += x[i].re*y[i].re + x[i].im*y[i].im
    end
    return Float(s)
end

#------------------------------------------------------------------------------

"""
### Create a new variable instance

The call:

    vcreate(x)

creates a new variable instance similar to `x`.
"""
vcreate{T<:DenseArray}(x::T) = T(size(x))

#= FIXME: force element type to be derived from `AbstractFloat` and (maybe) use
`similar` =#

#------------------------------------------------------------------------------

"""
### Copy contents

The call:

    vcopy!(dst, src) -> dst

copies the contents of `src` into `dst` (which must have the same type and
size) and returns `dst`.  Nothing is done if `src` and `dst` are the same
object.  To create a fresh copy of `src`, do:

    dst = vcopy(src)

"""
function vcopy!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                    src::DenseArray{T,N})
    if pointer(dst) != pointer(src)
        @assert size(src) == size(dst)
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = src[i]
        end
    end
    return dst
end

vcopy(src) = vcopy!(vcreate(src), src)

@doc @doc(vcopy) vcopy!

#------------------------------------------------------------------------------

"""
### Exchange contents

The call:

    vswap!(x, y)

exchanges the contents of `x` and `y` (which must have the same type and size).
Nothing is done if `src` and `dst` are the same object.
"""
function vswap!{T,N}(x::DenseArray{T,N}, y::DenseArray{T,N})
    if pointer(x) != pointer(y)
        @assert size(x) == size(y)
        @inbounds @simd for i in eachindex(x, y)
            temp = x[i]
            x[i] = y[i]
            y[i] = temp
        end
    end
end

#------------------------------------------------------------------------------

"""
### Scaling

    vscale(alpha, src) -> dst

yields a new *vector* `dst` whose components are the corresponding components
of `src` multiplied by the scalar `alpha`.  Alternatively:

    vscale!(dst, alpha) -> dst

performs in-place scaling of `dst` and

    vscale!(dst, alpha, src) -> dst

stores in `dst` the result of scaling `src` by `alpha`.  The two latter methods
return argument `dst`.

"""
vscale{T<:Real}(alpha::T, x) =
    alpha == one(T) ? vcopy(x) : vscale!(vcreate(x), alpha, x)

vscale!{T<:AbstractFloat,N}(dst::DenseArray{T,N}, alpha::Real) =
    vscale!(dst, T(alpha))

function vscale!{T<:AbstractFloat,N}(dst::DenseArray{T,N}, alpha::T)
    if alpha == zero(T)
        @inbounds @simd for i in eachindex(dst)
            dst[i] = alpha
        end
    elseif alpha == -one(T)
        @inbounds @simd for i in eachindex(dst)
            dst[i] = -dst[i]
        end
    elseif alpha != one(T)
        @inbounds @simd for i in eachindex(dst)
            dst[i] *= alpha
        end
    end
    return dst
end

function vscale!{T<:AbstractFloat,N}(dst::DenseArray{T,N}, alpha::Real,
                                     src::DenseArray{T,N})
    vscale!(dst, T(alpha), src)
end

function vscale!{T<:AbstractFloat,N}(dst::DenseArray{T,N}, alpha::T,
                                     src::DenseArray{T,N})
    @assert size(src) == size(dst)
    if alpha == zero(T)
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = alpha
        end
    elseif alpha == -one(T)
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = -src[i]
        end
    elseif alpha == one(T)
        if pointer(dst) != pointer(src)
            @inbounds @simd for i in eachindex(dst, src)
                dst[i] = src[i]
            end
        end
    else
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = alpha*src[i]
        end
    end
    return dst
end

vscale!{V}(dst::V, alpha::Real, src::V) = vcombine!(dst, alpha, src, 0, src)

vscale!(dst, alpha::Real) = vscale!(dst, alpha, dst)

@doc @doc(vscale) vscale!

#------------------------------------------------------------------------------

"""
### Fill with a value

The call:

    vfill!(x, alpha) -> x

sets all elements of `x` with the scalar value `alpha` and return `x`.
"""
function vfill!{T<:AbstractFloat,N}(x::DenseArray{T,N}, alpha::T)
    @inbounds @simd for i in eachindex(x)
        x[i] = alpha
    end
    return x
end

vfill!{T<:AbstractFloat,N}(x::DenseArray{T,N}, alpha::Real) =
    vfill!(x, T(alpha))

#------------------------------------------------------------------------------

"""
### Increment an array by a scaled step

The call:

    vupdate!(dst, alpha, x) -> dst

increments the components of the destination *vector* `dst` by those of
`alpha*x` and returns `dst`.  The code is optimized for some specific values of
the multiplier `alpha`.  For instance, if `alpha` is zero, then `dst` left
unchanged without using `x`.

Another possibility is:

    vupdate!(dst, sel, alpha, x) -> dst

with `sel` a selection of indices to which apply the operation.  Note that if
an indice is repeated, the operation will be performed several times at this
location.

"""
function vupdate!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                      alpha::T,
                                      x::DenseArray{T,N})
    @assert size(dst) == size(x)
    if alpha == one(T)
        @inbounds @simd for i in eachindex(dst, x)
            dst[i] += x[i]
        end
    elseif alpha == -one(T)
        @inbounds @simd for i in eachindex(dst, x)
            dst[i] -= x[i]
        end
    elseif alpha != zero(T)
        @inbounds @simd for i in eachindex(dst, x)
            dst[i] += alpha*x[i]
        end
    end
    return dst
end

function vupdate!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                      alpha::Real,
                                      x::DenseArray{T,N})
    vupdate!(dst, T(alpha), x)
end

function vupdate!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                      sel::AbstractVector{Int},
                                      alpha::T,
                                      x::DenseArray{T,N})
    @assert size(dst) == size(x)
    const n = length(dst)
    if alpha == one(T)
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            1 ≤ j ≤ n || throw(BoundsError())
            dst[j] += x[j]
        end
    elseif alpha == -one(T)
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            1 ≤ j ≤ n || throw(BoundsError())
            dst[j] -= x[j]
        end
    elseif alpha != zero(T)
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            1 ≤ j ≤ n || throw(BoundsError())
            dst[j] += alpha*x[j]
        end
    end
    return dst
end

function vupdate!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                      sel::AbstractVector{Int},
                                      alpha::Real,
                                      x::DenseArray{T,N})
    vupdate!(dst, sel, T(alpha), x)
end

#------------------------------------------------------------------------------

"""
### Elementwise multiplication

    vproduct(x, y) -> dst

yields the elementwise multiplication of `x` by `y`.  To avoid allocating the
result, the destination array `dst` can be specified with the in-place version
of the method:

    vproduct!(dst, x, y) -> dst

Another possibility is:

    vproduct!(dst, sel, x, y) -> dst

with `sel` a selection of indices to which apply the operation.

"""
vproduct{V}(x::V, y::V) = vproduct!(vcreate(x), x, y)

vproduct!{V}(dst::V, src::V) = vproduct!(dst, dst, src)

function vproduct!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                       x::DenseArray{T,N},
                                       y::DenseArray{T,N})
    @assert size(dst) == size(x) == size(y)
    @inbounds @simd for i in eachindex(dst, x, y)
        dst[i] = x[i]*y[i]
    end
    return dst
end

function vproduct!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                       sel::AbstractVector{Int},
                                       x::DenseArray{T,N},
                                       y::DenseArray{T,N})
    @assert size(dst) == size(x) == size(y)
    const n = length(dst)
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
        1 ≤ j ≤ n || throw(BoundsError())
        dst[j] = x[j]*y[j]
    end
    return dst
end

@doc @doc(vproduct) vproduct!

#------------------------------------------------------------------------------

vcombine(alpha::Real, x) = vscale(alpha, x)

vcombine!{T}(dst::T, alpha::Real, x::T) = vscale!(dst, alpha, x)

"""
### Linear combination of arrays

    vcombine(alpha, x)          -> dst
    vcombine(alpha, x, beta, y) -> dst

yields the linear combinations `alpha*x` and `alpha*x + beta*y`.  To avoid
allocating the result, the destination array `dst` can be specified with the
in-place version of the method:

    vcombine!(dst, alpha, x)          -> dst
    vcombine!(dst, alpha, x, beta, y) -> dst

The code is optimized for some specific values of the coefficients `alpha` and
`beta`.  For instance, if `alpha` (resp. `beta`) is zero, then the contents of
`x` (resp. `y`) is not used.

The source(s) and the destination can be the same.  For instance, the two
following lines of code produce the same result:

    vcombine!(dst, 1, dst, alpha, x)
    vupdate!(dst, alpha, x)

and the following statements also yied the same result:

    vcombine!(dst, alpha, x)
    vscale!(dst, alpha, x)

"""
vcombine{T}(alpha::Real, x::T, beta::Real, y::T) =
    vcombine!(vcreate(x), alpha, x, beta, y)

function vcombine!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                       alpha::Real, x::DenseArray{T,N},
                                       beta::Real, y::DenseArray{T,N})
    vcombine!(dst, T(alpha), x, T(beta), y)
end

function vcombine!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                       alpha::T, x::DenseArray{T,N},
                                       beta::T, y::DenseArray{T,N})
    @assert size(dst) == size(x) == size(y)
    if alpha == zero(T)
        vcombine!(dst, beta, y)
    elseif beta == zero(T)
        vcombine!(dst, alpha, x)
    elseif alpha == one(T)
        if beta == one(T)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] + y[i]
            end
        elseif beta == -one(T)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] - y[i]
            end
        else
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] + beta*y[i]
            end
        end
    elseif alpha == -one(T)
        if beta == one(T)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = y[i] - x[i]
            end
        elseif beta == -one(T)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = -x[i] - y[i]
            end
        else
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = beta*y[i] - x[i]
            end
        end
    else
        if beta == one(T)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = alpha*x[i] + y[i]
            end
        elseif beta == -one(T)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = alpha*x[i] - y[i]
            end
        else
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = alpha*x[i] + beta*y[i]
            end
        end
    end
    return dst
end

@doc @doc(vcombine) vcombine!

#-------------------------------------------------------------------------------
# PROJECTING VARIABLES

# clamp for scalars and unset bounds:
@inline clamp{T<:Real}(x::T, lo::T, hi::T) = max(lo, min(x, hi))
@inline clamp{T<:Real}(x::T, ::Void, hi::T) = min(x, hi)
@inline clamp{T<:Real}(x::T, lo::T, ::Void) = max(lo, x)

"""
    project_variables!(dst, lo, hi, src) -> dst

stores in destination `dst` the projection of the source variables `src` in the
box whose lower bound is `lo` and upper bound is `hi`.  The destination `dst`
is returned.

This is the same as `dst = clamp(src, lo, hi)` except that the result is
preallocated and that the operation is *much* faster (by a factor of 2-3).

"""
function project_variables!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::Real,
                                                hi::Real,
                                                src::DenseArray{T,N})
    project_variables!(dst, T(lo), T(hi), src)
end

function project_variables!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::T,
                                                hi::T,
                                                src::DenseArray{T,N})
    @assert size(dst) == size(src)
    @assert lo ≤ hi # this also check for NaN
    const bounded_below = (lo > T(-Inf))
    const bounded_above = (hi < T(+Inf))
    if bounded_below && bounded_above
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = clamp(src[i], lo, hi)
        end
    elseif bounded_below
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = clamp(src[i], lo, nothing)
        end
    elseif bounded_above
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = clamp(src[i], nothing, hi)
        end
    elseif !is(dst, src)
        vcopy!(dst, src)
    end
    return dst
end

function project_variables!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::DenseArray{T,N},
                                                hi::Real,
                                                src::DenseArray{T,N})
    project_variables!(dst, lo, T(hi), src)
end

function project_variables!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::DenseArray{T,N},
                                                hi::T,
                                                src::DenseArray{T,N})
    @assert size(dst) == size(src) == size(lo)
    if hi < T(+Inf)
        @inbounds @simd for i in eachindex(dst, src, lo)
            dst[i] = clamp(src[i], lo[i], hi)
        end
    else
        @inbounds @simd for i in eachindex(dst, src, lo)
            dst[i] = clamp(src[i], lo[i], nothing)
        end
    end
    return dst
end

function project_variables!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::Real,
                                                hi::DenseArray{T,N},
                                                src::DenseArray{T,N})
    project_variables!(dst, T(lo), hi, src)
end

function project_variables!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::T,
                                                hi::DenseArray{T,N},
                                                src::DenseArray{T,N})
    @assert size(dst) == size(src) == size(hi)
    if lo > T(-Inf)
        @inbounds @simd for i in eachindex(dst, src, hi)
            dst[i] = clamp(src[i], lo, hi[i])
        end
    else
        @inbounds @simd for i in eachindex(dst, src, hi)
            dst[i] = clamp(src[i], nothing, hi[i])
        end
    end
    return dst
end

function project_variables!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::DenseArray{T,N},
                                                hi::DenseArray{T,N},
                                                src::DenseArray{T,N})
    @assert size(dst) == size(src) == size(lo) == size(hi)
    @inbounds  @simd for i in eachindex(dst, src, lo, hi)
        dst[i] = clamp(src[i], lo[i], hi[i])
    end
    return dst
end

#------------------------------------------------------------------------------
# PROJECTING DIRECTION

# Orientation is indicated by a singleton.
abstract Orientation
immutable Forward  <: Orientation; end
immutable Backward <: Orientation; end

const FORWARD = Forward()
const BACKWARD = Backward()

convert(::Type{Orientation}, ::Union{Forward,Type{Forward}}) = FORWARD
convert(::Type{Orientation}, ::Union{Backward,Type{Backward}}) = BACKWARD
convert(::Type{Orientation}, s::Real) = (s > 0 ? FORWARD :
                                         s < 0 ? BACKWARD :
                                         error("invalid orientation"))
sign(::Union{Forward,Type{Forward}}) = +1
sign(::Union{Backward,Type{Backward}}) = -1

Orientation(x) = convert(Orientation, x)
orientation(T::DataType, x) = Orientation(x) === FORWARD ? +one(T) : -one(T)

@inline projdir{T<:AbstractFloat}(::Forward, x::T, d::T, lo::T, hi::T) =
    (d > zero(T) ? x < hi : x > lo) ? d : zero(T)

@inline projdir{T<:AbstractFloat}(::Backward, x::T, d::T, lo::T, hi::T) =
    (d < zero(T) ? x < hi : x > lo) ? d : zero(T)

@inline projdir{T<:AbstractFloat}(::Forward, x::T, d::T, ::Void, hi::T) =
    (d < zero(T) || x < hi) ? d : zero(T)

@inline projdir{T<:AbstractFloat}(::Backward, x::T, d::T, ::Void, hi::T) =
    (d > zero(T) || x < hi) ? d : zero(T)

@inline projdir{T<:AbstractFloat}(::Forward, x::T, d::T, lo::T, ::Void) =
    (d > zero(T) || x > lo) ? d : zero(T)

@inline projdir{T<:AbstractFloat}(::Backward, x::T, d::T, lo::T, ::Void) =
    (d < zero(T) || x > lo) ? d : zero(T)

function project_direction!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::T,
                                                hi::T,
                                                x::DenseArray{T,N},
                                                o::Orientation,
                                                d::DenseArray{T,N})
    @assert size(dst) == size(x) == size(d)
    @assert lo ≤ hi # this also check for NaN
    const bounded_above = (hi < T(+Inf))
    const bounded_below = (lo > T(-Inf))
    if bounded_below && bounded_above
        @inbounds @simd for i in eachindex(dst, x, d)
            dst[i] = projdir(o, x[i], d[i], lo, hi)
        end
    elseif bounded_below
        @inbounds @simd for i in eachindex(dst, x, d)
            dst[i] = projdir(o, x[i], d[i], lo, nothing)
        end
    elseif bounded_above
        @inbounds @simd for i in eachindex(dst, x, d)
            dst[i] = projdir(o, x[i], d[i], nothing, hi)
        end
    elseif !is(dst, d)
        vcopy!(dst, d)
    end
    return dst
end

function project_direction!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::DenseArray{T,N},
                                                hi::T,
                                                x::DenseArray{T,N},
                                                o::Orientation,
                                                d::DenseArray{T,N})
    @assert size(dst) == size(x) == size(d) == size(lo)
    if hi < T(+Inf)
        @inbounds @simd for i in eachindex(dst, x, d, lo)
            dst[i] = projdir(o, x[i], d[i], lo[i], hi)
        end
    else
        @inbounds @simd for i in eachindex(dst, x, d, lo)
            dst[i] = projdir(o, x[i], d[i], lo[i], nothing)
        end
    end
    return dst
end

function project_direction!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::T,
                                                hi::DenseArray{T,N},
                                                x::DenseArray{T,N},
                                                o::Orientation,
                                                d::DenseArray{T,N})
    @assert size(dst) == size(x) == size(d) == size(hi)
    if lo > T(-Inf)
        @inbounds @simd for i in eachindex(dst, x, d, hi)
            dst[i] = projdir(o, x[i], d[i], lo, hi[i])
        end
    else
        @inbounds @simd for i in eachindex(dst, x, d, hi)
            dst[i] = projdir(o, x[i], d[i], nothing, hi[i])
        end
    end
    return dst
end

function project_direction!{T<:AbstractFloat,N,}(dst::DenseArray{T,N},
                                                 lo::DenseArray{T,N},
                                                 hi::DenseArray{T,N},
                                                 x::DenseArray{T,N},
                                                 o::Orientation,
                                                 d::DenseArray{T,N})
    @assert size(dst) == size(x) == size(d) == size(lo) == size(hi)
    @inbounds @simd for i in eachindex(dst, x, d, lo, hi)
        dst[i] = projdir(o, x[i], d[i], lo[i], hi[i])
    end
    return dst
end

function project_direction!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::Real,
                                                hi::Real,
                                                x::DenseArray{T,N},
                                                orient,
                                                d::DenseArray{T,N})
    project_direction!(dst, T(lo), T(hi), x, Orientation(orient), d)
end

function project_direction!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::DenseArray{T,N},
                                                hi::Real,
                                                x::DenseArray{T,N},
                                                orient,
                                                d::DenseArray{T,N})
    project_direction!(dst, lo, T(hi), x, Orientation(orient), d)
end

function project_direction!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                                lo::Real,
                                                hi::DenseArray{T,N},
                                                x::DenseArray{T,N},
                                                orient,
                                                d::DenseArray{T,N})
    project_direction!(dst, T(lo), hi, x, Orientation(orient), d)
end

function project_gradient!{T<:AbstractFloat,N}(dst::DenseArray{T,N},
                                               lo::Union{Real,DenseArray{T,N}},
                                               hi::Union{Real,DenseArray{T,N}},
                                               x::DenseArray{T,N},
                                               d::DenseArray{T,N})
    project_direction!(dst, lo, hi, x, BACKWARD, d)
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
function step_limits{T<:AbstractFloat,N}(lo::T,
                                         hi::T,
                                         x::AbstractArray{T,N},
                                         orient,
                                         d::AbstractArray{T,N})
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

function step_limits{T<:AbstractFloat,N}(lo::AbstractArray{T,N},
                                         hi::T,
                                         x::AbstractArray{T,N},
                                         orient,
                                         d::AbstractArray{T,N})
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
            @simd for i in eachindex(x, d, lo)
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
            @simd for i in eachindex(x, d, lo)
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

function step_limits{T<:AbstractFloat,N}(lo::T,
                                         hi::AbstractArray{T,N},
                                         x::AbstractArray{T,N},
                                         orient,
                                         d::AbstractArray{T,N})
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
            @simd for i in eachindex(x, d, hi)
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
            @simd for i in eachindex(x, d, hi)
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

function step_limits{T<:AbstractFloat,N}(lo::AbstractArray{T,N},
                                         hi::AbstractArray{T,N},
                                         x::AbstractArray{T,N},
                                         orient,
                                         d::AbstractArray{T,N})
    @assert size(lo) == size(x)
    @assert size(hi) == size(x)
    @assert size(d)  == size(x)
    const ZERO = zero(T)
    const s = orientation(T, orient)
    smin = T(Inf)
    smax = ZERO
    @inbounds begin
        @simd for i in eachindex(x, d, lo, hi)
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

function step_limits{T<:AbstractFloat,N}(lo::Real,
                                         hi::Real,
                                         x::AbstractArray{T,N},
                                         orient,
                                         d::AbstractArray{T,N})
    step_limits(T(lo), T(hi), x, orient, d)
end

function step_limits{T<:AbstractFloat,N}(lo::AbstractArray{T,N},
                                         hi::Real,
                                         x::AbstractArray{T,N},
                                         orient,
                                         d::AbstractArray{T,N})
    step_limits(lo, T(hi), x, orient, d)
end

function step_limits{T<:AbstractFloat,N}(lo::Real,
                                         hi::AbstractArray{T,N},
                                         x::AbstractArray{T,N},
                                         orient,
                                         d::AbstractArray{T,N})
    step_limits(T(lo), hi, x, orient, d)
end

#------------------------------------------------------------------------------
# GETTING FREE VARIABLES

@inline may_move{T<:AbstractFloat}(::Forward, x::T, d::T, lo::T, ::Void) =
    d > zero(T) || (d != zero(T) && x > lo)

@inline may_move{T<:AbstractFloat}(::Backward, x::T, d::T, lo::T, ::Void) =
    d < zero(T) || (d != zero(T) && x > lo)

@inline may_move{T<:AbstractFloat}(::Forward, x::T, d::T, ::Void, hi::T) =
    d < zero(T) || (d != zero(T) && x < hi)

@inline may_move{T<:AbstractFloat}(::Backward, x::T, d::T, ::Void, hi::T) =
    d > zero(T) || (d != zero(T) && x < hi)

@inline may_move{T<:AbstractFloat}(::Forward, x::T, d::T, lo::T, hi::T) =
    d != zero(T) && (d < zero(T) ? x > lo : x < hi)

@inline may_move{T<:AbstractFloat}(::Backward, x::T, d::T, lo::T, hi::T) =
    d != zero(T) && (d > zero(T) ? x > lo : x < hi)

"""
## Get free variables when following a direction

    sel =  get_free_variables(lo, hi, x, orient, d)

yields the list of components of the variables `x` which are allowed to vary
along the search direction `sign(orient)*d` under box constraints with `lo` and
`hi` the lower and upper bounds.

If the projected gradient `gp` of the objective function is available, the free
variables can be obtained by:

    sel =  get_free_variables(gp)

where the projected gradient `gp` has been computed as:

    project_direction!(gp, lo, hi, x, -1, g)

with `g` the gradient of the objective function at `x`.

"""
function get_free_variables{T<:AbstractFloat,N}(gp::DenseArray{T,N})
    const ZERO = zero(T)
    const n = length(gp)
    sel = Array{Int}(n)
    j = 0
    @inbounds @simd for i in 1:n
        if gp[i] != ZERO
            j += 1
            sel[j] = i
        end
    end
    return (j == n ? sel : resize!(sel, j))
end

function get_free_variables{T<:AbstractFloat,N}(lo::T,
                                                hi::T,
                                                x::DenseArray{T,N},
                                                o::Orientation,
                                                d::DenseArray{T,N})
    @assert size(x) == size(d)
    @assert lo ≤ hi # this also check for NaN
    const bounded_below = (lo > T(-Inf))
    const bounded_above = (hi < T(+Inf))
    const n = length(x)
    sel = Array{Int}(n)
    j = 0
    if bounded_below && bounded_above
        @inbounds @simd for i in 1:n
            if may_move(o, x[i], d[i], lo, hi)
                j += 1
                sel[j] = i
            end
        end
    elseif bounded_below
        @inbounds @simd for i in 1:n
            if may_move(o, x[i], d[i], lo, nothing)
                j += 1
                sel[j] = i
            end
        end
    elseif bounded_above
        @inbounds @simd for i in 1:n
            if may_move(o, x[i], d[i], nothing, hi)
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

function get_free_variables{T<:AbstractFloat,N}(lo::DenseArray{T,N},
                                                hi::T,
                                                x::DenseArray{T,N},
                                                o::Orientation,
                                                d::DenseArray{T,N})
    @assert size(x) == size(d) == size(lo)
    const n = length(x)
    sel = Array{Int}(n)
    j = 0
    if hi < T(+Inf)
        @inbounds @simd for i in 1:n
            if may_move(o, x[i], d[i], lo[i], hi)
                j += 1
                sel[j] = i
            end
        end
    else
        @inbounds @simd for i in 1:n
            if may_move(o, x[i], d[i], lo[i], nothing)
                j += 1
                sel[j] = i
            end
        end
    end
    return (j == n ? sel : resize!(sel, j))
end

function get_free_variables{T<:AbstractFloat,N}(lo::T,
                                                hi::DenseArray{T,N},
                                                x::DenseArray{T,N},
                                                o::Orientation,
                                                d::DenseArray{T,N})
    @assert size(x) == size(d) == size(hi)
    const n = length(x)
    sel = Array{Int}(n)
    j = 0
    if lo > T(-Inf)
        @inbounds @simd for i in 1:n
            if may_move(o, x[i], d[i], lo, hi[i])
                j += 1
                    sel[j] = i
            end
        end
    else
        @inbounds @simd for i in 1:n
            if may_move(o, x[i], d[i], nothing, hi[i])
                j += 1
                sel[j] = i
            end
        end
    end
    return (j == n ? sel : resize!(sel, j))
end

function get_free_variables{T<:AbstractFloat,N}(lo::DenseArray{T,N},
                                                hi::DenseArray{T,N},
                                                x::DenseArray{T,N},
                                                o::Orientation,
                                                d::DenseArray{T,N})
    @assert size(x) == size(d) == size(lo) == size(hi)
    const n = length(x)
    sel = Array{Int}(n)
    j = 0
    @inbounds @simd for i in 1:n
        if may_move(o, x[i], d[i], lo[i], hi[i])
            j += 1
            sel[j] = i
        end
    end
    return (j == n ? sel : resize!(sel, j))
end

function get_free_variables{T<:AbstractFloat,N}(lo::Real,
                                                hi::Real,
                                                x::DenseArray{T,N},
                                                orient,
                                                d::DenseArray{T,N})
    get_free_variables(T(lo), T(hi), x, Orientation(orient), d)
end

function get_free_variables{T<:AbstractFloat,N}(lo::DenseArray{T,N},
                                                hi::Real,
                                                x::DenseArray{T,N},
                                                orient,
                                                d::DenseArray{T,N})
    get_free_variables(lo, T(hi), x, Orientation(orient), d)
end

function get_free_variables{T<:AbstractFloat,N}(lo::Real,
                                                hi::DenseArray{T,N},
                                                x::DenseArray{T,N},
                                                orient,
                                                d::DenseArray{T,N})
    get_free_variables(T(lo), hi, x, Orientation(orient), d)
end

#------------------------------------------------------------------------------
