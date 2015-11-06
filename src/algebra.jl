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
# Copyright (C) 2015, Éric Thiébaut & Jonathan Léger.
#
#------------------------------------------------------------------------------

module Algebra

# FIXME: scale! conflict with Base.scale!

import Base.scale!
export inner, norm1, norm2, normInf
export swap!
export scale!, update!, combine!

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
    return sqrt(s)
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
    return s
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
    return s
end

"""
### Compute scalar product

The call:
```
    inner(x,y)
```
computes the inner product (a.k.a. scalar product) between `x` and `y` (which
must have the same size).  The triple inner product between `w`, `x` and `y`
can be computed by:
```
    inner(w,x,y)
```
"""
function inner{T<:AbstractFloat,N}(x::Array{T,N}, y::Array{T,N})
    @assert(size(x) == size(y))
    s::T = 0
    @simd for i in 1:length(x)
        @inbounds s += x[i]*y[i]
    end
    return s
end

# FIXME: this is a suggestion:
function inner{T<:Integer,N}(j::Array{T,N}, k::Array{T,N})
    @assert(size(j) == size(k))
    s::T = 0
    @simd for i in 1:length(j)
        @inbounds s += j[i]*k[i]
    end
    return s/lenght(x)
end

function inner{T<:AbstractFloat,N}(w::Array{T,N}, x::Array{T,N}, y::Array{T,N})
    @assert(size(x) == size(w))
    @assert(size(y) == size(w))
    s::T = zero(T)
    @simd for i in 1:length(w)
        @inbounds s += w[i]*x[i]*y[i]
    end
    return s
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

# FIXME: # # scale!(dst, alpha) --
# FIXME: # #
# FIXME: # # In-place scaling of destination *vector* `dst` by scalar `alpha`.
# FIXME: # #
# FIXME: # function scale!{T<:AbstractFloat,N}(dst::Array{T,N}, alpha::Real)
# FIXME: #     const a::T = alpha
# FIXME: #     if a == zero(T)
# FIXME: #         @simd for i in 1:length(dst)
# FIXME: #             @inbounds dst[i] = zero(T)
# FIXME: #         end
# FIXME: #     elseif a == -one(T)
# FIXME: #         @simd for i in 1:length(dst)
# FIXME: #             @inbounds dst[i] = -dst[i]
# FIXME: #         end
# FIXME: #     elseif a != one(T)
# FIXME: #         @simd for i in 1:length(dst)
# FIXME: #             @inbounds dst[i] *= a
# FIXME: #         end
# FIXME: #     end
# FIXME: # end

# scale!(dst, alpha, x) --
#
# Store `alpha*x` in destination *vector* `dst`.
#
function scale!{T<:Real,N}(dst::Array{T,N}, alpha::Real, x::Array{T,N})
    @assert(size(x) == size(dst))
    const a::T = alpha
    if a == zero(T)
        @simd for i in 1:length(dst)
            @inbounds dst[i] = zero(T)
        end
    elseif a == one(T)
        @simd for i in 1:length(dst)
            @inbounds dst[i] = x[i]
        end
    elseif a == -one(T)
        @simd for i in 1:length(dst)
            @inbounds dst[i] = -x[i]
        end
    else
        @simd for i in 1:length(dst)
            @inbounds dst[i] = a*x[i]
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
`alpha`.  For instance, if `alpha` is zero, then `dst` is filled with zeros
without using the contents of `x`.
"""
function update!{T<:AbstractFloat,N}(dst::Array{T,N},
                                     alpha::Real, x::Array{T,N})
    @assert(size(x) == size(dst))
    const a::T = alpha
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

"""
### Linear combination of arrays

The call:
```
    combine!(dst, alpha, x, beta, y)
```
stores the linear combination `alpha*x + beta*y` into the destination *vector*
`dst`.  The code is optimized for some specific values of the coefficients
`alpha` and `beta`.  For instance, if `alpha` (resp. `beta`) is zero, then the
contents of `x` (resp. `y`) is not used.
"""
function combine!{T<:AbstractFloat,N}(dst::Array{T,N},
                                      alpha::Real, x::Array{T,N},
                                      beta::Real,  y::Array{T,N})
    @assert(size(x) == size(dst))
    @assert(size(y) == size(dst))
    const n = length(dst)
    const a::T = alpha
    const b::T = beta
    @inbounds begin
        if a == zero(T)
            if b == zero(T)
                @simd for i in 1:n
                    dst[i] = zero(T)
                end
            elseif b == one(T)
                @simd for i in 1:n
                    dst[i] = y[i]
                end
            elseif b == -one(T)
                @simd for i in 1:n
                    dst[i] = -y[i]
                end
            else
                @simd for i in 1:n
                    dst[i] = b*y[i]
                end
            end
        elseif a == one(T)
            if b == zero(T)
                @simd for i in 1:n
                    dst[i] = x[i]
                end
            elseif b == one(T)
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
            if b == zero(T)
                @simd for i in 1:n
                    dst[i] = -x[i]
                end
            elseif b == one(T)
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
            if b == zero(T)
                @simd for i in 1:n
                    dst[i] = a*x[i]
                end
            elseif b == one(T)
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

end # module

# Local Variables:
# mode: Julia
# tab-width: 8
# indent-tabs-mode: nil
# fill-column: 79
# coding: utf-8
# ispell-local-dictionary: "american"
# End:
