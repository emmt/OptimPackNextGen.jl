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

export inner, norm1, norm2, normInf
export swap!
export scale!, update!, combine!

# norm2(v) --
#
# Return Euclidean (L2) norm of *vector* `v`.
#
function norm2{T<:AbstractFloat,N}(v::Array{T,N})
    s::T = zero(T)
    @simd for i in 1:length(v)
        @inbounds s += v[i]*v[i]
    end
    return sqrt(s)
end

# norm1(v) --
#
# Return L1 norm of *vector* `v`.
#
function norm1{T<:AbstractFloat,N}(v::Array{T,N})
    s::T = zero(T)
    @simd for i in 1:length(v)
        @inbounds s += abs(v[i])
    end
    return s
end

# normInf(v) --
#
# Return infinite norm of *vector* `v`.
#
function normInf{T<:AbstractFloat,N}(v::Array{T,N})
    s::T = zero(T)
    @simd for i in 1:length(v)
        @inbounds s = max(s, abs(v[i]))
    end
    return s
end

"""
Compute scalar product.

    inner(x,y)

Compute the inner product (a.k.a. scalar product) between *vectors* `x` and
`y`.  The *triple* inner product between *vectors* `w`, `x` and `y` can be
computed by:

    inner(w,x,y)

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

# swap!(x,y) --
#
# Exchange the contents of *vectors* `x` and `y`.
#
function swap!{T,N}(x::Array{T,N}, y::Array{T,N})
    @assert(size(x) == size(y))
    temp::T
    @simd for i in 1:length(x)
        @inbounds temp = x[i]
        @inbounds x[i] = y[i]
        @inbounds y[i] = temp
    end
end

# scale!(dst, alpha) --
#
# In-place scaling of destination *vector* `dst` by scalar `alpha`.
#
function scale!{T<:AbstractFloat,N}(dst::Array{T,N}, alpha::Real)
    const a::T = alpha
    if a == zero(T)
        @simd for i in 1:length(dst)
            @inbounds dst[i] = zero(T)
        end
    elseif a == -one(T)
        @simd for i in 1:length(dst)
            @inbounds dst[i] = -dst[i]
        end
    elseif a != one(T)
        @simd for i in 1:length(dst)
            @inbounds dst[i] *= a
        end
    end
end

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

# update!(dst, alpha, x) --
#
# Increment the components of destination *vector* `dst` by those of
# `alpha*x`.
#
function update!{T<:AbstractFloat,N}(dst::Array{T,N},
                                     alpha::Real, x::Array{T,N})
    @assert(size(x) == size(dst))
    const a::T = alpha
    if a == one(T)
        @simd for i in 1:length(dst)
            @inbounds dst[i] += x[i]
        end
    elseif a == -one(T)
        @simd for i in 1:length(dst)
            @inbounds dst[i] -= x[i]
        end
    elseif a != zero(T)
        @simd for i in 1:length(dst)
            @inbounds dst[i] += a*x[i]
        end
    end
end

# combine!(dst, alpha, x, beta, y) --
#
# Store the linear combination `alpha*x + beta*y` into the destination
# *vector* `dst`.
#
function combine!{T<:AbstractFloat,N}(dst::Array{T,N},
                                      alpha::Real, x::Array{T,N},
                                      beta::Real,  y::Array{T,N})
    @assert(size(x) == size(dst))
    @assert(size(y) == size(dst))
    const n = length(dst)
    const a::T = alpha
    const b::T = beta
    if a == zero(T)
        if b == zero(T)
            @simd for i in 1:n
                @inbounds dst[i] = zero(T)
            end
        elseif b == one(T)
            @simd for i in 1:n
                @inbounds dst[i] = y[i]
            end
        elseif b == -one(T)
            @simd for i in 1:n
                @inbounds dst[i] = -y[i]
            end
        else
            @simd for i in 1:n
                @inbounds dst[i] = b*y[i]
            end
        end
    elseif a == one(T)
        if b == zero(T)
            @simd for i in 1:n
                @inbounds dst[i] = x[i]
            end
        elseif b == one(T)
            @simd for i in 1:n
                @inbounds dst[i] = x[i] + y[i]
            end
        elseif b == -one(T)
            @simd for i in 1:n
                @inbounds dst[i] = x[i] - y[i]
            end
        else
            @simd for i in 1:n
                @inbounds dst[i] = x[i] + b*y[i]
            end
        end
    elseif a == -one(T)
        if b == zero(T)
            @simd for i in 1:n
                @inbounds dst[i] = -x[i]
            end
        elseif b == one(T)
            @simd for i in 1:n
                @inbounds dst[i] = y[i] - x[i]
            end
        elseif b == -one(T)
            @simd for i in 1:n
                @inbounds dst[i] = -x[i] - y[i]
            end
        else
            @simd for i in 1:n
                @inbounds dst[i] = b*y[i] - x[i]
            end
        end
    else
        if b == zero(T)
            @simd for i in 1:n
                @inbounds dst[i] = a*x[i]
            end
        elseif b == one(T)
            @simd for i in 1:n
                @inbounds dst[i] = a*x[i] + y[i]
            end
        elseif b == -one(T)
            @simd for i in 1:n
                @inbounds dst[i] = a*x[i] - y[i]
            end
        else
            @simd for i in 1:n
                @inbounds dst[i] = a*x[i] + b*y[i]
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
