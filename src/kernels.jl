#
# kernels.jl --
#
# Kernel functions used for linear filtering, windowing or linear
# interpolation.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

module Kernels

import Base: length, apply, call
export iscardinal, isnormalized
export BoxKernel, TriangleKernel, QuadraticKernel, CubicKernel,
       CatmullRomKernel, KeysKernel, MitchellNetraviliKernel, LanczosKernel

# This function is needed for rational constants.
@inline _{T<:AbstractFloat}(::Type{T}, num::Real, den::Real) = T(num)/T(den)

"""
Abstract `Kernel` type is the super type of kernel functions used for
filtering, windowing or interpolation.  For efficiency reasons, a kernel is
parameterized with the floating point type of its argument and return value.

Computing the value of a kernel function `ker` at position `x` is done by one
of:

    ker(x)
    call(ker, x)
    apply(ker, x)   # deprecated

`lenght(ker)` yields the size of the support of kernel `ker`.  All kernel
supports are symmetric; that is `ker(x)` is zero if `abs(x) > lenght(ker)/2`.
"""
abstract Kernel{T<:AbstractFloat}

"""
Kernels of type `SingletonKernel{T}` are kernels with no parameters (other than
their argument type), so that they can only have a single instance.
"""
abstract SingletonKernel{T<:AbstractFloat} <: Kernel{T}

# Default methods.
length{T<:Kernel}(::T) = length(T)
apply{T<:AbstractFloat}(ker::Kernel{T}, x::Real) = call(ker, x)
apply{T<:SingletonKernel}(::T, x::Real) = call(T, x)

"""
`isnormalized(ker)` returns a boolean indicating whether the kernel `ker` has
the partition of unity property.  That is, the sum of the values computed by
the kernel `ker` on a unit spaced grid is equal to one.
"""
isnormalized{T<:Kernel}(::T) = isnormalized(T)

"""
`iscardinal(ker)` returns a boolean indicating whether the kernel `ker`
is zero for non-zero integer arguments.
"""
iscardinal{T<:Kernel}(::T) = iscardinal(T)

#------------------------------------------------------------------------------
"""
# Box Kernel

The box kernel (also known as Fourier window or Dirichlet window) is a 1st
order (constant) B-spline equals to `1` on `[-1/2,+1/2[` and `0` elsewhere.
"""
immutable BoxKernel{T} <: SingletonKernel{T}; end

BoxKernel{T<:AbstractFloat}(::Type{T}) = BoxKernel{T}()

const box = BoxKernel(Cdouble)

function call{T<:AbstractFloat}(::Type{BoxKernel{T}}, x::T)
    _(T,-1,2) <= x < _(T,1,2) ? one(T) : zero(T)
end

length{T<:BoxKernel}(::Type{T}) = 1
iscardinal{T<:BoxKernel}(::Type{T}) = true
isnormalized{T<:BoxKernel}(::Type{T}) = true

#------------------------------------------------------------------------------
"""
# Triangle Kernel

The triangle kernel (also known as a.k.a. Bartlett window or Fejér window)
is a 2nd order (linear) B-spline.
"""
immutable TriangleKernel{T} <: SingletonKernel{T}; end

TriangleKernel{T<:AbstractFloat}(::Type{T}) = TriangleKernel{T}()

const triangle = TriangleKernel(Cdouble)

function call{T<:AbstractFloat}(::Type{TriangleKernel{T}}, x::T)
    t = abs(x)
    t < 1 ? one(T) - t : zero(T)
end

length{T<:TriangleKernel}(::Type{T}) = 2
iscardinal{T<:TriangleKernel}(::Type{T}) = true
isnormalized{T<:TriangleKernel}(::Type{T}) = true

#------------------------------------------------------------------------------
"""
# Quadratic Kernel

The quadratic kernel is 3rd order (quadratic) B-spline.
"""
immutable QuadraticKernel{T} <: SingletonKernel{T}; end

QuadraticKernel{T<:AbstractFloat}(::Type{T}) = QuadraticKernel{T}()

const quadratic = QuadraticKernel(Cdouble)

function call{T<:AbstractFloat}(::Type{QuadraticKernel{T}}, x::T)
    t = abs(x)
    if t >= _(T,3,2)
        return zero(T)
    elseif t <= _(T,1,2)
        return _(T,3,4) - t*t
    else
        t -= _(T,3,2)
        return _(T,1,2)*t*t
    end
end

length{T<:QuadraticKernel}(::Type{T}) = 3
iscardinal{T<:QuadraticKernel}(::Type{T}) = false
isnormalized{T<:QuadraticKernel}(::Type{T}) = true

#------------------------------------------------------------------------------
"""
# Cubic Spline Kernel

The 4th order (cubic) B-spline kernel is also known as Parzen window or de la
Vallée Poussin window.
"""
immutable CubicKernel{T} <: SingletonKernel{T}; end

CubicKernel{T<:AbstractFloat}(::Type{T}) = CubicKernel{T}()

const cubic = CubicKernel(Cdouble)

function call{T<:AbstractFloat}(::Type{CubicKernel{T}}, x::T)
    t = abs(x);
    if t >= T(2)
        return zero(T)
    elseif t <= one(T)
        return (_(T,1,2)*t - one(T))*t*t + _(T,2,3)
    else
        t = T(2) - t
        return _(T,1,6)*t*t*t
    end
end

length{T<:CubicKernel}(::Type{T}) = 4
iscardinal{T<:CubicKernel}(::Type{T}) = false
isnormalized{T<:CubicKernel}(::Type{T}) = true

#------------------------------------------------------------------------------
"""
# Mitchell & Netravali kernels

These kernels are cubic splines which depends on 2 parameters `b` and `c`.
whatever the values of `(b,c)`, all these kernels are "normalized", symmetric
and their value and first derivative are continuous.

Taking `b = 0` is a sufficient and necessary condition to have cardinal
kernels.  This correspond to Keys's family of kernels.

Using the constraint: `b + 2c = 1` yields a cubic filter with, at least,
quadratic order approximation.

Some specific values of `(b,c)` yield other well known kernels:

    (b,c) = (1,0)     ==> cubic B-spline
    (b,c) = (0,-a)    ==> Keys's cardinal cubics
    (b,c) = (0,1/2)   ==> Catmull-Rom cubics
    (b,c) = (b,0)     ==> Duff's tensioned B-spline
    (b,c) = (1/3,1/3) ==> recommended by Mitchell-Netravali

Reference:

* Mitchell & Netravali ("Reconstruction Filters in Computer Graphics",
  Computer Graphics, Vol. 22, Number. 4, August 1988).
  http://www.cs.utexas.edu/users/fussell/courses/cs384g/lectures/mitchell/Mitchell.pdf.

"""
immutable MitchellNetraviliKernel{T} <: Kernel{T}
    b ::T
    c ::T
    p0::T
    p2::T
    p3::T
    q0::T
    q1::T
    q2::T
    q3::T
    function MitchellNetraviliKernel(b::T, c::T)
        new(b, c,
            T(   6 -  2*b       )/T(6),
            T( -18 + 12*b +  6*c)/T(6),
            T(  12 -  9*b -  6*c)/T(6),
            T(        8*b + 24*c)/T(6),
            T(     - 12*b - 48*c)/T(6),
            T(        6*b + 30*c)/T(6),
            T(   -      b -  6*c)/T(6))
    end
end

function MitchellNetraviliKernel{T<:AbstractFloat}(::Type{T}, b::Real, c::Real)
    MitchellNetraviliKernel{T}(T(b), T(c))
end

# Create Mitchell-Netravali kernel with default parameters.
function MitchellNetraviliKernel{T<:AbstractFloat}(::Type{T})
    MitchellNetraviliKernel(T, 1//3, 1//3)
end

const mitchell_netravili = MitchellNetraviliKernel(Cdouble)

function call{T<:AbstractFloat}(ker::MitchellNetraviliKernel{T}, x::T)
    t = abs(x)
    t >= T(2) ? zero(T) :
    t <= one(T) ? (ker.p3*t + ker.p2)*t*t + ker.p0 :
    ((ker.q3*t + ker.q2)*t + ker.q1)*t + ker.q0
end

length{T<:MitchellNetraviliKernel}(::Type{T}) = 4
iscardinal{T<:AbstractFloat}(ker::MitchellNetraviliKernel{T}) = (ker.b == zero(T))
isnormalized{T<:MitchellNetraviliKernel}(::Type{T}) = true

#------------------------------------------------------------------------------
"""
# Keys cardinal kernels

These kernels are piecewise normalized cardinal cubic spline which depend on
one parameter `a`.

Reference:

* Keys, Robert, G., "Cubic Convolution Interpolation for Digital Image
  Processing", IEEE Trans. Acoustics, Speech, and Signal Processing,
  Vol. ASSP-29, No. 6, December 1981, pp. 1153-1160.

"""
immutable KeysKernel{T} <: Kernel{T}
    a ::T
    p0::T
    p2::T
    p3::T
    q0::T
    q1::T
    q2::T
    q3::T
    function KeysKernel(a::T)
        new(a, 1, -a - 3, a + 2, -4*a, 8*a, -5*a, a)
    end
end

KeysKernel{T<:AbstractFloat}(::Type{T}, a::Real) = KeysKernel(T, T(a))
KeysKernel{T<:AbstractFloat}(::Type{T}, a::T) = KeysKernel{T}(a)

function call{T<:AbstractFloat}(ker::KeysKernel{T}, x::T)
    t = abs(x)
    t >= T(2) ? zero(T) :
    t <= one(T) ? (ker.p3*t + ker.p2)*t*t + ker.p0 :
    ((ker.q3*t + ker.q2)*t + ker.q1)*t + ker.q0
end

length{T<:KeysKernel}(::Type{T}) = 4
iscardinal{T<:KeysKernel}(::Type{T}) = true
isnormalized{T<:KeysKernel}(::Type{T}) = true

#------------------------------------------------------------------------------
# Catmull-Rom kernel is a special case of Mitchell & Netravali kernel.

immutable CatmullRomKernel{T} <: SingletonKernel{T}; end

CatmullRomKernel{T<:AbstractFloat}(::Type{T}) = CatmullRomKernel{T}()

const catmull_rom = CatmullRomKernel(Cdouble)

function call{T<:AbstractFloat}(::Type{CatmullRomKernel{T}}, x::T)
    t = abs(x)
    t >= T(2) ? zero(T) :
    t <= one(T) ? (_(T,3,2)*t - _(T,5,2))*t*t + one(T) :
    ((_(T,5,2) - _(T,1,2)*t)*t - T(4))*t + T(2)
end

length{T<:CatmullRomKernel}(::Type{T}) = 4
iscardinal{T<:CatmullRomKernel}(::Type{T}) = true
isnormalized{T<:CatmullRomKernel}(::Type{T}) = true

#------------------------------------------------------------------------------
"""
# Lanczos Resampling Kernel

`LanczosKernel(a)` creates a Lanczos kernel of support size `2a`

The Lanczos kernel does not have the partition of unity property.  However,
Lanczos kernel tends to be normalized for large support size.
[link](https://en.wikipedia.org/wiki/Lanczos_resampling)
"""
immutable LanczosKernel{T} <: Kernel{T}
    s::Int # support
    a::T   # 1/2 support
    b::T   # a/pi^2
    c::T   # pi/a
    function LanczosKernel(n::Integer)
        @assert n > 0
        a = T(n)
        new(2*n, a, a/pi^2, pi/a)
    end
end

LanczosKernel{T<:AbstractFloat}(::Type{T}, n::Integer) = LanczosKernel{T}(n)

function call{T<:AbstractFloat}(ker::LanczosKernel{T}, x::T)
    abs(x) >= ker.a ? zero(T) :
    x == zero(T) ? one(T) :
    ker.b*sin(pi*x)*sin(ker.c*x)/(x*x)
end

length{T<:AbstractFloat}(ker::LanczosKernel{T}) = ker.s
iscardinal{T<:LanczosKernel}(::Type{T}) = true
isnormalized{T<:LanczosKernel}(::Type{T}) = false

#------------------------------------------------------------------------------

# Methods needed to cope with type conversions (must be "specialized" to avoid
# dispatching ambiguities).
for K in (KeysKernel, MitchellNetraviliKernel, LanczosKernel)
    @eval begin
        function call{T<:AbstractFloat}(ker::$K{T}, x::Real)
            call(ker, T(x))
        end
    end
end

# Provide methods for singleton kernels.
call{T<:SingletonKernel}(::T, x::Real) = call(T, x)
for K in (BoxKernel, TriangleKernel, QuadraticKernel, CubicKernel,
          CatmullRomKernel)
    @eval begin
        function call{T<:AbstractFloat}(::Type{$K{T}}, x::Real)
            call($K{T}, T(x))
        end
    end
end

function call{T<:AbstractFloat,R<:Real,N}(ker::Kernel{T},
                                          x::AbstractArray{R,N})
    y = Array(T, size(x))
    @inbounds for i in 1:length(x)
        y[i] = ker(T(x[i]))
    end
    return y
end

function call{T<:AbstractFloat,N}(ker::Kernel{T},
                                  x::AbstractArray{T,N})
    y = Array(T, size(x))
    @inbounds for i in 1:length(x)
        y[i] = ker(x[i])
    end
    return y
end

end # module
