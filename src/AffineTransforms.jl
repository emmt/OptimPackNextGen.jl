#
# AffineTransforms.jl --
#
# Implementation of affine transforms which are notably useful for coordinate
# transforms.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>
#
# This file is free software; as a special exception the author gives unlimited
# permission to copy and/or distribute it, with or without modifications, as
# long as this notice is preserved.
#
# This software is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY, to the extent permitted by law; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#------------------------------------------------------------------------------

module AffineTransforms

export AffineTransform2D,
   combine, rotate, translate, invert, intercept,
   jacobian

import Base: convert, det, scale, show, *, +, /, \
"""
# Affine 2D Transforms

An affine 2D transform `C` is defined by 6 real coefficients, `Cxx`, `Cxy`,
`Cx`, `Cyx`, `Cyy` and `Cy`.  Such a transform maps `(x,y)` as `(xp,yp)` given
by:
```
  xp = Cxx*x + Cxy*y + Cx
  yp = Cyx*x + Cyy*y + Cy
```

The immutable type `AffineTransform2D` is used to store an affine 2D transform
`C`, it can be created by:
```
  C = AffineTransform2D(T) # yields the identity with type T
  C = AffineTransform2D(Cxx, Cxy, Cx, Cyx, Cyy, Cy)
```

Many operations are available to manage or apply affine transforms:
```
  (xp, yp) = A(x, y)         # idem
  (xp, yp) = A(xy)           # idem
  (xp, yp) = A*xy            # idem

  B = convert(T, A)     # convert coefficients of transform A to be of type T

  C = combine(A, B)     # combine 2 transforms, C = apply B then A
  C = A*B               # idem

  B = translate(tx, ty, A)   # B = apply A then translate by (tx,ty)
  B = translate(t, A)        # idem with t = (tx,ty)
  B = t + A                  # idem

  B = translate(A, tx, ty)   # B = translate by (tx,ty) then apply A
  B = translate(A, t)        # idem with t = (tx,ty)
  B = A + t                  # idem

  B = rotate(θ, A)   # B = apply A then rotate by angle θ
  C = rotate(A, θ)   # C = rotate by angle θ then apply A

  B = scale(ρ, A)    # B = apply A then scale by ρ
  B = ρ*A            # idem
  C = scale(A, ρ)    # C = scale by ρ then apply A
  C = A*ρ            # idem

  B = invert(A)      # reciprocal coordinate transform
  C = A/B            # right division, same as: C = combine(A, invert(B))
  C = A\B            # left division, same as: C = combine(invert(A), B)
```

"""
immutable AffineTransform2D{T<:AbstractFloat}
    xx::T
    xy::T
    x ::T
    yx::T
    yy::T
    y ::T
    function AffineTransform2D(xx::Real, xy::Real, x::Real,
                               yx::Real, yy::Real, y::Real)
        new(xx, xy, x, yx, yy, y)
    end
end

# Use Cdouble type by default.
AffineTransform2D() = AffineTransform2D(Cdouble)
AffineTransform2D(xx::Real, xy::Real, x::Real,
                  yx::Real, yy::Real, y::Real) =
                      AffineTransform2D(Cdouble, xx, xy, x, yx, yy, y)

function AffineTransform2D{T<:AbstractFloat}(::Type{T})
    const ZERO = zero(T)
    const ONE = one(T)
    AffineTransform2D{T}(ONE, ZERO, ZERO, ZERO, ONE, ZERO)
end

function AffineTransform2D{T<:AbstractFloat}(::Type{T},
                                             xx::Real, xy::Real, x::Real,
                                             yx::Real, yy::Real, y::Real)
    AffineTransform2D{T}(xx, xy, x, yx, yy, y)
end

# The following is a no-op when the destination type matches that of the
# source.
#
# The trick is to have a more restrictive signature than the general case
# above.  So the template type T for the result must have the same
# restrictions as in the general case.
#
# Another trick to remember: you can call a specific constructor, e.g.
# AffineTransform2D{Float16}, but this is not allowed for methods for which
# you must rely on Julia dispatching rules.
#
# When you make specialized versions of methods beware of infinite loops
# resulting from recursively calling the same method.  The diagnostic is a
# stack overflow.
#
convert{T<:AbstractFloat}(::Type{AffineTransform2D{T}}, A::AffineTransform2D{T}) = A

convert{T<:AbstractFloat, S}(::Type{AffineTransform2D{T}}, A::AffineTransform2D{S}) =
    AffineTransform2D{T}(A.xx, A.xy, A.x, A.yx, A.yy, A.y)

#------------------------------------------------------------------------------
# apply the transform to some coordinates:

(A::AffineTransform2D{T}){T<:AbstractFloat}(x::T, y::T) =
    (A.xx*x + A.xy*y + A.x,
     A.yx*x + A.yy*y + A.y)

(A::AffineTransform2D{T}){T<:AbstractFloat,T1<:Real,T2<:Real}(x::T1, y::T2) =
    A(convert(T, x), convert(T, y))

(A::AffineTransform2D{T}){T<:AbstractFloat}(t::NTuple{2,T}) =
    A(t[1], t[2])

(A::AffineTransform2D{T}){T<:AbstractFloat,T1<:Real,T2<:Real}(t::Tuple{T1,T2}) =
    A(convert(T, t[1]), convert(T, t[2]))

#------------------------------------------------------------------------------
# Combine a translation with an affine transform.

# Left-translating results in translating the output of the transform.
translate{T<:AbstractFloat}(x::T, y::T, A::AffineTransform2D{T}) =
    AffineTransform2D{T}(A.xx, A.xy, A.x + x,
                         A.yx, A.yy, A.y + y)

# Right-translating results in translating the input of the transform.
translate{T<:AbstractFloat}(A::AffineTransform2D{T}, x::T, y::T) =
    AffineTransform2D{T}(A.xx, A.xy, A.xx*x + A.xy*y + A.x,
                         A.yx, A.yy, A.yx*x + A.yy*y + A.y)

translate{T<:AbstractFloat}(A::AffineTransform2D{T}, x::Real, y::Real) =
    translate(A, convert(T, x), convert(T, y))

translate{T<:AbstractFloat}(A::AffineTransform2D{T},t::NTuple{2,T}) =
    translate(A, t[1], t[2])

translate{T<:AbstractFloat,T1<:Real,T2<:Real}(A::AffineTransform2D{T}, t::Tuple{T1,T2}) =
    translate(A, convert(T, t[1]), convert(T, t[2]))

translate{T<:AbstractFloat,T1<:Real,T2<:Real}(x::T1, y::T2, A::AffineTransform2D{T}) =
    translate(convert(T, x), convert(T, y), A)

translate{T<:AbstractFloat}(t::NTuple{2,T}, A::AffineTransform2D{T}) =
    translate(t[1], t[2], A)

translate{T<:AbstractFloat,T1<:Real,T2<:Real}(t::Tuple{T1,T2}, A::AffineTransform2D{T}) =
    translate(convert(T, t[1]), convert(T, t[2]), A)

#------------------------------------------------------------------------------
"""
### Scaling an affine transform

There are two ways to combine a scaling by a factor `ρ` with an affine
transform `A`.  Left-scaling as in:
```
    B = scale(ρ, A)
```
results in scaling the output of the transform; while right-scaling as in:
```
    C = scale(A, ρ)
```
results in scaling the input of the transform.  The above examples yield
transforms which behave as:
```
    B*t = ρ*(A*t) = ρ*A(t)
    C*t = A*(ρ*t) = A(ρ*t)
```
where `t` is any 2-element tuple.
"""
function scale{T<:AbstractFloat}(ρ::T, A::AffineTransform2D{T})
    AffineTransform2D{T}(ρ*A.xx, ρ*A.xy, ρ*A.x,
                         ρ*A.yx, ρ*A.yy, ρ*A.y)
end
function scale{T<:AbstractFloat}(A::AffineTransform2D{T}, ρ::T)
    AffineTransform2D{T}(ρ*A.xx, ρ*A.xy, A.x,
                         ρ*A.yx, ρ*A.yy, A.y)
end

#------------------------------------------------------------------------------
"""
### Rotating an affine transform

There are two ways to combine a rotation by angle `θ` (in radians
counterclockwise) with an affine transform `A`.  Left-rotating as in:
```
    B = rotate(θ, A)
```
results in rotating the output of the transform; while right-rotating as in:
```
    C = rotate(A, θ)
```
results in rotating the input of the transform.  The above examples are
similar to:
```
    B = R*A
    C = A*R
```
where `R` implements rotation by angle `θ` around `(0,0)`.
"""
function rotate{T<:AbstractFloat}(θ::T, A::AffineTransform2D{T})
    cs = cos(θ)
    sn = sin(θ)
    AffineTransform2D{T}(cs*A.xx - sn*A.yx,
                         cs*A.xy - sn*A.yy,
                         cs*A.x  - sn*A.y,
                         cs*A.yx + sn*A.xx,
                         cs*A.yy + sn*A.xy,
                         cs*A.y  + sn*A.x)
end

function rotate{T<:AbstractFloat}(A::AffineTransform2D{T}, θ::T)
    cs = cos(θ)
    sn = sin(θ)
    AffineTransform2D{T}(A.xx*cs + A.xy*sn,
                         A.xy*cs - A.xx*sn,
                         A.x,
                         A.yx*cs + A.yy*sn,
                         A.yy*cs - A.yx*sn,
                         A.y)
end

for func in (:scale, :rotate)
    @eval begin
        function $func{S<:Real,T<:AbstractFloat}(q::S, A::AffineTransform2D{T})
            $func(convert(T, q), A)
        end
        function $func{S<:Real,T<:AbstractFloat}(A::AffineTransform2D{T}, q::S)
            $func(A, convert(T, q))
        end
    end
end

#------------------------------------------------------------------------------

"""
`det(A)` returns the determinant of the linear part of the affine
transform `A`.
"""
det{T<:AbstractFloat}(A::AffineTransform2D{T}) = A.xx*A.yy - A.xy*A.yx

"""
`jacobian(A)` returns the Jacobian of the affine transform `A`, that is the
absolute value of the determinant of its linear part.
"""
jacobian{T<:AbstractFloat}(A::AffineTransform2D{T}) = abs(det(A))

"""
`invert(A)` returns the inverse of the affine transform `A`.
"""
function invert{T<:AbstractFloat}(A::AffineTransform2D{T})
    d = det(A)
    d == zero(T) && error("transformation is not invertible")
    Txx =  A.yy/d
    Txy = -A.xy/d
    Tyx = -A.yx/d
    Tyy =  A.xx/d
    AffineTransform2D{T}(Txx, Txy, -Txx*A.x - Txy*A.y,
                         Tyx, Tyy, -Tyx*A.x - Tyy*A.y)
end

"""
`combine(A,B)` yields `A*B`, the affine transform which combines the two
affine transforms `A` and `B`, that is the affine transform which applies
`B` and then `A`.
"""
function combine{T<:AbstractFloat}(A::AffineTransform2D{T},
                                   B::AffineTransform2D{T})
    AffineTransform2D{T}(A.xx*B.xx + A.xy*B.yx,
                         A.xx*B.xy + A.xy*B.yy,
                         A.xx*B.x  + A.xy*B.y + A.x,
                         A.yx*B.xx + A.yy*B.yx,
                         A.yx*B.xy + A.yy*B.yy,
                         A.yx*B.x  + A.yy*B.y + A.y)
end

"""
`rightdivide(A,B)` yields `A/B`, the right division of the affine
transform `A` by the affine transform `B`.
"""
function rightdivide{T<:AbstractFloat}(A::AffineTransform2D{T},
                                       B::AffineTransform2D{T})
    d = det(B)
    d == zero(T) && error("right operand is not invertible")
    Rxx = (A.xx*B.yy - A.xy*B.yx)/d
    Rxy = (A.xy*B.xx - A.xx*B.xy)/d
    Ryx = (A.yx*B.yy - A.yy*B.yx)/d
    Ryy = (A.yy*B.xx - A.yx*B.xy)/d
    AffineTransform2D{T}(Rxx, Rxy, A.x - (Rxx*B.x + Rxy*B.y),
                         Ryx, Ryy, A.y - (Ryx*B.y + Ryy*B.y))

end

"""
`leftdivide(A,B)` yields `A\\B`, the left division of the affine
transform `A` by the affine transform `B`.
"""
function leftdivide{T<:AbstractFloat}(A::AffineTransform2D{T},
                                      B::AffineTransform2D{T})
    d = det(A)
    d == zero(T) && error("left operand is not invertible")
    Txx =  A.yy/d
    Txy = -A.xy/d
    Tyx = -A.yx/d
    Tyy =  A.xx/d
    Tx = B.x - A.x
    Ty = B.y - A.y
    AffineTransform2D{T}(Txx*B.xx + Txy*B.yx,
                         Txx*B.xy + Txy*B.yy,
                         Txx*Tx   + Txy*Ty,
                         Tyx*B.xx + Tyy*B.yx,
                         Tyx*B.xy + Tyy*B.yy,
                         Tyx*Tx   + Tyy*Ty)
end

for func in (:combine, :rightdivide, :leftdivide)
    @eval begin
        function $func{R<:AbstractFloat,
            S<:AbstractFloat}(A::AffineTransform2D{R},
                              B::AffineTransform2D{S})
            T = AffineTransform2D{promote_type(R, S)}
            $func(convert(T, A), convert(T, B))
        end
    end
end

"""
`intercept(A)` returns the tuple `(x,y)` such that `A(x,y) = (0,0)`.
"""
function intercept{T<:AbstractFloat}(A::AffineTransform2D{T})
    d = det(A)
    d == zero(T) && error("transformation is not invertible")
    return ((A.xy*A.y - A.yy*A.x)/d, (A.yx*A.x - A.xx*A.y)/d)
end


+{T<:AbstractFloat}(t::NTuple{2}, A::AffineTransform2D{T}) = translate(t, A)

+{T<:AbstractFloat}(A::AffineTransform2D{T}, t::NTuple{2}) = translate(A, t)

*{R<:AbstractFloat,S<:AbstractFloat}(A::AffineTransform2D{R},
                                     B::AffineTransform2D{S}) = combine(A, B)

*{T<:AbstractFloat}(A::AffineTransform2D{T}, t::NTuple{2}) = A(t)

*{S<:Real,T<:AbstractFloat}(ρ::S, A::AffineTransform2D{T}) = scale(ρ, A)

*{S<:Real,T<:AbstractFloat}(A::AffineTransform2D{T}, ρ::S) = scale(A, ρ)

\{R<:AbstractFloat,S<:AbstractFloat}(A::AffineTransform2D{R},
                                     B::AffineTransform2D{S}) = leftdivide(A, B)

/{R<:AbstractFloat,S<:AbstractFloat}(A::AffineTransform2D{R},
                                     B::AffineTransform2D{S}) = rightdivide(A, B)

function show{T}(io::IO, A::AffineTransform2D{T})
    println(io, typeof(A), ":")
    println(io, "  ", A.xx, "  ", A.xy, " | ", A.x)
    println(io, "  ", A.yx, "  ", A.yy, " | ", A.y)
end

function runtests()
    B = AffineTransform2D(1, 0, -3, 0.1, 1, +2)
    show(B)
    println()
    A = invert(B)
    show(A)
    println()
    C = combine(A, B)
    show(C)
    println()
    U = convert(AffineTransform2D{Float16},C)
    show(U)
    println()
    V = convert(AffineTransform2D{Float64},C)
    show(V)
    println()
    show(B(1, 4))
    println()
    show(B(1f0, 4))
    println()
    show(B(1.0, 4.0))
    println()
    show(B(1.0, 4))
    println()
    show(B((1f0, 4f0)))
    println()
    show(B((1.0, 4f0)))
    println()
    xy = intercept(B)
    xpyp = B*xy
    println("$xy --> $xpyp")
    nothing
end

end # module
