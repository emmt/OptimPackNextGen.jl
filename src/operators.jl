#
# operators.jl --
#
# Implement rules for linear operators.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

import Base: *, ⋅, ctranspose, call, diag

import ..subrange, ..dimlist

"""
`LinearOperator{OUT,INP}` is the abstract type from which inherit all linear
operators.  It is parameterized by `INP` and `OUT` respectively the input and
output types of the "vectors".  The input and output types of a linear operator
are respectively obtained by:

    input_type(A)
    output_type(A)

The methods `apply_direct` and `apply_adjoint` should be implemented for any
linear operator types:

    apply_direct(A, x)
    apply_adjoint(A, x)

respectively yield the result of applying the linear operator `A` or its
adjoint to the "vector" `x`.

The `Base.call` method and the `*` operator are overloaded so that:

    A*x  = A(x)  = call(A, x)  = apply_direct(A, x)
    A'*x = A'(x) = call(A', x) = apply_adjoint(A, x)

Although TiPi provides default versions, the methods `apply_direct!` and
`apply_adjoint!` may also be implemented:

    apply_direct!(dst, A, x)
    apply_adjoint!(dst, A, x)

which store in the destination `dst` the result of applying the operator `A` or
its adjoint to the source `src`.

"""
abstract LinearOperator{OUT,INP}


# Default methods for a linear operator:

input_type{E,F}(A::LinearOperator{E,F}) = F
output_type{E,F}(A::LinearOperator{E,F}) = E

function apply_direct{E,F}(A::LinearOperator{E,F}, x::F)
    error("method `apply_direct` not implemented for this operator")
end

function apply_direct{E,F,G}(A::LinearOperator{E,G},
                             B::LinearOperator{G,F})
    Product(A,B)
end

function apply_direct!{E,F}(dst::E, A::LinearOperator{E,F}, src::F)
    vcopy!(dst, apply_direct(A, src))
end

function apply_adjoint{E,F}(A::LinearOperator{E,F}, x::E)
    error("method `apply_adjoint` not implemented for this operator")
end

function apply_adjoint{E,F,G}(A::LinearOperator{G,E},
                              B::LinearOperator{G,F})
    Product(Adjoint(A),B)
end

function apply_adjoint!{E,F}(dst::F, A::LinearOperator{E,F}, src::E)
    vcopy!(dst, apply_adjoint(A, src))
end

@doc @doc(LinearOperator) apply_direct
@doc @doc(LinearOperator) apply_direct!
@doc @doc(LinearOperator) apply_adjoint
@doc @doc(LinearOperator) apply_adjoint!

# Overload the `call` method so that `A(x)` makes sense for `A` a linear
# operator and `x` a "vector", or another operator.

call{E,F}(A::LinearOperator{E,F}, x::F) = apply_direct(A, x) ::E

call{E,F,G}(A::LinearOperator{E,F}, B::LinearOperator{F,G}) = Product(A, B)

function call{E,F}(A::LinearOperator{E,F}, x)
    error("argument of operator has wrong type or incompatible input/output types in product of operators")
end

function apply!{E,F}(dst::F, A::LinearOperator{E,F}, src::E)
    apply_direct!(dst, A, src)
end

apply{E,F}(A::LinearOperator{E,F}, x::E) = A(x)


"""
An `Endomorphism` is a `LinearOperator` with the same input and output spaces.
"""
abstract Endomorphism{E} <: LinearOperator{E,E}

"""
A `SelfAdjointOperator` is an `Endomorphism` which is its own adjoint.
"""
abstract SelfAdjointOperator{E} <: Endomorphism{E}

apply_adjoint{E}(A::SelfAdjointOperator{E}, x::E) = apply_direct(A, x)

function apply_adjoint!{E}(dst::E, A::SelfAdjointOperator{E}, src::E)
    apply_direct!(dst, A, src)
end

"""
`Product{E,F,L,R}` is used to mark the product of two linear operators from `F`
to `E`, parameters `L` and `R` are the types of the left and right operands.
They must be such that:

    L <: LinearOperator{E,G}
    R <: LinearOperator{G,F}

for some intermediate type `G`.
"""
immutable Product{E,F,L<:LinearOperator,R<:LinearOperator} <: LinearOperator{E,F}
    lop::L
    rop::R
end

function Product{E,F,G}(A::LinearOperator{E,G}, B::LinearOperator{G,F})
    Product{E,F,typeof(A),typeof(B)}(A,B)
end

function apply_direct{E,F,G}(A::Product{E,F,G}, x::F)
    apply_direct(A.lop, apply_direct(A.rop, x))
end

function apply_direct!{E,F,G}(dst::E, A::Product{E,F,G}, src::F)
    apply_direct!(dst, A.lop, apply_direct(A.rop, x))
end

function apply_adjoint{E,F,G}(A::Product{E,F,G}, x::E)
    apply_adjoint(A.rop, apply_adjoint(A.lop, x))
end

function apply_adjoint!{E,F,G}(dst::F, A::Product{E,F,G}, src::E)
    apply_adjoint!(dst, A.rop, apply_adjoint(A.lop, x))
end


"""
`Adjoint{E,F,T}` is used to mark the adjoint of a linear operator of type
`T <: LinearOperator{E,F}`.
"""
immutable Adjoint{E,F,T<:LinearOperator} <: LinearOperator{F,E}
    op::T
end

Adjoint{E}(A::SelfAdjointOperator{E}) = A
Adjoint{E,F}(A::LinearOperator{E,F}) = Adjoint{E,F,typeof(A)}(A)
Adjoint{E,F,T}(A::Adjoint{E,F,T}) = A.op
Adjoint{E,F,L,R}(A::Product{E,F,L,R}) = Product(Adjoint(A.rop), Adjoint(A.lop))

apply_direct{E,F,T}(A::Adjoint{E,F,T}, x::E) = apply_adjoint(A.op, x)

function apply_direct!{E,F,T}(dst::F, A::Adjoint{E,F,T}, src::E)
    apply_adjoint!(dst, A.op, src)
end

apply_adjoint{E,F,T}(A::Adjoint{E,F,T}, x::F) = apply_direct(A.op, x)

function apply_adjoint!{E,F,T}(dst::E, A::Adjoint{E,F,T}, src::F)
    apply_direct!(dst, A.op, src)
end

# Overload `*` and `ctranspose` so that are no needs to overload `Ac_mul_B`
# `Ac_mul_Bc`, etc. to have `A'*x`, `A*B*C*x`, etc. yield the expected result.
ctranspose{E,F}(A::LinearOperator{E,F}) = Adjoint(A)
*{E,F}(A::LinearOperator{E,F}, x) = A(x)
⋅{E,F}(A::LinearOperator{E,F}, x) = A(x)

#------------------------------------------------------------------------------
# DIAGONAL OPERATOR

abstract AbstractDiagonalOperator{E} <: SelfAdjointOperator{E}

# FIXME: assume real weights
"""
A `DiagonalOperator` is a self-adjoint linear operator defined by a "vector"
of weights `w` as:

    W = DiagonalOperator(w)

and behaves as follows:

    W(x) = vproduct(w, x)
"""
immutable DiagonalOperator{E} <: AbstractDiagonalOperator{E}
    diag::E
end

function apply_direct{E}(A::DiagonalOperator{E}, src::E)
    vproduct(A.diag, src)
end

function apply_direct!{E}(dst::E, A::DiagonalOperator{E}, src::E)
    vproduct!(dst, A.diag, src)
end

diag{E}(A::DiagonalOperator{E}) = A.diag

#------------------------------------------------------------------------------
# IDENTITY OPERATOR

immutable Identity{T} <: AbstractDiagonalOperator{T}; end

Identity{T}(::Type{T}) = Identity{T}()

Identity() = Identity{Any}()

apply_direct{T}(::Identity{T}, x::T) = x

#const identity = Identity()

is_identity{E}(A::LinearOperator{E,E}) = is(A, Identity(E)) || is(A, Identity())

"""

    Identity(T)

yields identity operator over arguments of type `T` while

    Identity()

yields identity operator for any type of argument.
""" Identity

#------------------------------------------------------------------------------
# SCALING OPERATOR

# FIXME: assume real scale
"""
A `ScalingOperator` is a self-adjoint linear operator defined by a scalar
`alpha` to operate on "vectors" of type `E` as:

    A = ScalingOperator(alpha, E)

or

    A = ScalingOperator(E, alpha)

and behaves as follows:

    A(x) = vscale(alpha, x)

If `E` is not provided, the operator operates on any "vector":

    A = ScalingOperator(alpha)

"""
immutable ScalingOperator{E} <: AbstractDiagonalOperator{E}
    alpha::Float
    ScalingOperator{E}(::Type{E}, alpha::Float) = new(alpha)
end

ScalingOperator{E}(::Type{E}, alpha::Real) = ScalingOperator{E}(E, Float(alpha))
ScalingOperator{E}(alpha::Real, ::Type{E}) = ScalingOperator{E}(E, Float(alpha))
ScalingOperator(alpha::Real) = ScalingOperator{Any}(Any, Float(alpha))

function apply_direct{E}(A::ScalingOperator{E}, src::E)
    vscale(A.alpha, src)
end

function apply_direct!{E}(dst::E, A::ScalingOperator{E}, src::E)
    vscale!(dst, A.alpha, src)
end

#------------------------------------------------------------------------------
# RANK-1 OPERATOR

"""
A `RankOneOperator` is defined by two "vectors" `l` and `r` as:

    A = RankOneOperator(l, r)

and behaves as follows:

    A(x)  = vscale(vdot(r, x)), l)
    A'(x) = vscale(vdot(l, x)), r)
"""
immutable RankOneOperator{E,F} <: LinearOperator{E,F}
    lvect::E
    rvect::F
    RankOneOperator(lvect::E, rvect::F) = new(lvect, rvect)
end

function apply_direct{E,F}(op::RankOneOperator{E,F}, src::F)
    vscale(vdot(op.rvect, src), op.lvect)
end

function apply_direct!{E,F}(dst::E, op::RankOneOperator{E,F}, src::F)
    vscale!(dst, vdot(op.rvect, src), op.lvect)
end

function apply_adjoint{E,F}(op::RankOneOperator{E,F}, src::E)
    vscale(vdot(op.lvect, src), op.rvect)
end

function apply_adjoint!{E,F}(dst::F, op::RankOneOperator{E,F}, src::E)
    vscale!(dst, vdot(op.lvect, src), op.rvect)
end

#------------------------------------------------------------------------------
# CROPPING AND ZERO-PADDING OPERATORS

typealias Region{N} NTuple{N,UnitRange{Int}}

immutable CroppingOperator{T,N} <: LinearOperator{Array{T,N},Array{T,N}}
    outdims::NTuple{N,Int}
    inpdims::NTuple{N,Int}
    region::Region{N}
    function CroppingOperator{T,N}(::Type{T},
                                   outdims::NTuple{N,Int},
                                   inpdims::NTuple{N,Int})
        for k in 1:N
            if outdims[k] > inpdims[k]
                error("output dimensions must be smaller or equal input ones")
            end
        end
        new(outdims, inpdims, subrange(outdims, inpdims))
    end
end

function apply_direct{T,N}(A::CroppingOperator{T,N}, src::Array{T,N})
    @assert size(src) == input_size(A)
    _crop(src, A.region)
end

function apply_direct!{T,N}(dst::Array{T,N}, A::CroppingOperator{T,N},
                            src::Array{T,N})
    @assert size(src) == input_size(A)
    @assert size(dst) == output_size(A)
    _crop!(dst, src, A.region)
end

function apply_adjoint{T,N}(A::CroppingOperator{T,N}, src::Array{T,N})
    @assert size(src) == output_size(A)
    _zeropad(input_size(A), A.region, src)
end

function apply_adjoint!{T,N}(dst::Array{T,N}, A::CroppingOperator{T,N},
                             src::Array{T,N})
    @assert size(src) == output_size(A)
    @assert size(dst) == input_size(A)
    _zeropad(dst, A.region, src)
end

immutable ZeroPaddingOperator{T,N} <: LinearOperator{Array{T,N},Array{T,N}}
    outdims::NTuple{N,Int}
    inpdims::NTuple{N,Int}
    region::Region{N}
    function ZeroPaddingOperator{T,N}(::Type{T},
                                   outdims::NTuple{N,Int},
                                   inpdims::NTuple{N,Int})
        for k in 1:N
            if outdims[k] < inpdims[k]
                error("output dimensions must be larger or equal input ones")
            end
        end
        new(outdims, inpdims, subrange(inpdims, outdims))
    end
end

function apply_direct{T,N}(A::ZeroPaddingOperator{T,N}, src::Array{T,N})
    @assert size(src) == input_size(A)
    _zeropad(output_size(A), A.region, src)
end

function apply_direct!{T,N}(dst::Array{T,N}, A::ZeroPaddingOperator{T,N},
                            src::Array{T,N})
    @assert size(src) == input_size(A)
    @assert size(dst) == output_size(A)
    _zeropad!(dst, A.region, src)
end

function apply_adjoint{T,N}(A::ZeroPaddingOperator{T,N}, src::Array{T,N})
    @assert size(src) == output_size(A)
    _crop(src, A.region)
end

function apply_adjoint!{T,N}(dst::Array{T,N}, A::ZeroPaddingOperator{T,N},
                             src::Array{T,N})
    @assert size(src) == output_size(A)
    @assert size(dst) == input_size(A)
    _crop!(dst, src, A.region)
end

for Operator in (:CroppingOperator, :ZeroPaddingOperator)
    @eval begin

        function $Operator{T,N}(::Type{T},
                                outdims::NTuple{N,Integer},
                                inpdims::NTuple{N,Integer})
            $Operator{T,N}(T, dimlist(outdims), dimlist(inpdims))
        end

        function $Operator{T,N}(out::Array{T,N}, inp::Array{T,N})
            $Operator{T,N}(T, size(out), size(inp))
        end

        function $Operator{T,N}(out::Array{T,N}, inpdims::NTuple{N,Integer})
            $Operator{T,N}(T, size(out), dimlist(inpdims))
        end

        function $Operator{T,N}(outdims::NTuple{N,Integer}, inp::Array{T,N})
            $Operator{T,N}(T, dimlist(outdims), size(inp))
        end

        eltype{T,N}(A::$Operator{T,N}) = T

        input_size{T,N}(A::$Operator{T,N}) = A.inpdims

        output_size{T,N}(A::$Operator{T,N}) = A.outdims

    end
end

"""
The following methods define a cropping operator:

    C = CroppingOperator(T, outdims, inpdims)
    C = CroppingOperator(out, inp)
    C = CroppingOperator(out, inpdims)
    C = CroppingOperator(outdims, inp)

where `T` is the type of the elements of the arrays transformed by the
operator, `outdims` (resp. `inpdims`) are the dimensions of the output
(resp. input) of the operator, `out` (resp. `inp`) is a template array which
represents the structure of the output (resp. input) of the operator.  That is
to say, `T = eltype(out)` and `outdims = size(out)` (resp. `T = eltype(inp)`
and `inpdims = size(inp)`).

The adjoint of a cropping operator is a zero-padding operator which can be
defined by one of:

    Z = ZeroPaddingOperator(T, outdims, inpdims)
    Z = ZeroPaddingOperator(out, inp)
    Z = ZeroPaddingOperator(out, inpdims)
    Z = ZeroPaddingOperator(outdims, inp)

### See Also
zeropad, crop.

""" CroppingOperator

@doc @doc(CroppingOperator) ZeroPaddingOperator

_crop{T,N}(src::Array{T,N}, region::Region{N}) = src[region...]

function _crop!{T,N}(dst::Array{T,N}, src::Array{T,N}, region::Region{N})
    vcopy!(dst, src[region...])
end

function _zeropad{T,N}(dims::NTuple{N,Int}, region::Region{N}, src::Array{T,N})
    dst = Array(T, dims)
    _zeropad!(dst, region, src)
    return dst
end

function _zeropad!{T,N}(dst::Array{T,N}, region::Region{N}, src::Array{T,N})
    fill!(dst, zero(T))
    dst[region...] = src
end

#------------------------------------------------------------------------------
# CHECKING OPERATOR

function check_operator{E,F}(y::E, A::LinearOperator{E,F}, x::F)
    v1 = vdot(y, A*x)
    v2 = vdot(A'*y, x)
    (v1, v2, v1 - v2)
end

function check_operator{T<:AbstractFloat,M,N}(outdims::NTuple{M,Int},
                                              A::LinearOperator{Array{T,M},
                                                                Array{T,N}},
                                              inpdims::NTuple{N,Int})
    check_operator(Array{T}(randn(outdims)), A, Array{T}(randn(inpdims)))
end

"""
    (v1, v2, v1 - v2) = check_operator(y, A, x)

yields `v1 = vdot(y, A*x)`, `v2 = vdot(A'*y, x)` and their difference for `A` a
linear operator, `y` a "vector" of the output space of `A` and `x` a "vector"
of the input space of `A`.  In principle, the two inner products should be the
same whatever `x` and `y`; otherwise the operator has a bug.

Simple linear operators operating on Julia `Array` can be tested on random
"vectors" with:

    (v1, v2, v1 - v2) = check_operator(outdims, A, inpdims)

with `outdims` and `outdims` the dimensions of the output and input "vectors"
for `A`.  The element type is automatically guessed from the type of `A`.

""" check_operator

#------------------------------------------------------------------------------
