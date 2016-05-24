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

import Base: *, ⋅, ctranspose, call

"""
`LinearOperator{OUT,INP}` is the abstract type from which inherit all linear
operators.  It is parameterized by `INP` and `OUT` respectively the input and
output types of the "vectors".
"""
abstract LinearOperator{OUT,INP}

"""
An `Endomorphism` is a `LinearOperator` with the same input and output spaces.
"""
abstract Endomorphism{E} <: LinearOperator{E,E}

"""
A `SelfAdjointOperator` is an `Endomorphism` which is its own adjoint.
"""
abstract SelfAdjointOperator{E} <: Endomorphism{E}

"""
`Adjoint{E,F,T}` is used to mark the adjoint of a linear operator of type
`T <: LinearOperator{E,F}`.
"""
immutable Adjoint{E,F,T<:LinearOperator} <: LinearOperator{F,E}
    op::T
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

Adjoint{E}(A::SelfAdjointOperator{E}) = A
Adjoint{E,F}(A::LinearOperator{E,F}) = Adjoint{E,F,typeof(A)}(A)
Adjoint{E,F,T}(A::Adjoint{E,F,T}) = A.op
Adjoint{E,F,L,R}(A::Product{E,F,L,R}) = Product(Adjoint(A.rop), Adjoint(A.lop))

function Product{E,F,G}(A::LinearOperator{E,G},
                        B::LinearOperator{G,F})
    Product{E,F,typeof(A),typeof(B)}(A,B)
end


# Overload the `call` method so that `A(x)` makes sense for `A` a linear
# operator or a product of linear operators and `x` a "vector", or another
# operator.
call{E,F}(A::LinearOperator{E,F}, x::F) = apply_direct(A, x) ::E
call{E,F,T}(A::Adjoint{E,F,T}, x::E) = apply_adjoint(A.op, x) ::F
call{E,F,G}(A::LinearOperator{E,F}, B::LinearOperator{F,G}) = Product(A, B)
call{E,F,L,R}(A::Product{E,F,L,R}, x::F) = A.lop(A.rop(x))
function call{E,F}(A::LinearOperator{E,F}, x)
    error("argument of operator has wrong type or incompatible input/output types in product of operators")
end

#call(Adjoint{_<:LinearOperator{G<:Any, H<:Any}, #F<:Any, #T<:Any}, _<:LinearOperator{#G<:Any, #H<:Any})

*{E,F}(A::LinearOperator{E,F}, x) = call(A, x)

# Overload `*` and `ctranspose` so that are no needs to overload `Ac_mul_B`
# `Ac_mul_Bc`, etc. to have `A'*x`, `A*B*C*x`, etc. yield the expected result.

ctranspose{E,F}(A::LinearOperator{E,F}) = Adjoint(A)
*{E,F}(A::LinearOperator{E,F}, x) = A(x)
⋅{E,F}(A::LinearOperator{E,F}, x) = A(x)

@doc """

    apply_direct(A, x)

yields the result of applying the linear operator `A` to the "vector" `x`.
To be used, this method has to be provided for each types derived from
`LinearOperator`.  This method is the one called in the following cases:

    A*x
    A(x)
    call(A, x)

"""
function apply_direct{E,F}(A::LinearOperator{E,F}, x::F)
    error("method `apply_direct` not implemented for this operator")
end

@doc """

    apply_adjoint(A, x)

yields the result of applying the adjoint of the linear operator `A` to the
"vector" `x`.  To be used, this method has to be provided for each non-self
adjoint types derived from `LinearOperator`.  This method is the one called in
the following cases:

    A'*x
    A'(x)
    call(A', x)

"""
function apply_adjoint{E,F}(A::LinearOperator{E,F}, x::E)
    error("method `apply_adjoint` not implemented for this operator")
end

apply_adjoint{E,F,T}(A::Adjoint{E,F,T}, x::F) = apply_direct(A.op, x)
apply_adjoint{E}(A::SelfAdjointOperator{E}, x::E) = apply_direct(A, x)

immutable Identity{T} <: SelfAdjointOperator{T}; end
Identity{T}(::Type{T}) = Identity{T}()
apply_direct{T}(::Identity{T}, x::T) = x

@doc """

    apply_direct!(dst, A, src)

stores in the destination "vector" `dst` the result of applying the linear
operator `A` to the source "vector" `src`.
"""
function apply_direct!{E,F}(dst::E, A::LinearOperator{E,F}, src::F)
    vcopy!(dst, A(src))
end

@doc """

    apply_adjoint!(dst, A, src)

stores in the destination "vector" `dst` the result of applying the adjoint of
the linear operator `A` to the source "vector" `src`.
"""
function apply_adjoint!{E,F}(dst::F, A::LinearOperator{E,F}, src::E)
    vcopy!(dst, A'(src))
end

function apply!{E,F}(dst::F, A::LinearOperator{E,F}, src::E)
    apply_direct!(dst, A, src)
end

function apply!{E,F,T}(dst::E, A::Adjoint{E,F,T}, src::F)
    apply_adjoint!(dst, A.op, src)
end

apply{E,F}(A::LinearOperator{E,F}, x::E) = A(x)

#------------------------------------------------------------------------------

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

"""
#------------------------------------------------------------------------------

immutable Rank1Operator{E,F} <: LinearOperator{E,F}
    lvect::E
    rvect::F
    Rank1Operator(lvect::E, rvect::F) = new(lvect, rvect)
end

function apply_direct{E,F}(op::Rank1Operator{E,F}, src::F)
    vscale(vdot(op.rvect, src), op.lvect)
end

function apply_direct!{E,F}(dst::E, op::Rank1Operator{E,F}, src::F)
    vscale!(dst, vdot(op.rvect, src), op.lvect)
end

function apply_adjoint{E,F}(op::Rank1Operator{E,F}, src::E)
    vscale(vdot(op.lvect, src), op.rvect)
end

function apply_adjoint!{E,F}(dst::F, op::Rank1Operator{E,F}, src::E)
    vscale!(dst, vdot(op.lvect, src), op.rvect)
end

#------------------------------------------------------------------------------

# FIXME: assume real weights
immutable WeightingOperator{E} <: SelfAdjointOperator{E}
    w::E
    WeightingOperator{E}(w::E) = new(w)
end

function apply_direct{E}(op::WeightingOperator{E}, src::E)
    vproduct(op.w, src)
end

function apply_direct!{E}(dst::E, op::WeightingOperator{E}, src::E)
    vproduct!(dst, op.w, src)
end

#------------------------------------------------------------------------------
