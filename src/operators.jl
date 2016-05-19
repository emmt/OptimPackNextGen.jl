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
A `SelfAdjointOperator` is a `LinearOperator` which is its own adjoint.
"""
abstract SelfAdjointOperator{E} <: LinearOperator{E,E}

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

#------------------------------------------------------------------------------
