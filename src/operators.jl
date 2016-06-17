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

# FIXME: fix doc. self-adjoint, endomorphism, etc. ScalingOperator --> Scaled(Identity)

import Base: *, ⋅, +, -, \, /, ctranspose, call, diag, inv, apply, A_mul_B!,
             show, showerror

import TiPi: subrange, dimlist, contents

immutable UnimplementedOperation <: Exception end

showerror(io::IO, e::UnimplementedOperation) =
    print(io, "attempt to apply unimplemented operation")

unimplemented_operation(args...) = throw(UnimplementedOperation())

#------------------------------------------------------------------------------
# ABSTRACT TYPES

# Traits are part of the signature of an operators.  For now that's a bitwise
# combination.
const Nonlinear = 0
const Linear    = 1

abstract Operator{T,E,F} # T = traits, E = output space, F = input space
typealias NonlinearOperator{E,F}   Operator{Nonlinear,E,F}
typealias LinearOperator{E,F}      Operator{Linear,E,F}
typealias NonlinearEndomorphism{E} NonlinearOperator{E,E}
typealias LinearEndomorphism{E}    LinearOperator{E,E}
typealias SelfAdjointOperator{E}   LinearEndomorphism{E}

# Type to store the product of two operators.
immutable Product{T,E,F} <: Operator{T,E,F}
    lop::Operator
    rop::Operator
end

# Type to store the product of an operator by a scalar.
immutable Scaled{T,E,F} <: Operator{T,E,F}
    sc::Float
    op::Operator{T,E,F}
end

# Type to store the linear combination of two operators (preferred over storage
# of a simple `Addition` of `Scaled` operators because it corresponds to the
# `vcombine!` method with no intermediate temporaries).
immutable Combination{T,E,F} <: Operator{T,E,F}
    lsc::Float
    lop::Operator
    rsc::Float
    rop::Operator
end

# Type to store the adjoint of a linear operator, as in the expression `A'`.
immutable Adjoint{E,F} <: LinearOperator{E,F}
    op::LinearOperator{F,E}
end

# Type to represent the inverse of an operator.  For instance, 'A\B` yields
# `Inverse(A)*B`.
immutable Inverse{T,E,F} <: Operator{T,E,F}
    op::Operator{T,F,E}
end

doc"""
# Linear Combination of Operators

    combine(α, A, β, B)

yields an operator with performs as `α*A + β*B`.  Parameters `α` and `β` are
scalars, parameters `A` and `B` are operators which must have the same input
and output types.

"""
function combine end

for (L, lsc, lop) in     ((:Scaled,   :(alpha*A.sc), :(A.op)),
                          (:Operator, :alpha,        :A))
    for (R, rsc, rop) in ((:Scaled,   :(beta*B.sc),  :(B.op)),
                          (:Operator, :beta,         :B))
        @eval function combine{Ta,Tb,E,F}(alpha::Real, A::$L{Ta,E,F},
                                          beta::Real, B::$R{Tb,E,F})
            Combination{Ta&Tb,E,F}($lsc, $lop, $rsc, $rop)
        end
    end
end
combine{Ta,Tb,Ea,Eb,F}(::Real, ::Operator{Ta,Ea,F}, ::Real, ::Operator{Tb,Eb,F}) =
    throw(ArgumentError("cannot add/subtract operators with different input types"))
combine{Ta,Tb,E,Fa,Fb}(::Real, ::Operator{Ta,E,Fa}, ::Real, ::Operator{Tb,E,Fb}) =
    throw(ArgumentError("cannot add/subtract operators with different ouput types"))
combine{Ta,Tb,Ea,Eb,Fa,Fb}(::Real, ::Operator{Ta,Ea,Fa}, ::Real, ::Operator{Tb,Eb,Fb}) =
    throw(ArgumentError("cannot add/subtract operators with different input and ouput types"))

# Methods for multiplying, scaling, combining, etc., the operators.  To allow
# for some simplications to occur at compile time (FIXME: check this), the
# scale factor is factorized outside of the construction while the adjoint is
# propagated inside the construction.
#
#
# Ultimately, I want something like:
#
#     A = H'*W*H + µ*D'*D
#
# to work properly, in particular:
#
#     A*x -> H'*(W*(H*x)) + µ*(D'*(D*x))
#
# with parenthesis to indicate order of operations and
#
#     A' === A
#
# should be true.
#
# Outer constructors are not provided, instead there are methods like `combine`
# which filter the result and perform simplifications.

# Product of two operators.
for (L, R, sc, lop, rop) in
    ((:Scaled,   :Scaled,   :(A.sc*B.sc), :(A.op), :(B.op)),
     (:Operator, :Scaled,   :(B.sc),      :A,      :(B.op)),
     (:Scaled,   :Operator, :(A.sc),      :(A.op), :B))

    @eval *{Ta,Tb,E,F,G}(A::$L{Ta,E,G}, B::$R{Tb,G,F}) = ($sc)*(($lop)*($rop))
end
*{Ta,Tb,E,F,G}(A::Operator{Ta,E,G}, B::Operator{Tb,G,F}) =
    Product{Ta&Tb,E,F}(A, B)

# Left scalar muliplication of an operator (using parenthesis to emphasize
# order of operations).
*(alpha::Real, A::Scaled) = (alpha*A.sc)*(A.op)
*(alpha::Real, A::Combination) =
    combine(alpha*A.lsc, A.lop, alpha*A.rsc, A.rop)
*{R<:Real,T,E,F}(alpha::R, A::Operator{T,E,F}) =
    (alpha == one(R) ? A : Scaled{T,E,F}(alpha, A))

# Right vector muliplication.
*{T,E,F}(A::Operator{T,E,F}, x::F) = apply_direct(A, x) ::E

# Unary minus and unary plus.
-(A::Operator) = -1*A
+(A::Operator) = A

# Dot notation.
⋅(A::Operator, x) = A*x

# Right scalar division of an operator.
/(A::Scaled, alpha::Real) = (A.sc/alpha)*A.op
/{T<:Real}(A::Combination, alpha::T) =
    (alpha == one(T) ? A : combine(A.lsc/alpha, A.lop, A.rsc/alpha, A.rop))
/{T,E,F,R<:Real}(A::Operator{T,E,F}, alpha::R) =
    (alpha == one(R) ? A : Scaled{T,E,F}(1/alpha, A))

# For the `+` and `-` binary operators, we rely on the `combine` methods.
+(A::Operator, B::Operator) = combine(1, A,  1, B)
-(A::Operator, B::Operator) = combine(1, A, -1, B)

# The adjoint of a linear operator is implemented via the `ctranspose` method.
ctranspose(A::Adjoint) = A.op
ctranspose(A::Scaled{Linear}) = (A.sc)*(A.op')
ctranspose(A::Product{Linear}) = (A.rop')*(A.lop')
ctranspose(A::Combination{Linear}) = combine(A.lsc, (A.lop'), A.rsc, (A.rop'))
ctranspose{E,F}(A::LinearOperator{E,F}) = Adjoint{F,E}(A)
ctranspose(::Operator) =
    throw(ArgumentError("cannot take the adjoint of a nonlinear operator"))

# The inverse of an operator is implemented via the `inv` method and the '\'
# binary operator.  It is assumed that the inverse exists, thus `inv(inv(A))`
# yields `A`. (FIXME: is this really a good idea?)
inv(A::Inverse) = A.op
inv(A::Scaled{Linear}) = (1/A.sc)*inv(A.op)
inv{T,E,F}(A::Operator{T,E,F}) = Inverse{T,F,E}(A.op)
\{T,E,F}(A::Operator{T,E,F}, x::E) = apply_inverse(A, x)
\{Ta,Tb,E,F,G}(A::Operator{Ta,E,F}, B::Operator{Tb,E,G}) = inv(A)*B
\(::Operator, ::Operator) = throw(ArgumentError("type mismath in left division of operators"))
\(::Operator, ::Any) = throw(ArgumentError("bad argument type for inverse operator"))


# Overload `call` and `A_mul_B!` so that are no needs to overload `Ac_mul_B`
# `Ac_mul_Bc`, etc. to have `A'*x`, `A*B*C*x`, etc. yield the expected result.
call(A::Operator, x::Any) = A*x

A_mul_B!{T,E,F}(y::E, A::Operator{T,E,F}, x::F) = apply_direct!(y, A, x) ::E
A_mul_B!{T,E,F}(::Any, ::Operator{T,E,F}, ::F) =
    throw(ArgumentError("bad output type for calling operator"))
A_mul_B!{T,E,F}(::E, ::Operator{T,E,F}, ::Any) =
    throw(ArgumentError("bad input type for calling operator"))
A_mul_B!{T,E,F}(::Any, ::Operator{T,E,F}, ::F) =
    throw(ArgumentError("bad output and input types for calling operator"))


is_self_adjoint{T,E,F}(A::Operator{T,E,F}) = (A' === A)

is_linear{T,E,F}(::Operator{T,E,F}) = ((T|Linear) == Linear)

input_type{T,E,F}(::Operator{T,E,F}) = F
output_type{T,E,F}(::Operator{T,E,F}) = E

function apply_direct end
function apply_direct! end

function apply_inverse end
function apply_inverse! end

function apply_adjoint end
function apply_adjoint! end

function apply_inverse_adjoint end
function apply_inverse_adjoint! end

# Implement `apply_*` methods for `Adjoint` and `Inverse` subtypes.
for (class, direct, adjoint, inverse, inverse_adjoint) in
    ((:Adjoint, :adjoint, :direct, :inverse_adjoint, :inverse),
     (:Inverse, :inverse, :inverse_adjoint, :direct, :adjoint))

    @eval apply_direct{E,F}(A::$class{E,F}, x::F) =
        $(symbol("apply_", direct))(A.op, x)
    @eval apply_adjoint{E,F}(A::$class{E,F}, x::E) =
        $(symbol("apply_", adjoint))(A.op, x)
    @eval apply_inverse{E,F}(A::$class{E,F}, x::E) =
        $(symbol("apply_", inverse))(A.op, x)
    @eval apply_inverse_adjoint{E,F}(A::$class{E,F}, x::F) =
        $(symbol("apply_", inverse_adjoint))(A.op, x)

    @eval apply_direct!{E,F}(y::E, A::$class{E,F}, x::F) =
        $(symbol("apply_", direct, "!"))(y, A.op, x)
    @eval apply_adjoint!{E,F}(y::F, A::$class{E,F}, x::E) =
        $(symbol("apply_", adjoint, "!"))(y, A.op, x)
    @eval apply_inverse!{E,F}(y::F, A::$class{E,F}, x::E) =
        $(symbol("apply_", inverse, "!"))(y, A.op, x)
    @eval apply_inverse_adjoint!{E,F}(y::E, A::$class{E,F}, x::F) =
        $(symbol("apply_", inverse_adjoint, "!"))(y, A.op, x)
end


# Implement `apply_*` methods for other subtypes.

apply_direct{T,E,F}(A::Product{T,E,F}, x::F) =
    apply_direct(A.lop, apply_direct(A.rop, x))
apply_adjoint{T,E,F}(A::Product{T,E,F}, x::E) =
    apply_adjoint(A.rop, apply_adjoint(A.lop, x))
apply_inverse{T,E,F}(A::Product{T,E,F}, x::E) =
    apply_inverse(A.rop, apply_inverse(A.lop, x))
apply_inverse_adjoint{T,E,F}(A::Product{T,E,F}, x::F) =
    apply_inverse_adjoint(A.lop, apply_inverse_adjoint(A.rop, x))

apply_direct!{T,E,F}(y::E, A::Product{T,E,F}, x::F) =
    apply_direct!(y, A.lop, apply_direct(A.rop, x))
apply_adjoint!{T,E,F}(y::F, A::Product{T,E,F}, x::E) =
    apply_adjoint!(y, A.rop, apply_adjoint(A.lop, x))
apply_inverse!{T,E,F}(A::Product{T,E,F}, x::E) =
    apply_inverse!(y, A.rop, apply_inverse(A.lop, x))
apply_inverse_adjoint!{T,E,F}(A::Product{T,E,F}, x::F) =
    apply_inverse_adjoint!(y, A.lop, apply_inverse_adjoint(A.rop, x))


apply_direct{T,E,F}(A::Scaled{T,E,F}, x::F) =
    vscale!(apply_direct(A.op, x), A.sc) ::E

apply_direct{T,E,F}(A::Combination{T,E,F}, x::F) =
    vcombine(A.lsc, apply_direct(A.lop, x), A.rsc, apply_direct(A.rop, x)) ::E


# Default methods for a linear operator when destination is given:
apply_direct!{T,E,F}(y::E, A::Operator{T,E,F}, x::F) =
    vcopy!(y, apply_direct(A, x))
apply_adjoint!{E,F}(y::F, A::LinearOperator{E,F}, x::E) =
    vcopy!(y, apply_adjoint(A, x))
apply_inverse!{T,E,F}(y::F, A::Operator{T,E,F}, x::E) =
    vcopy!(y, apply_inverse(A, x))
apply_inverse_adjoint!{E,F}(y::E, A::LinearOperator{E,F}, x::F) =
    vcopy!(y, apply_inverse_adjoint(A, x))

# Illegal arguments.
apply_direct{T,E,F,G}(::Operator{T,E,F}, ::G) =
    throw(ArgumentError("invalid input type for applying operator"))
apply_adjoint{T,E,F,G}(::Operator{T,E,F}, ::G) =
    throw(ArgumentError("invalid input type for applying adjoint of operator"))
apply_inverse{T,E,F,G}(::Operator{T,E,F}, ::G) =
    throw(ArgumentError("invalid input type for applying inverse of operator"))
apply_inverse_adjoint{T,E,F,G}(::Operator{T,E,F}, ::G) =
    throw(ArgumentError("invalid input type for applying inverse adjoint of operator"))

apply_direct!{T,E,F,G,H}(::G, ::Operator{T,E,F}, ::H) =
    throw(badtype(G === E, H === F, "operator"))
apply_adjoint!{T,E,F,G,H}(::G, ::Operator{T,E,F}, ::H) =
    throw(badtype(G === F, H === E, "adjoint of operator"))
apply_inverse!{T,E,F,G,H}(::G, ::Operator{T,E,F}, ::H) =
    throw(badtype(G === F, H === E, "inverse of operator"))
apply_inverse_adjoint!{T,E,F,G,H}(::G, ::Operator{T,E,F}, ::H) =
    throw(badtype(G === E, H === F, "inverse adjoint of operator"))

function badtype(out::Bool, inp::Bool, oper::ASCIIString)
    ArgumentError("invalid " * (inp ? "output type" :
                                out ? "input type" :
                                "input and output types") *
                  " for applying " * oper)
end


#------------------------------------------------------------------------------
# OPERATORS

doc"""
# Operators

`Operator{OUT,INP}` is the abstract type from which inherit all operators, an
operator is like a function which takes a single argument (the "input") and
produces a single result (the "output") whose types are known in advance.  An
operator is is parameterized by `OUT` and `INP` respectively the output and
input types.  The input and output types of an operator are respectively
obtained by:

    input_type(A)
    output_type(A)

The methods `apply` and `apply!` are assumed to be implemented for any
operator types:

    apply(A, x) -> y
    apply!(y, A, x) ->y

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
its adjoint to the source `src`.  It is assumed that `dst` is returned by
these methods.

Finally, if a linear operator `A` is invertible, it shall implement the
following methods:

    apply_inverse(A, x)
    apply_inverse!(dst, A, x)

and (unless `A` is self-adjoint):

    apply_inverse_adjoint(A, x)
    apply_inverse_adjoint!(dst, A, x)

Remember that it is assumed that the destination `dst` is returned by the
`apply_...!` methods which take this argument.

""" Operator

# Basic methods:
function input_type end
function input_eltype end
function input_size end
function input_ndims end
function output_type end
function output_eltype end
function output_size end
function output_ndims end
doc"""
The calls:

    input_type(A)
    output_type(A)

yield the type of the input argument and of the output of the operator `A`.  If
`A` operates on Julia arrays, the element type, list of dimensions, `i`-th
dimension and number of dimensions for the input and output are given by:

    input_eltype(A)          output_eltype(A)
    input_size(A)            output_size(A)
    input_size(A, i)         output_size(A, i)
    input_ndims(A)           output_ndims(A)

""" input_type
@doc @doc(input_type) input_eltype
@doc @doc(input_type) input_size
@doc @doc(input_type) input_ndims
@doc @doc(input_type) output_type
@doc @doc(input_type) output_eltype
@doc @doc(input_type) output_size
@doc @doc(input_type) output_ndims

input_type{E,F}(::Operator{E,F}) = F
output_type{E,F}(::Operator{E,F}) = E
for f in (:input_eltype, :output_eltype,
          :input_ndims,  :output_ndims)
    @eval $f(::Operator) =
        error($(string("method `",f,"` not implemented by this operator")))
end
for f in (:input_size, :output_size)
    @eval $f(::Operator) =
        error($(string("method `",f,"` not implemented by this operator")))
    @eval $f(::Operator, ::Integer) =
        error($(string("method `",f,"` not implemented by this operator")))
end

#------------------------------------------------------------------------------
# LINEAR OPERATORS

doc"""
# Linear Operators

 `LinearOperator{OUT,INP}` is the abstract type from which inherit all
linear operators.  It is parameterized by `OUT` and `INP` respectively the
output and input types of the "vectors" onto which operate the linear operator.
The input and output types of a linear operator are respectively obtained by:

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
its adjoint to the source `src`.  It is assumed that `dst` is returned by
these methods.

Finally, if a linear operator `A` is invertible, it shall implement the
following methods:

    apply_inverse(A, x)
    apply_inverse!(dst, A, x)

and (unless `A` is self-adjoint):

    apply_inverse_adjoint(A, x)
    apply_inverse_adjoint!(dst, A, x)

Remember that it is assumed that the destination `dst` is returned by the
`apply_...!` methods which take this argument.

""" LinearOperator

# Declare basic methods and link their documentation.
function apply end
function apply! end
function apply_direct end
function apply_direct! end
function apply_adjoint end
function apply_adjoint! end
function apply_inverse end
function apply_inverse! end
function apply_inverse_adjoint end
function apply_inverse_adjoint! end
@doc @doc(LinearOperator) apply
@doc @doc(LinearOperator) apply!
@doc @doc(LinearOperator) apply_direct
@doc @doc(LinearOperator) apply_direct!
@doc @doc(LinearOperator) apply_adjoint
@doc @doc(LinearOperator) apply_adjoint!
@doc @doc(LinearOperator) apply_inverse
@doc @doc(LinearOperator) apply_inverse!
@doc @doc(LinearOperator) apply_inverse_adjoint
@doc @doc(LinearOperator) apply_inverse_adjoint!

# Provide basic methods for inverse and adjoint:
for class in (:Adjoint, :Inverse)
    for f in (:type, :eltype, :size, :ndims)
        inpf = symbol("input_",f)
        outf = symbol("output_",f)
        @eval $inpf(A::$class) = $outf(A.op)
        @eval $outf(A::$class) = $inpf(A.op)
    end
    @eval input_size(A::$class, i::Integer) = output_size(A,i)
    @eval output_size(A::$class, i::Integer) = input_size(A,i)
end


#------------------------------------------------------------------------------
# FAKE OPERATORS

doc"""

Fake linear operators are sometimes needed as placeholders for optional
arguments or keywords in methods to indicate that their value has not
been specified while keeping a specific signature.  They came in several
flavors:

* `FakeLinearOperator{E,F}` inherits from `LinearOperator{E,F}`;

* `FakeLinearEndomorphism{E}` inherits from `LinearEndomorphism{E}`;

* `FakeSelfAdjointOperator{E}` inherits from `SelfAdjointOperator{E}`;

To check whether a linear operator, say `A`, is a "fake" one:

    is_fake(A)

"""
immutable FakeLinearOperator{E,F} <: LinearOperator{E,F} end

FakeLinearOperator{E,F}(::Type{E},::Type{F}) =
    FakeLinearOperator{E,F}()

immutable FakeOperatorException <: Exception end

showerror(io::IO, e::FakeOperatorException) =
    print(io, "attempt to apply fake linear operator")

apply_direct{E,F}(::FakeLinearOperator{E,F}, ::F) =
    throw(FakeOperatorException())
apply_direct!{E,F}(::E, ::FakeLinearOperator{E,F}, ::F) =
    throw(FakeOperatorException())
apply_adjoint{E,F}(::FakeLinearOperator{E,F}, ::E) =
    throw(FakeOperatorException())
apply_adjoint!{E,F}(::F, ::FakeLinearOperator{E,F}, ::E) =
    throw(FakeOperatorException())
apply_inverse{E,F}(::FakeLinearOperator{E,F}, ::E) =
    throw(FakeOperatorException())
apply_inverse!{E,F}(::F, ::FakeLinearOperator{E,F}, ::E) =
    throw(FakeOperatorException())
apply_inverse_adjoint{E,F}(::FakeLinearOperator{E,F}, ::F) =
    throw(FakeOperatorException())
apply_inverse_adjoint!{E,F}(::E, ::FakeLinearOperator{E,F}, ::F) =
    throw(FakeOperatorException())

is_fake(::FakeLinearOperator) = true
is_fake(::LinearOperator) = false

@doc @doc(FakeLinearOperator) is_fake

#------------------------------------------------------------------------------
# DIAGONAL OPERATOR

# FIXME: assume real weights
"""
A `DiagonalOperator` is a self-adjoint linear operator defined by a "vector"
of weights `w` as:

    W = DiagonalOperator(w)

and behaves as follows:

    W(x) = vproduct(w, x)
"""
immutable DiagonalOperator{E} <: LinearEndomorphism{E}
    diag::E
end

contents(A::DiagonalOperator) = A.diag

diag(A::DiagonalOperator) = A.diag

# FIXME: add rules so that A' = A or reintroduce inheritage
# FIXME: implement vdivide and apply_inverse
apply_direct{E}(A::DiagonalOperator{E}, src::E) =
    vproduct(A.diag, src)

apply_direct!{E}(dst::E, A::DiagonalOperator{E}, src::E) =
    vproduct!(dst, A.diag, src)

for f in (:eltype, :ndims, :size, :type)
    @eval $(Symbol(string("input_",f)))(A::DiagonalOperator) = $f(A.diag)
    @eval $(Symbol(string("output_",f)))(A::DiagonalOperator) = $f(A.diag)
end
input_size(A::DiagonalOperator, i::Integer) = size(A.diag, i)
output_size(A::DiagonalOperator, i::Integer) = size(A.diag, i)

#------------------------------------------------------------------------------
# IDENTITY OPERATOR

immutable Identity{T} <: LinearEndomorphism{T}; end

Identity{T}(::Type{T}) = Identity{T}()

Identity() = Identity{Any}()

# FIXME: add rules so that A' = A or reintroduce inheritage
# FIXME: implement vdivide and apply_inverse
for op in (:direct, :inverse)
    @eval $(Symbol(string("apply_",op))){T}(::Identity{T}, x::T) = x
    @eval $(Symbol(string("apply_",op,"!"))){T}(y::T, ::Identity{T}, x::T) =
        vcopy!(y, x)
end

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
immutable ScalingOperator{E} <: LinearEndomorphism{E}
    alpha::Float
end

ScalingOperator{E}(::Type{E}, alpha::Real) = ScalingOperator{E}(alpha)
ScalingOperator{E}(alpha::Real, ::Type{E}) = ScalingOperator{E}(alpha)
ScalingOperator(alpha::Real) = ScalingOperator{Any}(Any, Float(alpha))

contents(A::ScalingOperator) = A.alpha

apply_direct{E}(A::ScalingOperator{E}, src::E) =
    vscale(A.alpha, src)

apply_direct!{E}(dst::E, A::ScalingOperator{E}, src::E) =
    vscale!(dst, A.alpha, src)

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

apply_direct{E,F}(op::RankOneOperator{E,F}, src::F) =
    vscale(vdot(op.rvect, src), op.lvect)

apply_direct!{E,F}(dst::E, op::RankOneOperator{E,F}, src::F) =
    vscale!(dst, vdot(op.rvect, src), op.lvect)

apply_adjoint{E,F}(op::RankOneOperator{E,F}, src::E) =
    vscale(vdot(op.lvect, src), op.rvect)

apply_adjoint!{E,F}(dst::F, op::RankOneOperator{E,F}, src::E) =
    vscale!(dst, vdot(op.lvect, src), op.rvect)

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

for class in (:CroppingOperator, :ZeroPaddingOperator)
    @eval begin

        function $class{T,N}(::Type{T},
                                outdims::NTuple{N,Integer},
                                inpdims::NTuple{N,Integer})
            $class{T,N}(T, dimlist(outdims), dimlist(inpdims))
        end

        function $class{T,N}(out::Array{T,N}, inp::Array{T,N})
            $class{T,N}(T, size(out), size(inp))
        end

        function $class{T,N}(out::Array{T,N}, inpdims::NTuple{N,Integer})
            $class{T,N}(T, size(out), dimlist(inpdims))
        end

        function $class{T,N}(outdims::NTuple{N,Integer}, inp::Array{T,N})
            $class{T,N}(T, dimlist(outdims), size(inp))
        end

        eltype{T,N}(A::$class{T,N}) = T

        input_size{T,N}(A::$class{T,N}) = A.inpdims

        output_size{T,N}(A::$class{T,N}) = A.outdims

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
# OPERATORS BUILT ON TOP OF FUNCTIONS

immutable FunctionalLinearOperator{E,F} <: LinearOperator{E,F}
    direct::Function
    direct!::Function
    adjoint::Function
    adjoint!::Function
    inverse::Function
    inverse!::Function
    inverse_adjoint::Function
    inverse_adjoint!::Function
end

function FunctionalLinearOperator{E,F}(::Type{E}, ::Type{F};
                                       direct::Function=unimplemented_operation,
                                       direct!::Function=unimplemented_operation,
                                       adjoint::Function=unimplemented_operation,
                                       adjoint!::Function=unimplemented_operation,
                                       inverse::Function=unimplemented_operation,
                                       inverse!::Function=unimplemented_operation,
                                       inverse_adjoint::Function=unimplemented_operation,
                                       inverse_adjoint!::Function=unimplemented_operation)
    FunctionalLinearOperator{E,F}(direct, direct!, adjoint, adjoint!, inverse,
                                  inverse!, inverse_adjoint, inverse_adjoint!)
end

apply_direct{E,F}(A::FunctionalLinearOperator{E,F}, x::F) =
    A.direct(x)

apply_direct!{E,F}(y::E, A::FunctionalLinearOperator{E,F}, x::F) =
    A.direct!(y, x)

apply_adjoint{E,F}(A::FunctionalLinearOperator{E,F}, x::E) =
    A.adjoint(x)

apply_adjoint!{E,F}(y::F, A::FunctionalLinearOperator{E,F}, x::E) =
    A.adjoint!(y, x)

apply_inverse{E,F}(A::FunctionalLinearOperator{E,F}, x::E) =
    A.inverse(x)

apply_inverse!{E,F}(y::F, A::FunctionalLinearOperator{E,F}, x::E) =
    A.inverse!(y, x)

apply_inverse_adjoint{E,F}(A::FunctionalLinearOperator{E,F}, x::F) =
    A.inverse_adjoint(x)

apply_inverse_adjoint!{E,F}(y::E, A::FunctionalLinearOperator{E,F}, x::F) =
    A.inverse_adjoint!(y, x)


immutable FunctionalLinearEndomorphism{E} <: LinearEndomorphism{E}
    direct::Function
    direct!::Function
    adjoint::Function
    adjoint!::Function
    inverse::Function
    inverse!::Function
    inverse_adjoint::Function
    inverse_adjoint!::Function
end

function FunctionalLinearEndomorphism{E}(::Type{E};
                                         direct::Function=unimplemented_operation,
                                         direct!::Function=unimplemented_operation,
                                         adjoint::Function=unimplemented_operation,
                                         adjoint!::Function=unimplemented_operation,
                                         inverse::Function=unimplemented_operation,
                                         inverse!::Function=unimplemented_operation,
                                         inverse_adjoint::Function=unimplemented_operation,
                                         inverse_adjoint!::Function=unimplemented_operation)
    FunctionalLinearEndomorphism{E}(direct, direct!, adjoint, adjoint!, inverse,
                                    inverse!, inverse_adjoint, inverse_adjoint!)
end

apply_direct{E}(A::FunctionalLinearEndomorphism{E}, x::E) =
    A.direct(x)

apply_direct!{E}(y::E, A::FunctionalLinearEndomorphism{E}, x::E) =
    A.direct!(y, x)

apply_adjoint{E}(A::FunctionalLinearEndomorphism{E}, x::E) =
    A.adjoint(x)

apply_adjoint!{E}(y::E, A::FunctionalLinearEndomorphism{E}, x::E) =
    A.adjoint!(y, x)

apply_inverse{E}(A::FunctionalLinearEndomorphism{E}, x::E) =
    A.inverse(x)

apply_inverse!{E}(y::E, A::FunctionalLinearEndomorphism{E}, x::E) =
    A.inverse!(y, x)

apply_inverse_adjoint{E}(A::FunctionalLinearEndomorphism{E}, x::E) =
    A.inverse_adjoint(x)

apply_inverse_adjoint!{E}(y::E, A::FunctionalLinearEndomorphism{E}, x::E) =
    A.inverse_adjoint!(y, x)


immutable FunctionalSelfAdjointOperator{E} <: LinearEndomorphism{E}
    direct::Function
    direct!::Function
    inverse::Function
    inverse!::Function
end

function FunctionalSelfAdjointOperator{E}(::Type{E};
                                          direct::Function=unimplemented_operation,
                                          direct!::Function=unimplemented_operation,
                                          inverse::Function=unimplemented_operation,
                                          inverse!::Function=unimplemented_operation)
    FunctionalSelfAdjointOperator{E}(direct, direct!, inverse, inverse!)
end

apply_direct{E}(A::FunctionalSelfAdjointOperator{E}, x::E) =
    A.direct(x)

apply_direct!{E}(y::E, A::FunctionalSelfAdjointOperator{E}, x::E) =
    A.direct!(y, x)

apply_inverse{E}(A::FunctionalSelfAdjointOperator{E}, x::E) =
    A.inverse(x)

apply_inverse!{E}(y::E, A::FunctionalSelfAdjointOperator{E}, x::E) =
    A.inverse!(y, x)


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
