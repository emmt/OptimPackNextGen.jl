# Algebra

To solve inverse problems in a unified way, TiPi manipulates the unknowns of
the problems, the so-called "variables", at an abstract level requiring that a
few methods be implemented to manipulate the variables of interest.  Other
values can be scalar reals of type `TiPi.Float` (an alias to `Cdouble` which is
itself an alias to `Float64`) or integers of type `Int` (the default integer
type of Julia which is suitable for indexing arays).


## Variables and Vector Spaces

The "variables" are the unknowns of the considered inverse problem.  Variables
in TiPi belong to so-called "vector spaces" which can be anything needed to
store the variable values.  A variable instance is typically used as a template
to represent any variable of the same vector space.  The type of a variable is
not sufficient (for instance a Julia array type is only specified by the type
of its elements and its number of dimensions, whereas all dimensions must be
specified to build a similar array).

The values of a variables may have any type (or may even have each different
types) Vector spaces implement the following methods:

* `length{T}(x::T)` to give the number of components in variable `x`;

* `similar{T}(x::T)` to create a new variable of the same space as `x`;

* `copy!{T}(dst::T, src::T)` to copy the contents of `src` into `dst`;

* `swap!{T}(x::T, y::T)` to exchange the contents of `x` and `y`;

* `fill!{T}(x::T, alpha::Float)` to set all values of `x` with `alpha`;

* `scale!{T}(x::T, alpha::Float)` to scale all values of `x` by `alpha`;

* `inner{T}(x::T, y::{T})::Float` to compute the inner product of `x` and `y`,
  the result is expected to be a `Float`;

* `combine!{T}(dst::T, alpha::Float, x::T, beta::Float, y::T)` to perform
  `dst = alpha*x + beta*y`;

* `multiply!{T}(dst::T, x::T, y::T)`

The following methods may optionally be implemented:

* `norm2{T}(x::T)` to compute the Euclidean norm of `x`, if not provided, the
  default implementation is: `norm2(x) = sqrt(inner(x, x))`;

* `norm1{T}(x::T)` to compute the L1 norm of `x`;

* `normInf{T}(x::T)` to compute the infinite norm of `x`;

* `update!{T}(dst::T, alpha::Float, x::T)` to perform `dst += alpha*x`, the
  default implementation is: `update!{T}(dst::T, alpha::Float, x::T) =
  combine!(dst, 1.0, dst, alpha, x)`;

* `combine!{T}(dst::T, alpha::Float, x::T, beta::Float, y::T, gamma::Float,
  z::T)` to perform `dst = alpha*x + beta*y + gamma*z` the default
  implementation is:
  ```
  function combine!{T}(dst::T, alpha::Float, x::T,
                       beta::Float, y::T, gamma::Float, z::T)
        combine!(dst, alpha, x, beta, y)
        update!(dst, gamma, z)
  end
  ```

TiPi provides reasonnably optimized implementations of these methods for Julia
`Array` types (note that `length`, `similar`, `copy!`, `scale!` and `fill!` are
already provided by Julia for its arrays).  So TiPi can be used out-of-the box
if your unknowns are stored in the form of Julia arrays.  Otherwise, you'll
have to implement the above methods.


## Linear Operators

TiPi provides abstract types and methods for dealing with linear operators
which, with the "variables", are the basic building blocks of many inverse
problems.  This framework is suitable to have the following expressions
work as expected:

    A*x         # yield "operator" A applied to "vector" x
    A(x)        # idem
    call(A, x)  # idem

    A'*x        # yield the adjoint of "operator" A applied to "vector" x
    A'(x)       # idem
    call(A', x) # idem

Here "adjoint" closely follows the mathematical definition:

    inner(y, A*x) = inner(A'*y, x)

whatever `x` and `y` (of the correct type).  This definition clearly depends
on the inner product.

Of course combining operators is possible:

    A*B'*C*x
    (A*B*C)'*x = C'*B'*A'*x = C'(B'(A'(x)))

with the usual convention of using upper case latin letters for operators and
lower case latin letters for vectors.  Minimal type checking is automatically
performed: `A*x` requires that `x` belongs to the input space of `A`, `A'*x`
requires that `x` belongs to the output space of `A` and `A*B` requires that
the input space of `A` be the same as the output space of `B`.

For technical reasons, only left multiplication of a vector by an operator is
implemented.  This makes sense for inverse problems.

To benefit from this framework, an operator must be an instance of a concrete
type inherited from one of the abstract types `LinearOperator` or
`SelfAdjointOperator`.

* `LinearOperator{OUT,INP}` is the parametric abstract type from which inherit
  all linear operators.  The parameters `INP` and `OUT` are respectively the
  input and output types of the "vectors".

* A `SelfAdjointOperator{E}` is a more specialized `LinearOperator` which is
  its own adjoint.  Its only parameter is the input and out type of the
  "vectors".

Appart from defining the type for the operator, two methods, `apply_direct` and
`apply_adjoint` should be provided to apply the operator and its adjoint to an
argument of the suitable type.  The conventions are that:

    apply_direct(A, x)

yield the result of applying the adjoint of the linear operator `A` to the
"vector" `x`; while:

    apply_adjoint(A, x)

yields the result of applying the adjoint of the linear operator `A` to the
"vector" `x`.

Implementing a linear operator is typically done by:

    using TiPi.Algebra
    import TiPi.Algebra: apply_direct, apply_adjoint

    type MyOperator{E,F} <: LinearOperator{E,F}
        ...
    end

    apply_direct{E,F}(A::MyOperator{E,F}, x::F) = ...
    apply_adjoint{E,F}(A::MyOperator{E,F}, x::E) = ...

Explicitly importing `apply_direct` and `apply_adjoint` is needed to overload
these methods.

For a `SelfAdjointOperator` it is sufficient to provide `apply_direct`.
Implementing a self-adjoint operator amounts to:

    using TiPi.Algebra
    import TiPi.Algebra: apply_direct

    type MyOperator{E} <: SelfAdjointOperator{E}
        ...
    end

    apply_direct{E}(A::MyOperator{E}, x::E) = ...

