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
types).  Vector spaces implement the following methods:

* `length{T}(x::T)` to give the number of components in variable `x`;

* `vcreate{T}(x::T)` to create a new variable of the same space as `x`;

* `vcopy!{T}(dst::T, src::T)` to copy the contents of `src` into `dst`;

* `swap!{T}(x::T, y::T)` to exchange the contents of `x` and `y`;

* `vfill!{T}(x::T, alpha::Float)` to set all values of `x` with `alpha`;

* `vscale!{T}(x::T, alpha::Float)` to scale all values of `x` by `alpha`;

* `vdot{T}(x::T, y::{T})::Float` to compute the inner product of `x` and `y`,
  the result is expected to be a `Float`;

* `vcombine!{T}(dst::T, alpha::Float, x::T, beta::Float, y::T)` to perform
  `dst = alpha*x + beta*y`;

* `vproduct!{T}(dst::T, x::T, y::T)` stores the elementwise multiplication
  of `x` by `y` in `dst`.

The following methods may optionally be implemented:

* `vnorm2{T}(x::T)` to compute the Euclidean norm of `x`, if not provided, the
  default implementation is: `vnorm2(x) = sqrt(vdot(x, x))`;

* `vnorm1{T}(x::T)` to compute the L1 norm of `x`;

* `vnorminf{T}(x::T)` to compute the infinite norm of `x`;

* `update!{T}(dst::T, alpha::Float, x::T)` to perform `dst += alpha*x`, the
  default implementation is:
  ```julia
  update!{T}(dst::T, alpha::Float, x::T) = vcombine!(dst, 1.0, dst, alpha, x)`
  ```

* `vcombine!{T}(dst::T, alpha::Float, x::T, beta::Float, y::T, gamma::Float, z::T)` to perform `dst = alpha*x + beta*y + gamma*z` the default implementation is:
  ```julia
  function vcombine!{T}(dst::T, alpha::Float, x::T,
                        beta::Float, y::T, gamma::Float, z::T)
        vcombine!(dst, alpha, x, beta, y)
        update!(dst, gamma, z)
  end
  ```

TiPi provides reasonably optimized implementations of these methods for Julia
`Array` types.  So TiPi can be used out-of-the box if your unknowns are stored
in the form of Julia arrays.  Otherwise, you'll have to implement the above
methods.

Note that `length`, `vcreate`, `vcopy!`, `vscale!` and `vfill!` are identical
or similar to methods already provided by Julia for its arrays but the
semantics is somewhat different.  For instance, compared to `vcopy!`, `copy!`
copies all the values of the source array, the elements of the destination may
have a different data type and the destination may have different dimensions
than the source (the only constraint is that the destination must have at least
as many elements as the source).

The following methods must be implemented (`S` is a floating-point scalar type
and `V` is the type of your variables):

* `vnorm1(x::V, y::V)`
* `vnorminf(x::V, y::V)`
* `vdot(x::V, y::V)`
* `vcreate(x::V)`
* `vcopy!(dst::V, src::V)`
* `vswap!(x::V, y::V)`
* `vfill!(x::V, alpha::S)`
* `vupdate!(dst::V, alpha::S, x::V)`
* `vproduct!(dst::V, x::V, y::V)`
* `vcombine!(dst::V, alpha::S, x::V, beta::S, y::V)`

The following methods have default based on other provided method but you may
consider implementing a more efficient version:

* `vnorm2(x::V, y::V)` defaults to `sqrt(vdot(x, y))`;
* `vscale!(dst::V, alpha::S)` defaults to `vscale!(dst, alpha, dst)`;
* `vscale!(dst::V, alpha::S, src::V)` defaults to `vcombine!(dst, alpha, src, 0, src)`;
* `vproduct!(dst::V, src::V)` defaults to `vproduct!(dst, dst, src)`;

In order to use bound constraint optimization, you must provide the following
methods (`F` is the type of object returned by `get_free_variables`):

* `project_variables`;
* `project_direction`;
* `get_free_variables`;
* `step_limits`;
* `vdot(sel::F, x::V, y::V)`
* `vupdate!(dst::V, sel::F, alpha::S, x::V)`
* `vproduct!(dst::V, sel::F, x::V, y::V)`


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

    vdot(y, A*x) = vdot(A'*y, x)

whatever `x` and `y` (of the correct type).  This definition clearly depends on
the inner product which is implemented by the `vdot` method for the considered
variables (see above).

Of course combining operators is possible.  For instance:

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
  its own adjoint.  Its only parameter is the type of the input and output
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


