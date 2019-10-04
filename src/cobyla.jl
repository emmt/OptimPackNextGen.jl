#
# cobyla.jl --
#
# Julia interface to Mike Powell's COBYLA method.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2015-2019, Éric Thiébaut
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module Cobyla

export
    cobyla,
    cobyla!

using Compat
using Compat.Printf

import
    ..AbstractContext,
    ..AbstractStatus,
    ..getncalls,
    ..getradius,
    ..getreason,
    ..getstatus,
    ..grow!,
    ..iterate,
    ..restart

# The dynamic library implementing the method.
import .._libcobyla
const DLL = _libcobyla

# Status returned by most functions of the library.
struct Status <: AbstractStatus
    _code::Cint
end

# Possible status values returned by COBYLA.
const INITIAL_ITERATE      = Status( 2)
const ITERATE              = Status( 1)
const SUCCESS              = Status( 0)
const BAD_NVARS            = Status(-1)
const BAD_NCONS            = Status(-2)
const BAD_RHO_RANGE        = Status(-3)
const BAD_SCALING          = Status(-4)
const ROUNDING_ERRORS      = Status(-5)
const TOO_MANY_EVALUATIONS = Status(-6)
const BAD_ADDRESS          = Status(-7)
const CORRUPTED            = Status(-8)

# Get a textual explanation of the status returned by COBYLA.
function getreason(status::Status)
    ptr = ccall((:cobyla_reason, DLL), Ptr{UInt8}, (Cint,), status._code)
    if ptr == C_NULL
        error("unknown COBYLA status: ", status._code)
    end
    unsafe_string(ptr)
end

"""
# Minimizing a function of many variables subject to inequality constraints

Mike Powell's **COBYLA** algorithm attempts to find the variables `x` which
solve the problem:

    min f(x)    s.t.   c(x) <= 0

where `x` is a vector of variables that has `n ≥ 1` components, `f(x)` is an
objective function and `c(x)` implement `m` inequality constraints.  The
algorithm employs linear approximations to the objective and constraint
functions, the approximations being formed by linear interpolation at `n+1`
points in the space of the variables.  We regard these interpolation points as
vertices of a simplex.  The parameter `rho` controls the size of the simplex
and it is reduced automatically from `rhobeg` to `rhoend`.  For each `rho`,
COBYLA tries to achieve a good vector of variables for the current size, and
then `rho` is reduced until the value `rhoend` is reached.  Therefore `rhobeg`
and `rhoend` should be set to reasonable initial changes to and the required
accuracy in the variables respectively, but this accuracy should be viewed as a
subject for experimentation because it is not guaranteed.  The subroutine has
an advantage over many of its competitors, however, which is that it treats
each constraint individually when calculating a change to the variables,
instead of lumping the constraints together into a single penalty function.
The name of the subroutine is derived from the phrase "Constrained Optimization
BY Linear Approximations".

The in-place version of the algorithm is called as:

    Cobyla.minimize!(fc, x, m, rhobeg, rhoend) -> (status, x, fx)

where `x` is a vector with the initial and final variables, `m`, `rhobeg`, and
`rhoend` have been defined already, while `fc` is a Julia function which is
called as:

    fc(x, cx) -> fx

to store in `cx` the values of the constraints at `x` and to return `fx` the
value of the objective function at `x`.  If there are no constraints
(i.e. `m=0`), then `fc` is called without the `cx` argument as:

    fc(x) -> fx

The method returns a tuple with `status` the termination condition (should be
`Cobyla.SUCCESS` unless keyword `check` is set `false`, see below), `x` the
solution found by the algorithm and `fx` the corresponding function value.

The method:

    Cobyla.minimize(fc, x0, m, rhobeg, rhoend) -> (status, x, fx)

is identical but to `Cobyla.minimize!` does not modify the vector `x0` of
initial variables.


## Scaling of variables

The proper scaling of the variables is important for the success of the
algorithm and the optional `scale` keyword should be specified if the typical
precision is not the same for all variables.  If specified, `scale` is an array
of strictly nonnegative values and of same size as the variables `x`, such that
`scale[i]*rho` (with `rho` the trust region radius) is the size of the trust
region for the `i`-th variable.  If keyword `scale` is not specified, a unit
scaling for all the variables is assumed.


## Keywords

The following keywords are available:

* `scale` specifies the typical magnitudes of the variables.  If specified, it
  must have as many elements as `x`, all strictly positive.

* `check` (`true` by default) specifies whether to throw an exception if the
  algorithm is not fully successful.

* `verbose` (`0` by default) set the amount of printing.

* `maxeval` set the maximum number of calls to the objective function.  The
  default setting is `maxeval = 30n` with `n = length(x)` the number of
  variables.


## References

The algorithm is described in:

* M.J.D. Powell, "A direct search optimization method that models the objective
  and constraint functions by linear interpolation," in Advances in
  Optimization and Numerical Analysis Mathematics and Its Applications,
  vol. 275 (eds. Susana Gomez and Jean-Pierre Hennart), Kluwer Academic
  Publishers, pp. 51-67 (1994).

"""
minimize(args...; kwds...) = optimize(args...; maximize=false, kwds...)
minimize!(args...; kwds...) = optimize!(args...; maximize=false, kwds...)
@doc @doc(minimize) minimize!

"""

    Cobyla.maximize(fc, x0, m, rhobeg, rhoend) -> (status, x, fx)
    Cobyla.maximize!(fc, x, m, rhobeg, rhoend) -> (status, x, fx)

are similar to `Cobyla.minimize` and `Cobyla.minimize!` respectively but
solve the contrained maximization problem:

    max f(x)    s.t.   c(x) <= 0

"""
maximize(args...; kwds...) = optimize(args...; maximize=true, kwds...)
maximize!(args...; kwds...) = optimize!(args...; maximize=true, kwds...)
@doc @doc(maximize) maximize!

# `_wrklen(...)` yields the number of elements in COBYLA workspace.
_wrklen(n::Integer, m::Integer) = _wrklen(Int(n), Int(m))
_wrklen(n::Int, m::Int) = n*(3*n + 2*m + 11) + 4*m + 6

# `_work(...)` yields a large enough workspace for NEWUOA.
_work(::Type{T}, len::Integer) where {T} = Vector{T}(undef, len)

# Wrapper for the objective function in COBYLA, the actual objective
# function is provided by the client data as a `jl_value_t*` pointer.
function _objfun(n::Cptrdiff_t, m::Cptrdiff_t, xptr::Ptr{Cdouble},
                 _c::Ptr{Cdouble}, fptr::Ptr{Cvoid})::Cdouble
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    return (m > 0 ? Cdouble(f(x, unsafe_wrap(Array, _c, m))) : Cdouble(f(x)))
end

# With precompilation, `__init__()` carries on initializations that must occur
# at runtime like `@cfunction` which returns a raw pointer.
const _objfun_c = Ref{Ptr{Cvoid}}()
function __init__()
    _objfun_c[] = @cfunction(_objfun, Cdouble,
                             (Cptrdiff_t, Cptrdiff_t, Ptr{Cdouble},
                              Ptr{Cdouble}, Ptr{Cvoid}))
end

"""
The methods:

    Cobyla.optimize(fc, x0, m, rhobeg, rhoend) -> (status, x, fx)
    Cobyla.optimize!(fc, x, m, rhobeg, rhoend) -> (status, x, fx)

are identical to `Cobyla.minimize` and `Cobyla.minimize!` respectively but have
an additional `maximize` keyword which is `false` by default and which
specifies whether to maximize the objective function; otherwise, the method
attempts to minimize the objective function.

"""
optimize(fc::Function, x0::AbstractVector{<:Real}, args...; kwds...) =
    optimize!(fc, copyto!(Array{Cdouble}(undef, length(x0)), x0),
              args...; kwds...)

function optimize!(fc::Function, x::DenseVector{Cdouble},
                   m::Integer, rhobeg::Real, rhoend::Real;
                   scale::DenseVector{Cdouble} = Array{Cdouble}(undef, 0),
                   maximize::Bool = false,
                   check::Bool = false,
                   verbose::Integer = 0,
                   maxeval::Integer = 30*length(x),
                   work::Vector{Cdouble} = _work(Cdouble, _wrklen(length(x), m)),
                   iact::Vector{Cptrdiff_t} = _work(Cptrdiff_t, m + 1))
    n = length(x)
    nscl = length(scale)
    if nscl == 0
        sclptr = Ptr{Cdouble}(0)
    elseif nscl == n
        sclptr = pointer(scale)
    else
        error("bad number of scaling factors")
    end
    grow!(work, _wrklen(n, m))
    grow!(iact, m + 1)
    status = Status(ccall((:cobyla_optimize, DLL), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Cint, Ptr{Cvoid}, Any,
                           Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble,
                           Cptrdiff_t, Cptrdiff_t, Ptr{Cdouble},
                           Ptr{Cptrdiff_t}),
                          n, m, maximize, _objfun_c[], fc,
                          x, sclptr, rhobeg, rhoend, verbose, maxeval,
                          work, iact))
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

@doc @doc(optimize) optimize!

# Simpler version, mostly for testing.

function cobyla!(f::Function, x::DenseVector{Cdouble},
                 m::Integer, rhobeg::Real, rhoend::Real;
                 check::Bool = true,
                 verbose::Integer = 0,
                 maxeval::Integer = 30*length(x),
                 work::Vector{Cdouble} = _work(Cdouble, _wrklen(length(x), m)),
                 iact::Vector{Cptrdiff_t} = _work(Cptrdiff_t, m + 1))
    n = length(x)
    grow!(work, _wrklen(n, m))
    grow!(iact, m + 1)
    status = Status(ccall((:cobyla, DLL), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Ptr{Cvoid}, Any,
                           Ptr{Cdouble}, Cdouble, Cdouble, Cptrdiff_t,
                           Cptrdiff_t, Ptr{Cdouble}, Ptr{Cptrdiff_t}),
                          n, m, _objfun_c[], f, x, rhobeg, rhoend,
                          verbose, maxeval, work, iact))
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

cobyla(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    cobyla!(f, copy(x0), args...; kwds...)

"""

```julia
using OptimPackNextGen.Powell
ctx = Cobyla.Context(n, m, rhobeg, rhoend; verbose=0, maxeval=500)
```

creates a new reverse communication workspace for COBYLA algorithm.  A typical
usage is:


```julia
x = Array{Cdouble}(undef, n)
c = Array{Cdouble}(undef, m)
x[...] = ... # initial solution
ctx = Cobyla.Context(n, m, rhobeg, rhoend, verbose=1, maxeval=500)
status = getstatus(ctx)
while status == Cobyla.ITERATE
    fx = ...       # compute function value at X
    c[...] = ...   # compute constraints at X
    status = iterate(ctx, fx, x, c)
end
if status != Cobyla.SUCCESS
    println("Something wrong occured in COBYLA: ", getreason(status))
end
```

""" Context

# Context for reverse communication variant of COBYLA.
# Must be mutable to be finalized.
mutable struct Context <: AbstractContext
    ptr::Ptr{Cvoid}
    n::Int
    m::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    verbose::Int
    maxeval::Int
    function Context(n::Integer, m::Integer,
                     rhobeg::Real, rhoend::Real;
                     verbose::Integer=0, maxeval::Integer=500)
        n ≥ 2 || throw(ArgumentError("bad number of variables"))
        m ≥ 0 || throw(ArgumentError("bad number of constraints"))
        0 ≤ rhoend ≤ rhobeg ||
            throw(ArgumentError("bad trust region radius parameters"))
        ptr = ccall((:cobyla_create, DLL), Ptr{Cvoid},
                    (Cptrdiff_t, Cptrdiff_t, Cdouble, Cdouble,
                     Cptrdiff_t, Cptrdiff_t),
                    n, m, rhobeg, rhoend, verbose, maxeval)
        ptr != C_NULL || error(errno() == Base.Errno.ENOMEM
                               ? "insufficient memory"
                               : "unexpected error")
        return finalizer(ctx -> ccall((:cobyla_delete, DLL), Cvoid,
                                      (Ptr{Cvoid},), ctx.ptr),
                         new(ptr, n, m, rhobeg, rhoend, verbose, maxeval))
    end
end

@deprecate create(args...; kwds...) Context(args...; kwds...)

function iterate(ctx::Context, f::Real, x::DenseVector{Cdouble},
                 c::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    length(c) == ctx.m || error("bad number of constraints")
    Status(ccall((:cobyla_iterate, DLL), Cint,
                 (Ptr{Cvoid}, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}),
                 ctx.ptr, f, x, c))
end

function iterate(ctx::Context, f::Real, x::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    ctx.m == 0 || error("bad number of constraints")
    Status(ccall((:cobyla_iterate, DLL), Cint,
                 (Ptr{Cvoid}, Cdouble, Ptr{Cdouble}, Ptr{Cvoid}),
                 ctx.ptr, f, x, C_NULL))
end

restart(ctx::Context) =
    Status(ccall((:cobyla_restart, DLL), Cint, (Ptr{Cvoid},), ctx.ptr))

getstatus(ctx::Context) =
    Status(ccall((:cobyla_get_status, DLL), Cint, (Ptr{Cvoid},), ctx.ptr))

# Get the current number of function evaluations.  Result is -1 if
# something is wrong (e.g. CTX is NULL), nonnegative otherwise.
getncalls(ctx::Context) =
    Int(ccall((:cobyla_get_nevals, DLL), Cptrdiff_t, (Ptr{Cvoid},), ctx.ptr))

getradius(ctx::Context) =
    ccall((:cobyla_get_rho, DLL), Cdouble, (Ptr{Cvoid},), ctx.ptr)

getlastf(ctx::Context) =
    ccall((:cobyla_get_last_f, DLL), Cdouble, (Ptr{Cvoid},), ctx.ptr)

end # module Cobyla
