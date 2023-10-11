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
# Copyright (C) 2015-2022, Éric Thiébaut
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module Cobyla

export
    cobyla,
    cobyla!

using Printf

using ...Lib: opk_index

import
    ...Lib,
    ..AbstractContext,
    ..getncalls,
    ..getradius,
    ..getreason,
    ..getstatus,
    ..grow!,
    ..iterate,
    ..restart,
    ..Scale,
    ..defaultscale,
    ..to_scale

# Aliases.
const Status               = Lib.cobyla_status
const INITIAL_ITERATE      = Lib.COBYLA_INITIAL_ITERATE
const ITERATE              = Lib.COBYLA_ITERATE
const SUCCESS              = Lib.COBYLA_SUCCESS
const BAD_NVARS            = Lib.COBYLA_BAD_NVARS
const BAD_NCONS            = Lib.COBYLA_BAD_NCONS
const BAD_RHO_RANGE        = Lib.COBYLA_BAD_RHO_RANGE
const BAD_SCALING          = Lib.COBYLA_BAD_SCALING
const ROUNDING_ERRORS      = Lib.COBYLA_ROUNDING_ERRORS
const TOO_MANY_EVALUATIONS = Lib.COBYLA_TOO_MANY_EVALUATIONS
const BAD_ADDRESS          = Lib.COBYLA_BAD_ADDRESS
const CORRUPTED            = Lib.COBYLA_CORRUPTED

# Get a textual explanation of the status returned by COBYLA.
function getreason(status::Status)
    cstr = Lib.cobyla_reason(status)
    if cstr == C_NULL
        error("unknown COBYLA status: ", status)
    end
    unsafe_string(cstr)
end

"""
# Minimizing a function of many variables subject to inequality constraints

Mike Powell's **COBYLA** algorithm attempts to find the variables `x` which
solve the problem:

    min f(x)    s.t.   c(x) ≤ 0

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
precision is not the same for all variables. If `scale` is an array of strictly
positive values and of same size as the variables `x`, then `scale[i]*rho`
(with `rho` the trust region radius) is the size of the trust region for the
`i`-th variable. Keyword `scale` may also be set with a strictly positive
scalar to assume the same scaling factor for all variables. If keyword `scale`
is not specified, a unit scaling for all the variables is assumed.


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

    max f(x)    s.t.   c(x) ≤ 0

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
function _objfun(n::opk_index, m::opk_index, xptr::Ptr{Cdouble},
                 cptr::Ptr{Cdouble}, fptr::Ptr{Cvoid})::Cdouble
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    return (m > 0 ? Cdouble(f(x, unsafe_wrap(Array, cptr, m))) : Cdouble(f(x)))
end

# Without argument, yield the raw pointer that can be passed to C code.
_objfun() = @cfunction(_objfun, Cdouble, (opk_index, opk_index, Ptr{Cdouble},
                                          Ptr{Cdouble}, Ptr{Cvoid}))

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
    optimize!(fc, copyto!(Vector{Cdouble}(undef, length(x0)), x0),
              args...; kwds...)

function optimize!(fc::Function, x::DenseVector{Cdouble},
                   m::Integer, rhobeg::Real, rhoend::Real;
                   scale::Scale = defaultscale,
                   maximize::Bool = false,
                   check::Bool = false,
                   verbose::Integer = 0,
                   maxeval::Integer = 30*length(x),
                   work::Vector{Cdouble} = _work(Cdouble, _wrklen(length(x), m)),
                   iact::Vector{opk_index} = _work(opk_index, m + 1))
    n = length(x)
    scl = to_scale(scale, n)
    grow!(work, _wrklen(n, m))
    grow!(iact, m + 1)
    status = Lib.cobyla_optimize(
        n, m, maximize, _objfun(), fc, x, scl, rhobeg, rhoend,
        verbose, maxeval, work, iact)
    if check && status != COBYLA_SUCCESS
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
                 iact::Vector{opk_index} = _work(opk_index, m + 1))
    n = length(x)
    grow!(work, _wrklen(n, m))
    grow!(iact, m + 1)
    status = Lib.cobyla(
        n, m, _objfun(), f, x, rhobeg, rhoend, verbose, maxeval, work, iact)
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

cobyla(f::Function, x0::AbstractVector{<:Real}, args...; kwds...) =
    cobyla!(f, copyto!(Vector{Cdouble}(undef, length(x0)), x0), args...; kwds...)

"""
    using OptimPackNextGen.Powell
    ctx = Cobyla.Context(n, m, rhobeg, rhoend; verbose=0, maxeval=500)

creates a new reverse communication context for COBYLA algorithm. A typical
usage is:

    x = Vector{Cdouble}(undef, n)
    c = Vector{Cdouble}(undef, m)
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

"""
mutable struct Context <: AbstractContext
    ptr::Ptr{Lib.cobyla_context}
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
        ptr = Lib.cobyla_create(n, m, rhobeg, rhoend, verbose, maxeval)
        ptr != C_NULL || error(errno() == Base.Errno.ENOMEM
                               ? "insufficient memory"
                               : "unexpected error")
        ctx = new(ptr, n, m, rhobeg, rhoend, verbose, maxeval)
        return finalizer(_finalize, ctx)
    end
end

function _finalize(ctx::Context)
    if ctx.ptr != C_NULL
        Lib.cobyla_delete(ctx.ptr)
        ctx.ptr = C_NULL # do not delete twice
    end
end

function iterate(ctx::Context, f::Real, x::DenseVector{Cdouble},
                 c::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    length(c) == ctx.m || error("bad number of constraints")
    return Lib.cobyla_iterate(ctx.ptr, f, x, c)
end

function iterate(ctx::Context, f::Real, x::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    ctx.m == 0 || error("bad number of constraints")
    return Lib.cobyla_iterate(ctx.ptr, f, x, C_NULL)
end

restart(ctx::Context) = Lib.cobyla_restart(ctx.ptr)
getstatus(ctx::Context) = Lib.cobyla_get_status(ctx.ptr)
getncalls(ctx::Context) = Lib.cobyla_get_nevals(ctx.ptr) |> Int
getradius(ctx::Context) = Lib.cobyla_get_rho(ctx.ptr)
getlastf(ctx::Context) = Lib.cobyla_get_last_f(ctx.ptr)

end # module Cobyla
