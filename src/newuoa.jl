#
# newuoa.jl --
#
# Julia interface to Mike Powell's NEWUOA method.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2015-2022 Éric Thiébaut
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module Newuoa

export
    newuoa,
    newuoa!

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
const Status               = Lib.newuoa_status
const INITIAL_ITERATE      = Lib.NEWUOA_INITIAL_ITERATE
const ITERATE              = Lib.NEWUOA_ITERATE
const SUCCESS              = Lib.NEWUOA_SUCCESS
const BAD_NVARS            = Lib.NEWUOA_BAD_NVARS
const BAD_NPT              = Lib.NEWUOA_BAD_NPT
const BAD_RHO_RANGE        = Lib.NEWUOA_BAD_RHO_RANGE
const BAD_SCALING          = Lib.NEWUOA_BAD_SCALING
const ROUNDING_ERRORS      = Lib.NEWUOA_ROUNDING_ERRORS
const TOO_MANY_EVALUATIONS = Lib.NEWUOA_TOO_MANY_EVALUATIONS
const STEP_FAILED          = Lib.NEWUOA_STEP_FAILED
const BAD_ADDRESS          = Lib.NEWUOA_BAD_ADDRESS
const CORRUPTED            = Lib.NEWUOA_CORRUPTED

# Get a textual explanation of the status returned by NEWUOA.
function getreason(status::Status)
    cstr = Lib.newuoa_reason(status)
    if cstr == C_NULL
        error("unknown NEWUOA status: ", status)
    end
    unsafe_string(cstr)
end

"""
# Minimizing a function of many variables

Mike Powell's **NEWUOA** algorithm attempts to find the variables `x` which
solve the problem:

    min f(x)

where `x` is a vector of variables that has `n ≥ 2` components and `f(x)` is
an objective function.  The algorithm employs quadratic approximations to the
objective which interpolates the objective function at `m` points, the value
`m = 2n + 1` being recommended.  The parameter `rho` controls the size of the
trust region and it is reduced automatically from `rhobeg` to `rhoend` (such
that `0 < rhoend ≤ rhobeg`).

The in-place version of the algorithm is called as:

    Newuoa.minimize!(f, x, rhobeg, rhoend) -> (status, x, fx)

where `f` is the objective function, `x` is a vector with the initial and final
variables, `rhobeg` and `rhoend` are the initial and final sizes of the trust
region.  The result is a tuple of 3 values: `status` indicates whether the
algorithm was successful, `x` is the final value of the variables and `fx =
f(x)` is the objective function at `x`.  Normally, `status` should be
`Newuoa.SUCCESS`; otherwise, `getreason(status)` yields a textual explanation
about the failure.

The method:

    Newuoa.minimize(f, x0, rhobeg, rhoend) -> (status, x, fx)

is identical to `Newuoa.minimize!` but does not modify the vector `x0` of
initial variables.


## Precision and scaling of variables

Parameter `rhobeg` should be set to the typical size (in terms of Euclidean
norm of the change of variables) of the region to explorate and `rhoend` should
be set to the typical precision. The proper scaling of the variables is
important for the success of the algorithm and the optional `scale` keyword
should be specified if the typical precision is not the same for all variables.
If `scale` is an array of strictly positive values and of same size as the
variables `x`, then `scale[i]*rho` (with `rho` the trust region radius) is the
size of the trust region for the `i`-th variable. Keyword `scale` may also be
set with a strictly positive scalar to assume the same scaling factor for all
variables. If keyword `scale` is not specified, a unit scaling for all the
variables is assumed.


## Keywords

The following keywords are available:

* `scale` specifies the typical magnitudes of the variables. If specified, it
  must be a vector with as many elements as `x`, all strictly positive, or a
  strictly positive scalar to assume the same scaling factor for all variables.
  If not specified, `scale[i] = 1` is assumed for any `i ∈ 1:n`.

* `check` (`true` by default) specifies whether to throw an exception if the
  algorithm is not fully successful.

* `verbose` (`0` by default) set the amount of printing.

* `maxeval` specifies the maximum number of calls to the objective function.
  The default setting is `maxeval = 30n` with `n = length(x)` the number of
  variables.

* `npt` specifies the number of points to use for the quadratic approximation
  of the objective function.  The default setting is the recommended value:
  `npt = 2n + 1` with `n = length(x)` the number of variables.

* `work` specifies a workspace to (re)use.  It must be a vector of double
  precision floating-point values.  If it is too small, its size is
  automatically adjusted (by calling [`resize!`](@ref)).  This keyword is
  useful to avoid any new allocation (and garbage colection) when several
  similar optimizations are to be performed.


## References

The algorithm is described in:

* M.J.D. Powell, "The NEWUOA software for unconstrained minimization without
  derivatives," in Large-Scale Nonlinear Optimization, editors G. Di Pillo and
  M. Roma, Springer, pp. 255-297 (2006).

"""
minimize(args...; kwds...) = optimize(args...; maximize=false, kwds...)
minimize!(args...; kwds...) = optimize!(args...; maximize=false, kwds...)
@doc @doc(minimize) minimize!

"""

    Newuoa.maximize(f, x0, rhobeg, rhoend) -> (status, x, fx)
    Newuoa.maximize!(f, x, rhobeg, rhoend) -> (status, x, fx)

are similar to `Newuoa.minimize` and `Newuoa.minimize!` respectively but
solve the unconstrained maximization problem:

    max f(x)

"""
maximize(args...; kwds...) = optimize(args...; maximize=true, kwds...)
maximize!(args...; kwds...) = optimize!(args...; maximize=true, kwds...)
@doc @doc(maximize) maximize!

# `_wrklen(...)` yields the number of elements in NEWUOA workspace.
_wrklen(n::Integer, npt::Integer) = _wrklen(Int(n), Int(npt))
_wrklen(n::Int, npt::Int) = (npt + 13)*(npt + n) + div(3*n*(n + 3),2)
_wrklen(x::AbstractVector{<:AbstractFloat}, npt::Integer) =
    _wrklen(length(x), npt)
function _wrklen(x::AbstractVector{<:AbstractFloat},
                 scl::AbstractVector{<:AbstractFloat},
                 npt::Integer)
    return _wrklen(x, npt) + length(scl)
end

# `_work(...)` yields a large enough workspace for NEWUOA.
_work(x::AbstractVector{<:AbstractFloat}, npt::Integer) =
    Vector{Cdouble}(undef, _wrklen(x, npt))
function _work(x::AbstractVector{<:AbstractFloat},
               scl::AbstractVector{<:AbstractFloat},
               npt::Integer)
    return Vector{Cdouble}(undef, _wrklen(x, scl, npt))
end

# Wrapper for the objective function in NEWUOA, the actual objective function
# is provided by the client data as a `jl_value_t*` pointer.
function _objfun(n::opk_index, xptr::Ptr{Cdouble}, fptr::Ptr{Cvoid})::Cdouble
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    return Cdouble(f(x))
end

# With precompilation, `__init__()` carries on initializations that must occur
# at runtime like `@cfunction` which returns a raw pointer.
const _objfun_c = Ref{Ptr{Cvoid}}()
function __init__()
    _objfun_c[] = @cfunction(_objfun, Cdouble,
                             (opk_index, Ptr{Cdouble}, Ptr{Cvoid}))
end

"""
The methods:

    Newuoa.optimize(fc, x0, rhobeg, rhoend) -> (status, x, fx)
    Newuoa.optimize!(fc, x, rhobeg, rhoend) -> (status, x, fx)

are identical to `Newuoa.minimize` and `Newuoa.minimize!` respectively but have
an additional `maximize` keyword which is `false` by default and which
specifies whether to maximize the objective function; otherwise, the method
attempts to minimize the objective function.

"""
optimize(f::Function, x0::AbstractVector{<:Real}, args...; kwds...) =
    optimize!(f, copyto!(Array{Cdouble}(undef, length(x0)), x0),
              args...; kwds...)

function optimize!(f::Function, x::DenseVector{Cdouble},
                   rhobeg::Real, rhoend::Real;
                   scale::Scale = defaultscale,
                   maximize::Bool = false,
                   npt::Integer = 2*length(x) + 1,
                   check::Bool = true,
                   verbose::Integer = 0,
                   maxeval::Integer = 30*length(x),
                   work::Vector{Cdouble} = _work(x, scale, npt))
    n = length(x)
    scl = to_scale(scale, n)
    grow!(work, _wrklen(x, scl, npt))
    status = Lib.newuoa_optimize(
        n, npt, maximize, _objfun_c[], f, x,
        scl, rhobeg, rhoend, verbose, maxeval, work)
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

@doc @doc(optimize) optimize!

# Basic version similar to the FORTRAN version.
newuoa(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    newuoa!(f, copy(x0), args...; kwds...)

function newuoa!(f::Function, x::DenseVector{Cdouble},
                 rhobeg::Real, rhoend::Real;
                 npt::Integer = 2*length(x) + 1,
                 verbose::Integer = 0,
                 maxeval::Integer = 30*length(x),
                 check::Bool = true,
                 work::Vector{Cdouble} = _work(x, npt))
    n = length(x)
    grow!(work, _wrklen(x, npt))
    status = Lib.newuoa(
        n, npt, _objfun_c[], f, x, rhobeg, rhoend, verbose, maxeval, work)
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

"""

```julia
using OptimPackNextGen.Powell
ctx = Newuoa.create(n, rhobeg, rhoend; npt=..., verbose=..., maxeval=...)
```

creates a new reverse communication workspace for NEWUOA algorithm.  A typical
usage is:

```julia
x = Array{Cdouble}(undef, n)
x[...] = ... # initial solution
ctx = Newuoa.Context(n, rhobeg, rhoend; verbose=1, maxeval=500)
status = getstatus(ctx)
while status == Newuoa.ITERATE
    fx = ...       # compute function value at X
    status = iterate(ctx, fx, x)
end
if status != Newuoa.SUCCESS
    println("Something wrong occured in NEWUOA: ", getreason(status))
end
```

""" Context

# Context for reverse communication variant of NEWUOA.
# Must be mutable to be finalized.
mutable struct Context <: AbstractContext
    ptr::Ptr{Lib.newuoa_context}
    n::Int
    npt::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    verbose::Int
    maxeval::Int
    function Context(n::Integer, rhobeg::Real, rhoend::Real;
                     npt::Integer = 2*length(x) + 1,
                     verbose::Integer = 0,
                     maxeval::Integer = 30*length(x))
        ptr = Lib.newuoa_create(n, npt, rhobeg, rhoend, verbose, maxeval)
        ptr != C_NULL || error(errno() == Base.Errno.ENOMEM
                               ? "insufficient memory"
                               : "invalid argument(s)")
        ctx = new(ptr, n, npt, rhobeg, rhoend, verbose, maxeval)
        return finalizer(_finalize, ctx)
    end
end

function _finalize(ctx::Context)
    if ctx.ptr != C_NULL
        Lib.newuoa_delete(ctx.ptr)
        ctx.ptr = C_NULL
    end
end

@deprecate create(args...; kwds...) Context(args...; kwds...)

function iterate(ctx::Context, f::Real, x::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    return Lib.newuoa_iterate(ctx.ptr, f, x)
end

restart(ctx::Context) = Lib.newuoa_restart(ctx.ptr)
getstatus(ctx::Context) = Lib.newuoa_get_status(ctx.ptr)
getncalls(ctx::Context) = Lib.newuoa_get_nevals(ctx.ptr) |> Int
getradius(ctx::Context) = Lib.newuoa_get_rho(ctx.ptr)

end # module Newuoa
