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
# Copyright (C) 2015-2019 Éric Thiébaut
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module Newuoa

export
    newuoa,
    newuoa!

using Printf

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
import .._libnewuoa
const DLL = _libnewuoa

# Status returned by most functions of the library.
struct Status <: AbstractStatus
    _code::Cint
end

# Possible status values returned by NEWUOA.
const INITIAL_ITERATE      = Status( 2)
const ITERATE              = Status( 1)
const SUCCESS              = Status( 0)
const BAD_NVARS            = Status(-1)
const BAD_NPT              = Status(-2)
const BAD_RHO_RANGE        = Status(-3)
const BAD_SCALING          = Status(-4)
const ROUNDING_ERRORS      = Status(-5)
const TOO_MANY_EVALUATIONS = Status(-6)
const STEP_FAILED          = Status(-7)
const BAD_ADDRESS          = Status(-8)
const CORRUPTED            = Status(-9)

# Get a textual explanation of the status returned by NEWUOA.
function getreason(status::Status)
    ptr = ccall((:newuoa_reason, DLL), Ptr{UInt8}, (Cint,), status._code)
    if ptr == C_NULL
        error("unknown NEWUOA status: ", status._code)
    end
    unsafe_string(ptr)
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
norm of the change of variables) of the region to explorate and `rhoend`
should be set to the typical precision.  The proper scaling of the variables is
important for the success of the algorithm and the optional `scale` keyword
should be specified if the typical precision is not the same for all variables.
If specified, `scale` is an array of strictly nonnegative values and of same
size as the variables `x`, such that `scale[i]*rho` (with `rho` the trust
region radius) is the size of the trust region for the `i`-th variable.  If
keyword `scale` is not specified, a unit scaling for all the variables is
assumed.


## Keywords

The following keywords are available:

* `scale` specifies the typical magnitudes of the variables.  If specified, it
  must have as many elements as `x`, all strictly positive.

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
function _objfun(n::Cptrdiff_t, xptr::Ptr{Cdouble}, fptr::Ptr{Cvoid})::Cdouble
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    return Cdouble(f(x))
end

# With precompilation, `__init__()` carries on initializations that must occur
# at runtime like `@cfunction` which returns a raw pointer.
const _objfun_c = Ref{Ptr{Cvoid}}()
function __init__()
    _objfun_c[] = @cfunction(_objfun, Cdouble,
                             (Cptrdiff_t, Ptr{Cdouble}, Ptr{Cvoid}))
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
                   scale::DenseVector{Cdouble} = Cdouble[],
                   maximize::Bool = false,
                   npt::Integer = 2*length(x) + 1,
                   check::Bool = true,
                   verbose::Integer = 0,
                   maxeval::Integer = 30*length(x),
                   work::Vector{Cdouble} = _work(x, scale, npt))
    n = length(x)
    nscl = length(scale)
    if nscl == 0
        sclptr = Ptr{Cdouble}(0)
    elseif nscl == n
        sclptr = pointer(scale)
    else
        error("bad number of scaling factors")
    end
    grow!(work, _wrklen(x, scale, npt))
    status = Status(ccall((:newuoa_optimize, DLL), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Cint, Ptr{Cvoid}, Any,
                           Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble,
                           Cptrdiff_t, Cptrdiff_t, Ptr{Cdouble}),
                          n, npt, maximize, _objfun_c[], f, x, sclptr,
                          rhobeg, rhoend, verbose, maxeval, work))
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
    status = Status(ccall((:newuoa, DLL), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Ptr{Cvoid}, Any,
                           Ptr{Cdouble}, Cdouble, Cdouble, Cptrdiff_t,
                           Cptrdiff_t, Ptr{Cdouble}),
                          n, npt, _objfun_c[], f, x, rhobeg, rhoend,
                          verbose, maxeval, work))
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
    ptr::Ptr{Cvoid}
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
        ptr = ccall((:newuoa_create, DLL), Ptr{Cvoid},
                    (Cptrdiff_t, Cptrdiff_t, Cdouble, Cdouble,
                     Cptrdiff_t, Cptrdiff_t),
                    n, npt, rhobeg, rhoend, verbose, maxeval)
        ptr != C_NULL || error(errno() == Base.Errno.ENOMEM
                               ? "insufficient memory"
                               : "invalid argument(s)")
        return finalizer(ctx -> ccall((:newuoa_delete, DLL), Cvoid,
                                      (Ptr{Cvoid},), ctx.ptr),
                         new(ptr, n, npt, rhobeg, rhoend, verbose, maxeval))
    end
end

@deprecate create(args...; kwds...) Context(args...; kwds...)

function iterate(ctx::Context, f::Real, x::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    Status(ccall((:newuoa_iterate, DLL), Cint,
                       (Ptr{Cvoid}, Cdouble, Ptr{Cdouble}),
                       ctx.ptr, f, x))
end

restart(ctx::Context) =
    Status(ccall((:newuoa_restart, DLL), Cint, (Ptr{Cvoid},), ctx.ptr))

getstatus(ctx::Context) =
    Status(ccall((:newuoa_get_status, DLL), Cint, (Ptr{Cvoid},), ctx.ptr))

getncalls(ctx::Context) =
    Int(ccall((:newuoa_get_nevals, DLL), Cptrdiff_t, (Ptr{Cvoid},), ctx.ptr))

getradius(ctx::Context) =
    ccall((:newuoa_get_rho, DLL), Cdouble, (Ptr{Cvoid},), ctx.ptr)

end # module Newuoa
