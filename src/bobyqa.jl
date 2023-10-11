#
# bobyqa.jl --
#
# Julia interface to Mike Powell's BOBYQA method.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2015-2022, Éric Thiébaut
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module Bobyqa

export
    bobyqa,
    bobyqa!

using Printf

using ...Lib: opk_index

import
    ...Lib,
    ..AbstractContext,
    ..getreason,
    ..getstatus,
    ..grow!,
    ..iterate,
    ..restart,
    ..Scale,
    ..defaultscale,
    ..to_scale

# Aliases.
const Status               = Lib.bobyqa_status
const SUCCESS              = Lib.BOBYQA_SUCCESS
const BAD_NVARS            = Lib.BOBYQA_BAD_NVARS
const BAD_NPT              = Lib.BOBYQA_BAD_NPT
const BAD_RHO_RANGE        = Lib.BOBYQA_BAD_RHO_RANGE
const BAD_SCALING          = Lib.BOBYQA_BAD_SCALING
const TOO_CLOSE            = Lib.BOBYQA_TOO_CLOSE
const ROUNDING_ERRORS      = Lib.BOBYQA_ROUNDING_ERRORS
const TOO_MANY_EVALUATIONS = Lib.BOBYQA_TOO_MANY_EVALUATIONS
const STEP_FAILED          = Lib.BOBYQA_STEP_FAILED

# Get a textual explanation of the status returned by BOBYQA.
function getreason(status::Status)
    cstr = Lib.bobyqa_reason(status)
    if cstr == C_NULL
        error("unknown BOBYQA status: ", status)
    end
    unsafe_string(cstr)
end

"""
# Minimizing a function of many variables with bound constraints

Mike Powell's **BOBYQA** algorithm attempts to find the variables `x` which
solve the bound constrained problem:

    min f(x)  s.t.  xl ≤ x ≤ xu

where `x` is a vector of variables that has `n ≥ 2` components `f(x)` is an
objective function, `xl` and `xu` are bounds on the variables. The algorithm
employs quadratic approximations to the objective which interpolate the
objective function at `m` points, the value `m = 2n + 1` being recommended. The
parameter `rho` controls the size of the trust region and it is reduced
automatically from `rhobeg` to `rhoend` (such that `0 < rhoend ≤ rhobeg`).

The in-place version of the algorithm is called as:

    Bobyqa.minimize!(f, x, xl, xu, rhobeg, rhoend) -> (status, x, fx)

where `f` is the objective function, `x` is a vector with the initial and final
variables, `xl` and `xu` the lower and upper bounds for the variables, `rhobeg`
and `rhoend` are the initial and final sizes of the trust region. The result is
a tuple of 3 values: `status` indicates whether the algorithm was successful,
`x` is the final value of the variables and `fx = f(x)` is the objective
function at `x`. Normally, `status` should be `Bobyqa.SUCCESS`; otherwise,
`getreason(status)` yields a textual explanation about the failure. The initial
variables must be feasible, that is:

    xl ≤ x ≤ xu

must hold on entry.

The method:

    Bobyqa.minimize(f, x0, xl, xu, rhobeg, rhoend) -> (status, x, fx)

is identical to `Bobyqa.minimize!` but does not modify the vector `x0` of
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

An error occurs if any of the differences `xu[i] - xl[i]` is less than
`2*rhobeg*scale[i]`.


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
  of the objective function. The default setting is the recommended value: `npt
  = 2n + 1` with `n = length(x)` the number of variables.

* `work` specifies a workspace to (re)use. It must be a vector of double
  precision floating-point values. If it is too small, its size is
  automatically adjusted (by calling [`resize!`](@ref)). This keyword is useful
  to avoid any new allocation (and garbage colection) when several similar
  optimizations are to be performed.


## References

The algorithm is described in:

* M.J.D. Powell, "The BOBYQA algorithm for bound constrained optimization
  without derivatives", Technical Report NA2009/06 of the Department of Applied
  Mathematics and Theoretical Physics, Cambridge, England (2009).

"""
minimize(args...; kwds...) = optimize(args...; maximize=false, kwds...)
minimize!(args...; kwds...) = optimize!(args...; maximize=false, kwds...)
@doc @doc(minimize) minimize!

"""
    Bobyqa.maximize(f, x0, xl, xu, rhobeg, rhoend) -> (status, x, fx)
    Bobyqa.maximize!(f, x, xl, xu, rhobeg, rhoend) -> (status, x, fx)

are similar to `Bobyqa.minimize` and `Bobyqa.minimize!` respectively but
solve the bound constrained maximization problem:

    max f(x)  s.t.  xl ≤ x ≤ xu

"""
maximize(args...; kwds...) = optimize(args...; maximize=true, kwds...)
maximize!(args...; kwds...) = optimize!(args...; maximize=true, kwds...)
@doc @doc(maximize) maximize!

# `_wrklen(...)` yields the number of elements in BOBYQA workspace.
_wrklen(n::Integer, npt::Integer) = _wrklen(Int(n), Int(npt))
_wrklen(n::Int, npt::Int) = (npt + 5)*(npt + n) + div(3*n*(n + 5),2)
_wrklen(x::AbstractVector{<:AbstractFloat}, npt::Integer) =
    _wrklen(length(x), npt)
function _wrklen(x::AbstractVector{<:AbstractFloat},
                 scl::AbstractVector{<:AbstractFloat},
                 npt::Integer)
    return _wrklen(x, npt) + 3*length(scl)
end

# `_work(...)` yields a large enough workspace for BOBYQA.
_work(x::AbstractVector{<:AbstractFloat}, npt::Integer) =
    Vector{Cdouble}(undef, _wrklen(x, npt))
function _work(x::AbstractVector{<:AbstractFloat},
               scl::AbstractVector{<:AbstractFloat},
               npt::Integer)
    return Vector{Cdouble}(undef, _wrklen(x, scl, npt))
end

# Wrapper for the objective function in BOBYQA, the actual objective function
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

    Bobyqa.optimize(fc, x0, xl, xu, rhobeg, rhoend) -> (status, x, fx)
    Bobyqa.optimize!(fc, x, xl, xu, rhobeg, rhoend) -> (status, x, fx)

are identical to `Bobyqa.minimize` and `Bobyqa.minimize!` respectively but have
an additional `maximize` keyword which is `false` by default and which
specifies whether to maximize the objective function; otherwise, the method
attempts to minimize the objective function.

"""
optimize(f::Function, x0::AbstractVector{<:Real}, args...; kwds...) =
    optimize!(f, copyto!(Array{Cdouble}(undef, length(x0)), x0),
              args...; kwds...)

function optimize!(f::Function, x::DenseVector{Cdouble},
                   xl::DenseVector{Cdouble}, xu::DenseVector{Cdouble},
                   rhobeg::Real, rhoend::Real;
                   scale::Scale = defaultscale,
                   maximize::Bool = false,
                   npt::Integer = 2*length(x) + 1,
                   check::Bool = false,
                   verbose::Integer = 0,
                   maxeval::Integer = 30*length(x),
                   work::Vector{Cdouble} = _work(x, scale, npt))
    n = length(x)
    length(xl) == n || error("bad length for inferior bound")
    length(xu) == n || error("bad length for superior bound")
    scl = to_scale(scale, n)
    grow!(work, _wrklen(x, scl, npt))
    status = Lib.bobyqa_optimize(
        n, npt, (maximize ? Cint(1) : Cint(0)), _objfun_c[], f,
        x, xl, xu, scl, rhobeg, rhoend, verbose, maxeval, work)
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

@doc @doc(optimize) optimize!

# Basic version similar to the FORTRAN version.
bobyqa(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    bobyqa!(f, copy(x0), args...; kwds...)

function bobyqa!(f::Function, x::DenseVector{Cdouble},
                 xl::DenseVector{Cdouble}, xu::DenseVector{Cdouble},
                 rhobeg::Real, rhoend::Real;
                 npt::Integer = 2*length(x) + 1,
                 verbose::Integer = 0,
                 maxeval::Integer = 30*length(x),
                 check::Bool = true,
                 work::Vector{Cdouble} = _work(x, npt))
    n = length(x)
    length(xl) == n || error("bad length for inferior bound")
    length(xu) == n || error("bad length for superior bound")
    grow!(work, _wrklen(x, npt))
    status = Lib.bobyqa(
        n, npt, _objfun_c[], f, x, xl, xu,
        rhobeg, rhoend, verbose, maxeval, work)
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

end # module Bobyqa
