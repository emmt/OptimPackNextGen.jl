#
# newuoa.jl --
#
# Julia interface to Mike Powell's NEWUOA method.
#
# ----------------------------------------------------------------------------
#
# This file is part of OptimPack.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2015-2017, Éric Thiébaut.
#
# ----------------------------------------------------------------------------

module Newuoa

export
    newuoa,
    newuoa!

#import ..find_dll

#const DLL = find_dll("newuoa")

import ..libnewuoa
const DLL = libnewuoa

# FIXME: with Julia 0.5 all relative (prefixed by .. or ...) symbols must be
#        on the same line as `import`
import ..AbstractStatus, ..AbstractContext, ..getncalls, ..getradius, ..getreason, ..getstatus, ..iterate, ..restart

immutable Status <: AbstractStatus
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

where `x` is a vector of variables that has `n ≥ 2` components, and `f(x)` is
an objective function.  The algorithm employs quadratic approximations to the
objective which interpolate the objective function at `m` points, the value `m
= 2n + 1` being recommended.  The parameter `rho` controls the size of the
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

* `maxeval` set the maximum number of calls to the objective function.  The
  default setting is `maxeval = 30n` with `n = length(x)` the number of
  variables.

* `npt` specifies the number of points to use for the quadratic approximation
  of the objective function.  The default setting is the recommended value:
  `npt = 2n + 1` with `n = length(x)` the number of variables.


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

    Newuoa.maximize(f, x0, m, rhobeg, rhoend) -> (status, x, fx)
    Newuoa.maximize!(f, x, m, rhobeg, rhoend) -> (status, x, fx)

are similar to `Newuoa.minimize` and `Newuoa.minimize!` respectively but
solve the uncontrained maximization problem:

    max f(x)

"""
maximize(args...; kwds...) = optimize(args...; maximize=true, kwds...)
maximize!(args...; kwds...) = optimize!(args...; maximize=true, kwds...)
@doc @doc(maximize) maximize!

# Yield the number of elements in NEWUOA workspace.
_wslen(n::Integer, npt::Integer) =
    (npt + 13)*(npt + n) + div(3*n*(n + 3),2)

# Wrapper for the objective function in NEWUOA, the actual objective function
# is provided by the client data.
function _objfun(n::Cptrdiff_t, xptr::Ptr{Cdouble}, fptr::Ptr{Void})
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    convert(Cdouble, f(x))::Cdouble
end

const _objfun_c = cfunction(_objfun, Cdouble, (Cptrdiff_t, Ptr{Cdouble},
                                               Ptr{Void}))

"""
The methods:

    Newuoa.optimize(fc, x0, rhobeg, rhoend) -> (status, x, fx)
    Newuoa.optimize!(fc, x, rhobeg, rhoend) -> (status, x, fx)

are identical to `Newuoa.minimize` and `Newuoa.minimize!` respectively but have
an additional `maximize` keyword which is `false` by default and which
specifies whether to maximize the objective function; otherwise, the method
attempts to minimize the objective function.

"""
optimize(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    optimize!(f, copy(x0), args...; kwds...)

function optimize!(f::Function, x::DenseVector{Cdouble},
                   rhobeg::Real, rhoend::Real;
                   scale::DenseVector{Cdouble} = Array{Cdouble}(0),
                   maximize::Bool = false,
                   npt::Integer = 2*length(x) + 1,
                   check::Bool = true,
                   verbose::Integer = 0,
                   maxeval::Integer = 30*length(x))
    n = length(x)
    nw = _wslen(n, npt)
    nscl = length(scale)
    if nscl == 0
        sclptr = convert(Ptr{Cdouble}, C_NULL)
    elseif nscl == n
        sclptr = pointer(scale)
        nw += n
    else
        error("bad number of scaling factors")
    end
    work = Array{Cdouble}(nw)
    status = Status(ccall((:newuoa_optimize, DLL), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Cint, Ptr{Void},
                           Ptr{Void}, Ptr{Cdouble}, Ptr{Cdouble},
                           Cdouble, Cdouble, Cptrdiff_t, Cptrdiff_t,
                           Ptr{Cdouble}), n, npt, maximize, _objfun_c,
                          pointer_from_objref(f), x, sclptr, rhobeg,
                          rhoend, verbose, maxeval, work))
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

@doc @doc(optimize) optimize!

# Basic version similar to the FORTRAN version.
function newuoa!(f::Function, x::DenseVector{Cdouble},
                 rhobeg::Real, rhoend::Real;
                 npt::Integer = 2*length(x) + 1,
                 verbose::Integer = 0,
                 maxeval::Integer = 30*length(x),
                 check::Bool = true)
    n = length(x)
    work = Array{Cdouble}(_wslen(n, npt))
    status = Status(ccall((:newuoa, DLL), Cint,
                          (Cptrdiff_t, Cptrdiff_t, Ptr{Void}, Ptr{Void},
                           Ptr{Cdouble}, Cdouble, Cdouble, Cptrdiff_t,
                           Cptrdiff_t, Ptr{Cdouble}), n, npt, _objfun_c,
                          pointer_from_objref(f), x, rhobeg, rhoend,
                          verbose, maxeval, work))
    if check && status != SUCCESS
        error(getreason(status))
    end
    return (status, x, work[1])
end

newuoa(f::Function, x0::DenseVector{Cdouble}, args...; kwds...) =
    newuoa!(f, copy(x0), args...; kwds...)

# Context for reverse communication variant of NEWUOA.
type Context <: AbstractContext
    ptr::Ptr{Void}
    n::Int
    npt::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    verbose::Int
    maxeval::Int
end

"""

    using OptimPack.Powell
    ctx = Newuoa.create(n, rhobeg, rhoend; npt=..., verbose=..., maxeval=...)

creates a new reverse communication workspace for NEWUOA algorithm.  A typical
usage is:

    x = Array{Cdouble}(n)
    x[...] = ... # initial solution
    ctx = Newuoa.create(n, rhobeg, rhoend; verbose=1, maxeval=500)
    status = getstatus(ctx)
    while status == Newuoa.ITERATE
        fx = ...       # compute function value at X
        status = iterate(ctx, fx, x)
    end
    if status != Newuoa.SUCCESS
        println("Something wrong occured in NEWUOA: ", getreason(status))
    end

"""
function create(n::Integer, rhobeg::Real, rhoend::Real;
                       npt::Integer = 2*length(x) + 1,
                       verbose::Integer = 0,
                       maxeval::Integer = 30*length(x))
    ptr = ccall((:newuoa_create, DLL), Ptr{Void},
                (Cptrdiff_t, Cptrdiff_t, Cdouble, Cdouble,
                 Cptrdiff_t, Cptrdiff_t),
                n, npt, rhobeg, rhoend, verbose, maxeval)
    if ptr == C_NULL
        reason = (errno() == Base.Errno.ENOMEM
                  ? "insufficient memory"
                  : "invalid argument")
        error(reason)
    end
    ctx = Context(ptr, n, npt, rhobeg, rhoend, verbose, maxeval)
    finalizer(ctx, ctx -> ccall((:newuoa_delete, DLL), Void,
                                (Ptr{Void},), ctx.ptr))
    return ctx
end

function iterate(ctx::Context, f::Real, x::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    Status(ccall((:newuoa_iterate, DLL), Cint,
                       (Ptr{Void}, Cdouble, Ptr{Cdouble}),
                       ctx.ptr, f, x))
end

restart(ctx::Context) =
    Status(ccall((:newuoa_restart, DLL), Cint, (Ptr{Void},), ctx.ptr))

getstatus(ctx::Context) =
    Status(ccall((:newuoa_get_status, DLL), Cint, (Ptr{Void},),
                       ctx.ptr))

getncalls(ctx::Context) =
    Int(ccall((:newuoa_get_nevals, DLL), Cptrdiff_t, (Ptr{Void},), ctx.ptr))

getradius(ctx::Context) =
    ccall((:newuoa_get_rho, DLL), Cdouble, (Ptr{Void},), ctx.ptr)

function runtests(;revcom::Bool=false, scale::Real=1)
    # The Chebyquad test problem (Fletcher, 1965) for N = 2,4,6 and 8, with
    # NPT = 2N+1.
    function ftest(x::DenseVector{Cdouble})
        n = length(x)
        np = n + 1
        y = Array{Cdouble}(np, n)
        for j in 1:n
            y[1,j] = 1.0
            y[2,j] = x[j]*2.0 - 1.0
        end
        for i in 2:n
            for j in 1:n
                y[i+1,j] = y[2,j]*2.0*y[i,j] - y[i-1,j]
            end
        end
        f = 0.0
        iw = 1
        for i in 1:np
            sum = 0.0
            for j in 1:n
                sum += y[i,j]
            end
            sum /= n
            if iw > 0
                sum += 1.0/(i*i - 2*i)
            end
            iw = -iw
            f += sum*sum
        end
        return f
    end

    # Run the tests.
    rhoend = 1e-6
    for n = 2:2:8
        npt = 2*n + 1
        x = Array{Cdouble}(n)
        for i in 1:n
            x[i] = i/(n + 1)
        end
        rhobeg = x[1]*0.2
        @printf("\n\n    Results with N =%2d and NPT =%3d\n", n, npt)
        if revcom
            # Test the reverse communication variant.
            ctx = Newuoa.create(n, rhobeg, rhoend;
                                npt = npt, verbose = 2, maxeval = 5000)
            status = getstatus(ctx)
            while status == ITERATE
                fx = ftest(x)
                status = iterate(ctx, fx, x)
            end
            if status != SUCCESS
                println("Something wrong occured in NEWUOA: ",
                        getreason(status))
            end
        elseif scale != 1
            Newuoa.minimize!(ftest, x, rhobeg/scale, rhoend/scale;
                             scale = fill!(similar(x), scale),
                             npt = npt, verbose = 2, maxeval = 5000)
        else
            newuoa!(ftest, x, rhobeg, rhoend;
                    npt = npt, verbose = 2, maxeval = 5000)
        end
    end
end

end # module Newuoa
