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
# Copyright (C) 2015-2018, Éric Thiébaut.
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module Cobyla

export
    cobyla,
    cobyla!

using Compat
using Compat.Printf

import ..AbstractStatus, ..AbstractContext,
    ..getncalls, ..getradius, ..getreason, ..getstatus, ..iterate, ..restart,
    .._libcobyla

# The dynamic library implementing the method.
const _LIB = _libcobyla

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
    ptr = ccall((:cobyla_reason, _LIB), Ptr{UInt8}, (Cint,), status._code)
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

# Yield number of elements in COBYLA workspace.
_wslen(n::Integer, m::Integer) = n*(3*n + 2*m + 11) + 4*m + 6

# Wrapper for the objective function in COBYLA, the actual objective
# function is provided by the client data as a `jl_value_t*` pointer.
function _objfun(n::Cptrdiff_t, m::Cptrdiff_t, xptr::Ptr{Cdouble},
                 _c::Ptr{Cdouble}, fptr::Ptr{Cvoid})
    x = unsafe_wrap(Array, xptr, n)
    f = unsafe_pointer_to_objref(fptr)
    convert(Cdouble, (m > 0 ? f(x, unsafe_wrap(Array, _c, m)) : f(x)))::Cdouble
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
    optimize!(fc, copy!(Array{Cdouble}(undef, length(x0)), x0),
              args...; kwds...)

function optimize!(fc::Function, x::DenseVector{Cdouble},
                   m::Integer, rhobeg::Real, rhoend::Real;
                   scale::DenseVector{Cdouble} = Array{Cdouble}(undef, 0),
                   maximize::Bool = false,
                   check::Bool = false,
                   verbose::Integer = 0,
                   maxeval::Integer = 30*length(x))
    n = length(x)
    nscl = length(scale)
    if nscl == 0
        sclptr = convert(Ptr{Cdouble}, C_NULL)
    elseif nscl == n
        sclptr = pointer(scale)
    else
        error("bad number of scaling factors")
    end
    work = Array{Cdouble}(undef, _wslen(n, m))
    iact = Array{Cptrdiff_t}(undef, m + 1)
    status = Status(ccall((:cobyla_optimize, _LIB), Cint,
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
                 maxeval::Integer = 30*length(x))
    n = length(x)
    work = Array{Cdouble}(undef, _wslen(n, m))
    iact = Array{Cptrdiff_t}(undef, m + 1)
    status = Status(ccall((:cobyla, _LIB), Cint,
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

# Context for reverse communication variant of COBYLA.
mutable struct CobylaContext <: AbstractContext
    ptr::Ptr{Cvoid}
    n::Int
    m::Int
    rhobeg::Cdouble
    rhoend::Cdouble
    verbose::Int
    maxeval::Int
end

"""

    using OptimPack.Powell
    ctx = Cobyla.create(n, m, rhobeg, rhoend; verbose=0, maxeval=500)

creates a new reverse communication workspace for COBYLA algorithm.  A typical
usage is:

    x = Array{Cdouble}(undef, n)
    c = Array{Cdouble}(undef, m)
    x[...] = ... # initial solution
    ctx = Cobyla.create(n, m, rhobeg, rhoend, verbose=1, maxeval=500)
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
function create(n::Integer, m::Integer,
                       rhobeg::Real, rhoend::Real;
                       verbose::Integer=0, maxeval::Integer=500)
    if n < 2
        throw(ArgumentError("bad number of variables"))
    elseif m < 0
        throw(ArgumentError("bad number of constraints"))
    elseif rhoend < 0 || rhoend > rhobeg
        throw(ArgumentError("bad trust region radius parameters"))
    end
    ptr = ccall((:cobyla_create, _LIB), Ptr{Cvoid},
                (Cptrdiff_t, Cptrdiff_t, Cdouble, Cdouble,
                 Cptrdiff_t, Cptrdiff_t),
                n, m, rhobeg, rhoend, verbose, maxeval)
    if ptr == C_NULL
        reason = (errno() == Base.Errno.ENOMEM
                  ? "insufficient memory"
                  : "unexpected error")
        error(reason)
    end
    ctx = CobylaContext(ptr, n, m, rhobeg, rhoend, verbose, maxeval)
    finalizer(ctx, ctx -> ccall((:cobyla_delete, _LIB), Cvoid,
                                (Ptr{Cvoid},), ctx.ptr))
    return ctx
end

function iterate(ctx::CobylaContext, f::Real, x::DenseVector{Cdouble},
                 c::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    length(c) == ctx.m || error("bad number of constraints")
    Status(ccall((:cobyla_iterate, _LIB), Cint,
                 (Ptr{Cvoid}, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}),
                 ctx.ptr, f, x, c))
end

function iterate(ctx::CobylaContext, f::Real, x::DenseVector{Cdouble})
    length(x) == ctx.n || error("bad number of variables")
    ctx.m == 0 || error("bad number of constraints")
    Status(ccall((:cobyla_iterate, _LIB), Cint,
                 (Ptr{Cvoid}, Cdouble, Ptr{Cdouble}, Ptr{Cvoid}),
                 ctx.ptr, f, x, C_NULL))
end

restart(ctx::CobylaContext) =
    Status(ccall((:cobyla_restart, _LIB), Cint, (Ptr{Cvoid},), ctx.ptr))

getstatus(ctx::CobylaContext) =
    Status(ccall((:cobyla_get_status, _LIB), Cint, (Ptr{Cvoid},), ctx.ptr))

# Get the current number of function evaluations.  Result is -1 if
# something is wrong (e.g. CTX is NULL), nonnegative otherwise.
getncalls(ctx::CobylaContext) =
    Int(ccall((:cobyla_get_nevals, _LIB), Cptrdiff_t, (Ptr{Cvoid},), ctx.ptr))

getradius(ctx::CobylaContext) =
    ccall((:cobyla_get_rho, _LIB), Cdouble, (Ptr{Cvoid},), ctx.ptr)

getlastf(ctx::CobylaContext) =
    ccall((:cobyla_get_last_f, _LIB), Cdouble, (Ptr{Cvoid},), ctx.ptr)

function runtests(;revcom::Bool = false, scale::Real = 1.0)
    # Beware that order of operations may affect the result (whithin
    # rounding errors).  I have tried to keep the same ordering as F2C
    # which takes care of that, in particular when converting expressions
    # involving powers.
    prt(s) = println("\n       "*s)
    for nprob in 1:10
        if nprob == 1
            # Minimization of a simple quadratic function of two variables.
            prt("Output from test problem 1 (Simple quadratic)")
            n = 2
            m = 0
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = -1.0
            xopt[2] = 0.0
            ftest = (x::DenseVector{Cdouble}) -> begin
                r1 = x[1] + 1.0
                r2 = x[2]
                fc = 10.0*(r1*r1) + (r2*r2)
                return fc
            end
        elseif nprob == 2
            # Easy two dimensional minimization in unit circle.
            prt("Output from test problem 2 (2D unit circle calculation)")
            n = 2
            m = 1
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = sqrt(0.5)
            xopt[2] = -xopt[1]
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = x[1]*x[2]
                con[1] = 1.0 - x[1]*x[1] - x[2]*x[2]
                return fc
            end
        elseif nprob == 3
            # Easy three dimensional minimization in ellipsoid.
            prt("Output from test problem 3 (3D ellipsoid calculation)")
            n = 3
            m = 1
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = 1.0/sqrt(3.0)
            xopt[2] = 1.0/sqrt(6.0)
            xopt[3] = -0.33333333333333331
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = x[1]*x[2]*x[3]
                con[1] = 1.0 - (x[1]*x[1]) - 2.0*(x[2]*x[2]) - 3.0*(x[3]*x[3])
                return fc
            end
        elseif nprob == 4
            # Weak version of Rosenbrock's problem.
            prt("Output from test problem 4 (Weak Rosenbrock)")
            n = 2
            m = 0
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = -1.0
            xopt[2] = 1.0
            ftest = (x::DenseVector{Cdouble}) -> begin
                r2 = x[1]
                r1 = r2*r2 - x[2]
                r3 = x[1] + 1.0
                fc = r1*r1 + r3*r3
                return fc
            end
        elseif nprob == 5
            # Intermediate version of Rosenbrock's problem.
            prt("Output from test problem 5 (Intermediate Rosenbrock)")
            n = 2
            m = 0
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = -1.0
            xopt[2] = 1.0
            ftest = (x::DenseVector{Cdouble}) -> begin
                r2 = x[1]
                r1 = r2*r2 - x[2]
                r3 = x[1] + 1.0
                fc = r1*r1*10.0 + r3*r3
                return fc
            end
        elseif nprob == 6
            # This problem is taken from Fletcher's book Practical Methods
            # of Optimization and has the equation number (9.1.15).
            prt("Output from test problem 6 (Equation (9.1.15) in Fletcher)")
            n = 2
            m = 2
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = sqrt(0.5)
            xopt[2] = xopt[1]
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = -x[1] - x[2]
                r1 = x[1]
                con[1] = x[2] - r1*r1
                r1 = x[1]
                r2 = x[2]
                con[2] = 1.0 - r1*r1 - r2*r2
                return fc
            end
        elseif nprob == 7
            # This problem is taken from Fletcher's book Practical Methods
            # of Optimization and has the equation number (14.4.2).
            prt("Output from test problem 7 (Equation (14.4.2) in Fletcher)")
            n = 3
            m = 3
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = 0.0
            xopt[2] = -3.0
            xopt[3] = -3.0
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = x[3]
                con[1] = x[1]*5.0 - x[2] + x[3]
                r1 = x[1]
                r2 = x[2]
                con[2] = x[3] - r1*r1 - r2*r2 - x[2]*4.0
                con[3] = x[3] - x[1]*5.0 - x[2]
                return fc
            end
        elseif nprob == 8
            # This problem is taken from page 66 of Hock and Schittkowski's
            # book Test Examples for Nonlinear Programming Codes. It is
            # their test problem Number 43, and has the name Rosen-Suzuki.
            prt("Output from test problem 8 (Rosen-Suzuki)")
            n = 4
            m = 3
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = 0.0
            xopt[2] = 1.0
            xopt[3] = 2.0
            xopt[4] = -1.0
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                fc = (r1*r1 + r2*r2 + r3*r3*2.0 + r4*r4 - x[1]*5.0
                      - x[2]*5.0 - x[3]*21.0 + x[4]*7.0)
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                con[1] = (8.0 - r1*r1 - r2*r2 - r3*r3 - r4*r4 - x[1]
                          + x[2] - x[3] + x[4])
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                con[2] = (10.0 - r1*r1 - r2*r2*2.0 - r3*r3 - r4*r4*2.0
                          + x[1] + x[4])
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                con[3] = (5.0 - r1*r1*2.0 - r2*r2 - r3*r3 - x[1]*2.0
                          + x[2] + x[4])
                return fc
            end
        elseif nprob == 9
            # This problem is taken from page 111 of Hock and
            # Schittkowski's book Test Examples for Nonlinear Programming
            # Codes. It is their test problem Number 100.
            prt("Output from test problem 9 (Hock and Schittkowski 100)")
            n = 7
            m = 4
            xopt = Array{Cdouble}(undef, n)
            xopt[1] =  2.330499
            xopt[2] =  1.951372
            xopt[3] = -0.4775414
            xopt[4] =  4.365726
            xopt[5] = -0.624487
            xopt[6] =  1.038131
            xopt[7] =  1.594227
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                r1 = x[1] - 10.0
                r2 = x[2] - 12.0
                r3 = x[3]
                r3 *= r3
                r4 = x[4] - 11.0
                r5 = x[5]
                r5 *= r5
                r6 = x[6]
                r7 = x[7]
                r7 *= r7
                fc = (r1*r1 + r2*r2*5.0 + r3*r3 + r4*r4*3.0
                      + r5*(r5*r5)*10.0 + r6*r6*7.0 + r7*r7
                      - x[6]*4.0*x[7] - x[6]*10.0 - x[7]*8.0)
                r1 = x[1]
                r2 = x[2]
                r2 *= r2
                r3 = x[4]
                con[1] = (127.0 - r1*r1*2.0 - r2*r2*3.0 - x[3]
                          - r3*r3*4.0 - x[5]*5.0)
                r1 = x[3]
                con[2] = (282.0 - x[1]*7.0 - x[2]*3.0 - r1*r1*10.0
                          - x[4] + x[5])
                r1 = x[2]
                r2 = x[6]
                con[3] = (196.0 - x[1]*23.0 - r1*r1 - r2*r2*6.0
                          + x[7]*8.0)
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                con[4] = (r1*r1*-4.0 - r2*r2 + x[1]*3.0*x[2]
                          - r3*r3*2.0 - x[6]*5.0 + x[7]*11.0)
                return fc
            end
        elseif nprob == 10
            # This problem is taken from page 415 of Luenberger's book
            # Applied Nonlinear Programming. It is to maximize the area of
            # a hexagon of unit diameter.
            prt("Output from test problem 10 (Hexagon area)")
            n = 9
            m = 14
            xopt = fill!(Array{Cdouble}(undef, n), 0.0)
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = -0.5*(x[1]*x[4] - x[2]*x[3] + x[3]*x[9] - x[5]*x[9]
                           + x[5]*x[8] - x[6]*x[7])
                r1 = x[3]
                r2 = x[4]
                con[1] = 1.0 - r1*r1 - r2*r2
                r1 = x[9]
                con[2] = 1.0 - r1*r1
                r1 = x[5]
                r2 = x[6]
                con[3] = 1.0 - r1*r1 - r2*r2
                r1 = x[1]
                r2 = x[2] - x[9]
                con[4] = 1.0 - r1*r1 - r2*r2
                r1 = x[1] - x[5]
                r2 = x[2] - x[6]
                con[5] = 1.0 - r1*r1 - r2*r2
                r1 = x[1] - x[7]
                r2 = x[2] - x[8]
                con[6] = 1.0 - r1*r1 - r2*r2
                r1 = x[3] - x[5]
                r2 = x[4] - x[6]
                con[7] = 1.0 - r1*r1 - r2*r2
                r1 = x[3] - x[7]
                r2 = x[4] - x[8]
                con[8] = 1.0 - r1*r1 - r2*r2
                r1 = x[7]
                r2 = x[8] - x[9]
                con[9] = 1.0 - r1*r1 - r2*r2
                con[10] = x[1]*x[4] - x[2]*x[3]
                con[11] = x[3]*x[9]
                con[12] = -x[5]*x[9]
                con[13] = x[5]*x[8] - x[6]*x[7]
                con[14] = x[9]
                return fc
            end
        else
            error("bad problem number ($nprob)")
        end

        x = Array{Cdouble}(undef, n)
        for icase in 1:2
            fill!(x, 1.0)
            rhobeg = 0.5
            rhoend = (icase == 2 ? 1e-4 : 0.001)
            if revcom
                # Test the reverse communication variant.
                c = Array{Cdouble}(undef, max(m, 0))
                ctx = Cobyla.create(n, m, rhobeg, rhoend;
                                    verbose = 1, maxeval = 2000)
                status = getstatus(ctx)
                while status == Cobyla.ITERATE
                    if m > 0
                        # Some constraints.
                        fx = ftest(x, c)
                        status = iterate(ctx, fx, x, c)
                    else
                        # No constraints.
                        fx = ftest(x)
                        status = iterate(ctx, fx, x)
                    end
                end
                if status != Cobyla.SUCCESS
                    println("Something wrong occured in COBYLA: ",
                            getreason(status))
                end
            elseif scale == 1
                cobyla!(ftest, x, m, rhobeg, rhoend;
                        verbose = 1, maxeval = 2000)
            else
                Cobyla.minimize!(ftest, x, m, rhobeg/scale, rhoend/scale;
                                 scale = fill!(Array{Cdouble}(undef, n), scale),
                                 verbose = 1, maxeval = 2000)
            end
            if nprob == 10
                tempa = x[1] + x[3] + x[5] + x[7]
                tempb = x[2] + x[4] + x[6] + x[8]
                tempc = 0.5/sqrt(tempa*tempa + tempb*tempb)
                tempd = tempc*sqrt(3.0)
                xopt[1] = tempd*tempa + tempc*tempb
                xopt[2] = tempd*tempb - tempc*tempa
                xopt[3] = tempd*tempa - tempc*tempb
                xopt[4] = tempd*tempb + tempc*tempa
                for i in 1:4
                    xopt[i + 4] = xopt[i]
                end
            end
            temp = 0.0
            for i in 1:n
                r1 = x[i] - xopt[i]
                temp += r1*r1
            end
            @printf("\n     Least squares error in variables =%16.6E\n", sqrt(temp))
        end
        @printf("  ------------------------------------------------------------------\n")
    end
end

end # module Cobyla
