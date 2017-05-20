#
# Powell.jl --
#
# Mike Powell's derivative free optimization methods for Julia.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License:
#
# Copyright (C) 2015-2017, Éric Thiébaut.
#

module Powell

using Compat

# Locate the dynamic library.
if isfile(joinpath(dirname(@__FILE__),"..","deps","libs.jl"))
    include("../deps/libs.jl")
else
    error("OptimPackNextGen not properly installed.  Please create and edit file \"../deps/libs.jl\"")
end

export
    iterate,
    restart,
    getstatus,
    getreason,
    getradius,
    getncalls,
    getlastf,
    Cobyla,
    cobyla,
    cobyla!,
    Newuoa,
    newuoa,
    newuoa!,
    Bobyqa,
    bobyqa,
    bobyqa!

import Base: ==

@compat abstract type AbstractStatus end

@compat abstract type AbstractContext end

=={T<:AbstractStatus}(a::T, b::T) = a._code == b._code
==(a::AbstractStatus, b::AbstractStatus) = false

doc"""
The `iterate(ctx, ...)` method performs the next iteration of the reverse
communication associated with the context `ctx`.  Other arguments depend on the
type of algorithm.

For **COBYLA** algorithm, the next iteration is performed by:

    iterate(ctx, f, x, c) -> status

or

    iterate(ctx, f, x) -> status

on entry, the workspace status must be `COBYLA_ITERATE`, `f` and `c` are the
function value and the constraints at `x`, the latter can be omitted if there
are no constraints.  On exit, the returned value (the new workspace status) is:
`COBYLA_ITERATE` if a new trial point has been stored in `x` and if user is
requested to compute the function value and the constraints on the new point;
`COBYLA_SUCCESS` if algorithm has converged and `x` has been set with the
variables at the solution (the corresponding function value can be retrieved
with `getlastf`); anything else indicates an error (see `getreason`
for an explanatory message).


For **NEWUOA** algorithm, the next iteration is performed by:

    iterate(ctx, f, x) -> status

on entry, the wokspace status must be `NEWUOA_ITERATE`, `f` is the function
value at `x`.  On exit, the returned value (the new wokspace status) is:
`NEWUOA_ITERATE` if a new trial point has been stored in `x` and if user is
requested to compute the function value for the new point; `NEWUOA_SUCCESS` if
algorithm has converged; anything else indicates an error (see `getreason` for
an explanatory message).

"""
function iterate end

doc"""

    restart(ctx) -> status

restarts the reverse communication algorithm associated with the context `ctx`
using the same parameters.  The return value is the new status of the
algorithm, see `getstatus` for details.

"""
function restart end

doc"""

    getstatus(ctx) -> status

get the current status of the reverse communication algorithm associated with
the context `ctx`.  Possible values are:

* for **COBYLA**: `COBYLA_ITERATE`, if user is requested to compute `f(x)` and
  `c(x)`; `COBYLA_SUCCESS`, if algorithm has converged;

* for **NEWUOA**: `NEWUOA_ITERATE`, if user is requested to compute `f(x)`;
  `NEWUOA_SUCCESS`, if algorithm has converged;

Anything else indicates an error (see `getreason` for an explanatory message).

"""
function getstatus end

doc"""

    getreason(ctx) -> msg

or

    getreason(status) -> msg

get an explanatory message about the current status of the reverse
communication algorithm associated with the context `ctx` or with the status
returned by an optimization method of by `getstatus(ctx)`.

"""
function getreason end

getreason(ctx::AbstractContext) = getreason(getstatus(ctx))

doc"""

    getlastf(ctx) -> fx

get the last function value in the reverse communication algorithm associated
with the context `ctx`.  Upon convergence of `iterate`, this value corresponds
to the function at the solution; otherwise, this value corresponds to the
previous set of variables.

"""
function getlastf end

doc"""

    getncalls(ctx) -> nevals

get the current number of function evaluations in the reverse communication
algorithm associated with the context `ctx`.  Result is -1 if something is
wrong, nonnegative otherwise.

"""
function getncalls end

doc"""

    getradius(ctx) -> rho

get the current size of the trust region of the reverse communication algorithm
associated with the context `ctx`.  Result is 0 if algorithm not yet started
(before first iteration), -1 if something is wrong, strictly positive
otherwise.

"""
function getradius end

include("newuoa.jl")
importall .Newuoa

include("cobyla.jl")
importall .Cobyla

include("bobyqa.jl")
importall .Bobyqa

end # module Powell
