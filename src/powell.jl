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
# Copyright (C) 2015-2022, Éric Thiébaut
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module Powell

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

import Base: ==, iterate

abstract type AbstractContext end

"""
    iterate(ctx, ...) -> status

performs the next iteration of the reverse communication algorithm associated
with the context `ctx`.  Other arguments depend on the type of algorithm.

For **COBYLA** algorithm, the next iteration is performed by:

    iterate(ctx, f, x, c) -> status

or

    iterate(ctx, f, x) -> status

on entry, the context status must be `COBYLA_ITERATE`, `f` and `c` are the
function value and the constraints at `x`, the latter can be omitted if there
are no constraints.  On exit, the returned value (the new context status) is:
`COBYLA_ITERATE` if a new trial point has been stored in `x` and if user is
requested to compute the function value and the constraints on the new point;
`COBYLA_SUCCESS` if algorithm has converged and `x` has been set with the
variables at the solution (the corresponding function value can be retrieved
with `getlastf`); anything else indicates an error (see `getreason` for an
explanatory message).

For **NEWUOA** algorithm, the next iteration is performed by:

    iterate(ctx, f, x) -> status

on entry, the context status must be `NEWUOA_ITERATE`, `f` is the function
value at `x`.  On exit, the returned value (the new context status) is:
`NEWUOA_ITERATE` if a new trial point has been stored in `x` and if user is
requested to compute the function value for the new point; `NEWUOA_SUCCESS` if
algorithm has converged; anything else indicates an error (see `getreason` for
an explanatory message).

"""
function iterate end

"""
    restart(ctx) -> status

restarts the reverse communication algorithm associated with the context `ctx`
using the same parameters.  The return value is the new status of the
algorithm, see `getstatus` for details.

"""
function restart end

"""
    getstatus(ctx) -> status

yields the current status of the reverse communication algorithm associated
with the context `ctx`.  Possible values are:

* for **COBYLA**: `COBYLA_ITERATE`, if user is requested to compute `f(x)` and
  `c(x)`; `COBYLA_SUCCESS`, if algorithm has converged;

* for **NEWUOA**: `NEWUOA_ITERATE`, if user is requested to compute `f(x)`;
  `NEWUOA_SUCCESS`, if algorithm has converged;

Anything else indicates an error (see `getreason` for an explanatory message).

"""
function getstatus end

"""
    getreason(ctx) -> msg

or

    getreason(status) -> msg

yields an explanatory message about the current status of the reverse
communication algorithm associated with the context `ctx` or with the status
returned by an optimization method of by `getstatus(ctx)`.

"""
getreason(ctx::AbstractContext) = getreason(getstatus(ctx))

"""
    getlastf(ctx) -> fx

yields the last function value in the reverse communication algorithm
associated with the context `ctx`.  Upon convergence of `iterate`, this value
corresponds to the function at the solution; otherwise, this value corresponds
to the previous set of variables.

"""
function getlastf end

"""
    getncalls(ctx) -> nevals

yields the current number of function evaluations in the reverse communication
algorithm associated with the context `ctx`.  Result is -1 if something is
wrong, nonnegative otherwise.

"""
function getncalls end

"""
    getradius(ctx) -> rho

yields the current size of the trust region of the reverse communication
algorithm associated with the context `ctx`.  Result is 0 if algorithm has not
yet started (before first iteration), -1 if something is wrong, strictly
positive otherwise.

"""
function getradius end

"""
    grow!(x, n) -> x

grows vector `x` so that it has at least `n` elements, does nothing if `x` is
large enough.  Argument `x` is returned.

See also [`resize!`](@ref).

"""
function grow!(x::Vector, n::Integer)
    length(x) < n && resize!(x, n)
    return x
end

# Null-array to represent missing argument. It is sufficient to implement the abstract
# array API plus the Base.unsafe_convert method to return a null pointer.
struct NullArray{T,N} <: AbstractArray{T,N} end
const NullVector{T} = NullArray{T,1}
const NullMatrix{T} = NullArray{T,2}
Base.length(::NullArray{T,N}) where {T,N} = 0
Base.size(::NullArray{T,N}) where {T,N} = ntuple(Returns(0), Val(N))
Base.axes(::NullArray{T,N}) where {T,N} = ntuple(Returns(Base.OneTo(0)), Val(N))
Base.unsafe_convert(::Type{Ptr{S}}, ::NullArray{T,N}) where {T,N,S<:Union{Cvoid,T}} = Ptr{S}(0)

# Default scaling factors and allowed types for scaling factors.
const defaultscale = NullVector{Cdouble}()
const Scale = Union{Nothing,Real,AbstractVector{<:Real},typeof(defaultscale)}

# Convert the scaling factors.
function to_scale(scl::Union{Nothing,typeof(defaultscale)}, n::Integer)
    return defaultscale
end
function to_scale(scl::Real, n::Integer)
    (isfinite(scl) & (scl > zero(scl))) || throw(ArgumentError("invalid scaling factor"))
    return fill(Cdouble(scl), n)
end
function to_scale(scl::AbstractVector{<:Real}, n::Integer)
    length(scl) == n || throw(DimensionMismatch("bad number of scaling factors"))
    @inbounds for i in eachindex(scl)
        s = scl[i]
        (isfinite(s) & (s > zero(s))) || throw(ArgumentError("invalid scaling factor(s)"))
    end
    return scl isa DenseVector{Cdouble} ? scl : convert(Vector{Cdouble}, scl)
end

include("newuoa.jl")
import .Newuoa: newuoa, newuoa!

include("cobyla.jl")
import .Cobyla: cobyla, cobyla!

include("bobyqa.jl")
import .Bobyqa: bobyqa, bobyqa!

end # module Powell
