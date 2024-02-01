"""

Module `QuasiNewton` implements limited memory quasi-Newton methods for
`OptimPack`.

"""
module QuasiNewton

export
    VMLMB,
    configure!,
    get_reason,
    issuccess,
    minimize,
    minimize!,
    vmlmb,
    vmlmb!,
    vmlmb_CUTEst

# Imports from other packages.
using LinearAlgebra
using Printf
using TypeUtils, Unitless
using NumOptBase:
    NumOptBase,
    Bound,
    BoundedSet,
    PlusOrMinus,
    combine!,
    copy!,
    inner,
    is_bounded_below, is_bounded_above,
    linesearch_stepmax,
    norm2,
    project_direction!,
    project_variables!,
    scale!,
    unblocked_variables!
import NumOptBase: apply!, update!

# Imports from parent module.
using  ..OptimPackNextGen
using  OptimPackNextGen.LineSearches
using  OptimPackNextGen:
    auto_differentiate!,
    copy_variables,
    is_positive
import OptimPackNextGen:
    configure!,
    get_reason,
    minimize!,
    minimize,
    scalar_type,
    variables_type

#------------------------------------------------------------------------------

@enum Status begin
    LINESEARCH_FAILURE    = -2
    NOT_POSITIVE_DEFINITE = -1
    UNSTARTED_ALGORITHM   =  0
    TOO_MANY_EVALUATIONS  =  1
    TOO_MANY_ITERATIONS   =  2
    FTEST_SATISFIED       =  3
    GTEST_SATISFIED       =  4
    XTEST_SATISFIED       =  5
end

"""
    stats = OptimPack.QuasiNewton.Stats{T}(; fval::Real, gnorm::Real, step::Real,
                                             seconds::Real, iters::Integer,
                                             evals::Integer, rejects::Integer,
                                             status::Status)

builds an immutable object collecting information returned by a quasi-Newton
method. Type parameter `T` is the floating-point type for `fval`, `gnorm`, and
`step`. All members are mandatory and are specified by keyword:

- `fval` is the objective function value.
- `gnorm` is the Euclidean norm of the (projected) gradient.
- `step` is the line-search step length.
- `seconds` is the execution time in seconds.
- `iters` is the number of iterations.
- `evals` is the number of objective function (and gradient) evaluations.
- `rejects` is the number of rejected updates.
- `status` is the algorithm status.

The properties of the `stats` object are the same as these keywords.

"""
struct Stats{T<:AbstractFloat}
    fval::T          # Objective function value.
    gnorm::T         # Euclidean norm of the (projected) gradient.
    step::T          # Line-search step length.
    seconds::Float64 # Execution time in seconds.
    iters::Int       # Number of iterations.
    evals::Int       # Number of objective function evaluations.
    rejects::Int     # Number of rejected updates.
    status::Status   # Algorithm status.
    function Stats{T}(; fval::Real, gnorm::Real, step::Real, seconds::Real,
                        iters::Integer, evals::Integer, rejects::Integer,
                        status::Status) where {T<:AbstractFloat}
        return new{T}(fval, gnorm, step, seconds, iters, evals, rejects, status)
    end
end
function Stats(; fval::Real, gnorm::Real, step::Real, kwds...)
    T = float(promote_type(typeof(fval), typeof(gnorm), typeof(step)))
    return Stats{T}(; fval, gnorm, step, kwds...)
end

LinearAlgebra.issuccess(stats::Stats) = issuccess(stats.status)
LinearAlgebra.issuccess(status::Status) = Integer(status) > 0

get_reason(stats::Stats) = get_reason(stats.status)
get_reason(status::Status) =
    status == LINESEARCH_FAILURE    ? "error in line-search" :
    status == NOT_POSITIVE_DEFINITE ? "L-BFGS operator (or preconditioner) is not positive definite" :
    status == UNSTARTED_ALGORITHM   ? "algorithm not yet started" :
    status == TOO_MANY_EVALUATIONS  ? "too many evaluations" :
    status == TOO_MANY_ITERATIONS   ? "too many iterations" :
    status == FTEST_SATISFIED       ? "function reduction test satisfied" :
    status == GTEST_SATISFIED       ? "(projected) gradient test satisfied" :
    status == XTEST_SATISFIED       ? "variables change test satisfied" :
    "unknown status"

#------------------------------------------------------------------------------

"""
    OptimPack.QuasiNewton.LBFGS{T,X}

is the type of the object storing the limited memory BFGS approximation of the
Hessian of the objective function. Parameter `T` is the floating-point type for
scalars in computations and parameter `X` is the type of the variables.

"""
mutable struct LBFGS{T,X}
    m::Int           # maximum number of previous step to memorize
    mp::Int          # current number of memorized steps
    mrk::Int         # index of last update
    gamma::T         # gradient scaling
    S::Vector{X}     # memorized variable changes
    Y::Vector{X}     # memorized gradient changes
    rho::Vector{T}   # memorized values of s'⋅y
    alpha::Vector{T} # memorized values of `α`
end

"""
    OptimPack.QuasiNewton.LBFGS{T}(x, m) -> A

yields an object storing the limited memory BFGS approximation of the Hessian
of the objective function. Parameter `T` is the floating-point type for scalars
in computations; if not specified, it is determined from the element type of
`x`. Argument `x` is the variables and argument `m` is the number of previous
steps to memorize.

"""
function LBFGS(x::AbstractArray, m::Integer)
    # Determine floating-point type for scalars, at least `Float64` to limit
    # rounding errors in computations.
    T = promote_type(Float64, real_type(eltype(x)))
    return LBFGS{T}(x, m)
end

function LBFGS{T}(x::X, m::Integer) where {T<:AbstractFloat,X<:AbstractArray}
    float(eltype(x)) === eltype(x) || throw(ArgumentError(
        "variables type `$(eltype(x))` is not floating-point"))
    m ≥ zero(m) || throw(ArgumentError(
        "invalid number of previous step to memorize"))
    return LBFGS{T,X}(m,
                      0,                   # mp
                      0,                   # mrk
                      zero(T),             # gamma
                      workspace(x, m),     # S
                      workspace(x, m),     # Y
                      Vector{T}(undef, m), # rho
                      Vector{T}(undef, m)) # alpha
end

"""
    OptimPack.QuasiNewton.reset!(A::OptimPack.QuasiNewton.LBFGS) -> A

resets the object `A` storing the limited memory BFGS approximation of the
Hessian of the objective function and returns `A`. This amounts to forgetting
the memorized previous steps and thus restarting the LBFGS recurrence.

"""
function reset!(A::LBFGS)
    A.mp = 0
    A.mrk = 0
    A.gamma = zero(A.gamma)
    return A
end

"""
    OptimPack.QuasiNewton.update!(A::OptimPack.QuasiNewton.LBFGS, s, y) -> bool

updates the information stored by L-BFGS instance `A`. Arguments `s` and `y`
are the change in variables and gradient for the last iterate. The returned
value is a boolean indicating whether `s` and `y` were suitable to update an
L-BFGS approximation that be positive definite.

Even if `A.m = 0`, the value of the optimal gradient scaling `A.gamma` is
updated if possible (i.e., when `true` is returned).

See also [`OptimPack.QuasiNewton.LBFGS`](@ref).

"""
function update!(A::LBFGS{T}, s, y) where {T}
    sty = inner(T, s, y)
    accept = is_positive(sty)
    if accept
        A.gamma = sty/inner(T, y, y)
        if A.m ≥ 1
            mrk = next_slot(A)
            copy!(A.S[mrk], s)
            copy!(A.Y[mrk], y)
            A.rho[mrk] = sty
            A.mrk = mrk
            A.mp = min(A.mp + 1, A.m)
        end
    end
    return accept
end

"""
    OptimPack.QuasiNewton.index(A::OptimPack.QuasiNewton.LBFGS, j::Int) -> i

yields the index where the `j`-th previous step is memorized in L-BFGS instance
`A`. Argument `j` must be in `1:A.mp` where `A.mp` is the actual number of
memorized steps.

"""
index(A::LBFGS, j::Int) = ((A.mrk + A.m) - j)%A.m + 1

"""
    OptimPack.QuasiNewton.next_slot(A::OptimPack.QuasiNewton.LBFGS) -> i

yields the index where the next variables and gradient changes will be
memorized in L-BFGS instance `A`.

"""
next_slot(A::LBFGS) = (A.mrk % A.m) + 1 # FIXME check that = index(A, 0)

"""
    OptimPack.QuasiNewton.precond!(A, d) -> d

applies in-place the preconditionner `A` to the search direction `d` and
returns `d`.

If `A` is a scalar or an array of weights, `d` is simply scaled by `A`.
Otherwise, it is assumed that `A` is callable as `A(d)` to perform the
preconditonning. This latter rule may be overridden by specializing
`precond!(A, d)` for the type of `A` and, perhaps, `d`.

"""
precond!(α::Number, d) = scale!(d, α)
precond!(w::AbstractArray{<:Any,N}, d::AbstractArray{<:Any,N}) where {N} =
    lmul!(d, Diag(w), d)
function precond!(f, d)
    f(d)
    return d
end

"""
    OptimPack.QuasiNewton.precond!(γ, H₀, d) -> d

this syntax is for the L-BFGS recurence: if `H₀` not `nothing`, it is used to
precondition the search direction `d`; otherwise, if `γ > 0`, `d` is caled by
`γ`; otherwise `d` is left unchanged.

"""
precond!(gamma::Number, H0::Any, d) = precond!(H0, d)
function precond!(gamma::Number, H0::Nothing, d)
    if gamma > zero(gamma) && !isone(gamma)
        scale!(d, gamma)
    end
    return d
end

"""
    OptimPack.QuasiNewton.apply!(dst, A::OptimPack.QuasiNewton.LBFGS, b;
                                 freevars=nothing, precond=nothing) -> np

applies the L-BFGS approximation of the inverse Hessian stored by `A` to the
"vector" `b` and store the result in `dst`. On return, `np` is the number of
previous steps used in the L-BFGS recurrence. If `np = 0`, then `d` is only
affected by the preconditioner and by the list of free variables.

Keyword `freevars` is to restrict the L-BFGS approximation to the sub-space
spanned by the "free variables" not blocked by the constraints. If specified
and not `nothing`, `freevars` shall have the size as `d` and shall be equal to
zero where variables are blocked and to one elsewhere, then, on return, `d[i] =
0` if the `i`-th variable is blocked according to `freevars`.

Keyword `precond` is to specify another preconditioner than the default one
which consists in a uniform scaling of the variables.

"""
function apply!(dst, A::LBFGS{T}, b; freevars=nothing, precond=nothing) where {T}
    # Apply the 2-loop L-BFGS recursion algorithm by Matthies & Strang.
    cnt = A.mp # assume all memorized steps are acceptable
    if freevars === nothing || !iszero(minimum(freevars))
        # All variables are unconstrained, apply the regular L-BFGS recursion.
        copy!(dst, b)
        for j ∈ 1:A.mp
            i = index(A, j)
            A.alpha[i] = inner(T, dst, A.S[i])/A.rho[i]
            update!(dst, -A.alpha[i], A.Y[i])
        end
        precond!(A.gamma, precond, dst)
        for j ∈ A.mp:-1:1
            i = index(A, j)
            beta = inner(T, dst, A.Y[i])/A.rho[i];
            update!(dst, A.alpha[i] - beta, A.S[i])
        end
    else
        # L-BFGS recursion on a subset of free variables specified by a
        # selection of indices.
        lmul!(dst, Diag(freevars), b) # restrict argument to the subset of free variables
        rho = similar(A.rho) # FIXME
        wrk = similar(dst)     # FIXME
        gamma = zero(A.gamma)
        for j ∈ 1:A.mp
            i = index(A, j)
            Yᵢ = lmul!(wrk, Diag(freevars), A.Y[i])
            rho[i] = inner(T, A.S[i], Yᵢ) # FIXME inner(T, wgt, A.S[i], A.Y[i])
            if is_positive(rho[i])
                if !is_positive(gamma)
                    gamma = as(typeof(gamma), rho[i]/inner(T, Yᵢ, Yᵢ)) # FIXME norm2(Diag(wgt), A.Y[i])
                end
                A.alpha[i] = inner(T, dst, A.S[i])/rho[i]
                update!(dst, -alpha[i], yᵢ)
            else
                cnt -= one(cnt) # this previous step is not usable
            end
        end
        precond!(gamma, precond, dst)
        if cnt ≥ 1
            for j ∈ A.mp:-1:1
                i = index(A, j)
                if is_positive(rho[i])
                    beta = inner(T, dst, A.Y[i])/rho[i]
                    Sᵢ = lmul!(wrk, Diag(freevars), A.S[i])
                    update!(dst, A.alpha[i] - beta, Sᵢ) # FIXME update!(dst, ..., Diag(wgt), A.S[i])
                end
            end
        end
    end
    return cnt
end

#------------------------------------------------------------------------------

const default_mem      = 5
const default_lower    = nothing
const default_upper    = nothing
const default_fmin     = -Inf
const default_f2nd     = NaN
const default_dxrel    = 1e-4
const default_dxabs    = 1.0
const default_epsilon  = 0.0
const default_xtol     = (0.0, 1.0e-7)
const default_ftol     = (-Inf, 1.0e-8)
const default_gtol     = (0.0, 1.0e-6)
const default_maxiter  = typemax(Int)
const default_maxeval  = typemax(Int)
const default_verb     = 0
const default_output   = stdout
const default_observer = nothing

function default_printer(output::IO, x, stats::Stats)
    if stats.iters == 0
        @printf(output, "#%s%s\n#%s%s\n",
                " ITERS   EVALS  REJECTS",
                "             F(X)           ||G(X)||    STEP",
                "-----------------------",
                "----------------------------------------------")
    end
    @printf(output, "%7d %7d  %7d  %24.16E %9.2E %9.2E\n",
            stats.iters, stats.evals, stats.rejects, stats.fval, stats.gnorm,
            stats.step)
end

# `workspace(A, n)` yields a vector of `n` arrays similar to `A`.
function workspace(A::AbstractArray, n::Integer)
    wrk = Vector{typeof(A)}(undef, n)
    @inbounds for i in eachindex(wrk)
        wrk[i] = similar(A)
    end
    return wrk
end

# `workspace(A; empty=false)` yields an array similar to `A` except that
# its dimensions are all zero if `empty` is true.
workspace(A::AbstractArray; empty::Bool=false) =
    similar(A, empty ? ntuple(Returns(0), Val(ndims(A))) : size(A))

get_tolerances(rtol::Number; atol::T) where {T<:Number} =
    (atol, as(T, rtol))

get_tolerances(tol::Tuple{Number,Number}; atol::T) where {T<:Number} =
    (as(T, tol[1]), as(T, tol[2]))

mutable struct VMLMB{T<:AbstractFloat,X<:AbstractArray,A,B,L}
    # VMLMB context. T is floating-point type, X is type of variables,
    # B is l-BFGS type, L is linesearch type.
    variant::Symbol # method variant (one of `:LBFGS`, `:BLMVM`, or `:VMLMB`).
    g::X            # gradient
    g0::X           # gradient at start of line-search
    x0::X           # variables at start of line-search
    d::X            # ascent search direction
    s::X            # effective step
    pg::X           # projected gradient
    pg0::X          # projected gradient at start of line search (for BLMVM)
    best_x::X       # best solution found so far
    best_g::X       # gradient at best solution
    freevars::X     # subset of free variables (not for LBFGS)
    wrk_x::X
    wrk_alpha::Vector{T}
    wrk_rho::Vector{T}
    lbfgs::A  # FIXME: LBFGS{T,X}
    bounds::B # FIXME: BoundedSet{eltype(X),ndims(X)}
    lnsrch::L # FIXME: LineSearch{T}
    fmin::T
    f2nd::T
    dxabs::T
    dxrel::T
    epsilon::T
    fatol::T
    frtol::T
    gatol::T
    grtol::T
    xatol::T
    xrtol::T
end

"""
    ctx = OptimPack.QuasiNewton.VMLMB(x; kwds...)

builds a context object `ctx` with all settings and workspaces to run
VMLMB algorithm with variables `x`.

The input variables `x` needs not be properly initialized but must be a valid
instance in the sense that `similar(x)` yields a suitable object to work with.

All parameters (except the objective function and the initial variables) that may
influence the algorithm are stored into `ctx`.  Call:

    OptimPack.configure!(ctx; kwds...) -> ctx

to change the configurable settings.

"""
function VMLMB(x::AbstractArray; kwds...)
    T = scalar_type(x)
    return VMLMB{T}(x; kwds...)
end

function VMLMB{T}(x::X;
                  mem::Integer = default_mem,
                  lower = default_lower,
                  upper = default_upper,
                  blmvm::Bool = false,
                  fmin::Real = default_fmin,
                  f2nd::Real = default_f2nd,
                  dxabs::Real = default_dxabs,
                  dxrel::Real = default_dxrel,
                  epsilon::Real = default_epsilon,
                  xtol::Union{Real,NTuple{2,Real}} = default_xtol,
                  ftol::Union{Real,NTuple{2,Real}} = default_ftol,
                  gtol::Union{Real,NTuple{2,Real}} = default_gtol,
                  lnsrch::Union{LineSearch,Nothing} = nothing,
                  ) where {T<:AbstractFloat,X}
    # Check arguments.
    float(eltype(x)) === eltype(x) || throw(ArgumentError(
        "variables type `$(eltype(x))` is not floating-point"))
    mem = min(mem, length(x))

    # Extract tolerances.
    fatol, frtol = get_tolerances(ftol, atol=as(T, default_ftol[1]))
    gatol, grtol = get_tolerances(gtol, atol=as(T, default_gtol[1]))
    xatol, xrtol = get_tolerances(xtol, atol=as(T, default_xtol[1]))

    # If a line-search has been specified, make sure the line-search object is
    # of the correct type and is a private copy for this context; otherwise,
    # provide a default line search method. FIXME: Not type-stable in the
    # latter case.
    lnsrch = if lnsrch isa LineSearch
        LineSearch{T}(lnsrch)
    elseif is_bounded_below(lower) || is_bounded_above(upper)
        MoreToraldoLineSearch{T}(ftol=1e-3, gamma=(0.1,0.5))
    else
        MoreThuenteLineSearch{T}(ftol=1e-3, gtol=0.9, xtol=0.1)
    end

    # Build a bounded set.
    bounds = BoundedSet{eltype(x),ndims(x)}(lower, upper)

    # Determine the optimization method to emulate.
    variant = if ! is_bounded(bounds)
        :LBFGS
    elseif blmvm
        :BLMVM
    else
        :VMLMB
    end

    # Allocate workspaces.
    g        = workspace(x) # gradient
    g0       = workspace(x) # gradient at start of line-search
    x0       = workspace(x) # variables at start of line-search
    d        = workspace(x) # ascent search direction
    s        = workspace(x) # effective step
    pg       = workspace(x; # projected gradient
                         empty = variant === :LBFGS) # not needed if unconstrained
    pg0      = workspace(x; # projected gradient at start of line search
                         empty = variant !== :BLMVM) # only needed for BLMVM
    best_x   = workspace(x) # best solution found so far
    best_g   = workspace(x) # gradient at best solution
    freevars = workspace(x; # subset of free variables
                         empty = variant === :LBFGS) # not needed if unconstrained
    wrk_x    = workspace(x;
                         empty = variant === :LBFGS) # not needed if unconstrained
    wrk_alpha = Vector{T}(undef, mem)
    wrk_rho = Vector{T}(undef, mem)

    # Build L-BFGS operator
    lbfgs = LBFGS(x, mem)

    return VMLMB{T,X,typeof(lbfgs),typeof(bounds),typeof(lnsrch)}(
        variant,
        g,
        g0,
        x0,
        d,
        s,
        pg,
        pg0,
        best_x,
        best_g,
        freevars,
        wrk_x,
        wrk_alpha,
        wrk_rho,
        lbfgs,
        bounds,
        lnsrch,
        fmin,
        f2nd,
        dxabs, dxrel,
        epsilon,
        fatol, frtol,
        gatol, grtol,
        xatol, xrtol)
end

function configure!(ctx::VMLMB{T};
                    ftol::Union{Real,NTuple{2,Real}} = (ctx.fatol, ctx.frtol),
                    gtol::Union{Real,NTuple{2,Real}} = (ctx.gatol, ctx.grtol),
                    xtol::Union{Real,NTuple{2,Real}} = (ctx.xatol, ctx.xrtol),
                    fmin::Real = ctx.fmin,
                    f2nd::Real = ctx.f2nd,
                    dxabs::Real = ctx.dxabs,
                    dxrel::Real = ctx.dxrel,
                    epsilon::Real = ctx.epsilon) where {T}
    ctx.fatol, ctx.frtol = get_tolerances(ftol, atol=as(T, default_ftol[1]))
    ctx.gatol, ctx.grtol = get_tolerances(gtol, atol=as(T, default_gtol[1]))
    ctx.xatol, ctx.xrtol = get_tolerances(xtol, atol=as(T, default_xtol[1]))
    ctx.fmin = fmin
    ctx.f2nd = f2nd
    ctx.dxabs = dxabs
    ctx.dxrel = dxrel
    ctx.epsilon = epsilon
    return ctx
end

function Base.show(io::IO, ctx::VMLMB{T}) where {T}
    print(io, "VMLMB{", T, "}(#= variant = ", ctx.variant, ", ")
    print(io, "lnsrch = ")
    str = repr(typeof(ctx.lnsrch))
    print(io, SubString(str, nextind(str, findlast('.', str)), lastindex(str)))
    print(io, " =#; ")
    print(io, "fmin = ", ctx.fmin, ", ")
    print(io, "f2nd = ", ctx.f2nd, ", ")
    print(io, "dxabs = ", ctx.dxabs, ", ")
    print(io, "dxrel = ", ctx.dxrel, ", ")
    print(io, "epsilon = ", ctx.epsilon, ", ")
    print(io, "ftol = (", ctx.fatol, ", ", ctx.frtol, "), ")
    print(io, "gtol = (", ctx.gatol, ", ", ctx.grtol, "), ")
    print(io, "xtol = (", ctx.xatol, ", ", ctx.xrtol, "))")
end

"""
     vmlmb!(fg!, x; kwds...) -> x, stats

finds a local minimizer of `f(x)` starting at `x` and storing the best solution
in `x`. Method `vmlmb!` is the in-place version of [`vmlmb`](@ref) (which to see).

"""
function vmlmb!(fg!, x;
                # Context keywords (NOTE Must match keywords in VMLMB constructor).
                mem::Integer = default_mem,
                lower = default_lower,
                upper = default_upper,
                blmvm::Bool = false,
                fmin::Real = default_fmin,
                f2nd::Real = default_f2nd,
                dxabs::Real = default_dxabs,
                dxrel::Real = default_dxrel,
                epsilon::Real = default_epsilon,
                xtol::Union{Real,NTuple{2,Real}} = default_xtol,
                ftol::Union{Real,NTuple{2,Real}} = default_ftol,
                gtol::Union{Real,NTuple{2,Real}} = default_gtol,
                lnsrch::LineSearch = default_linesearch(; lower, upper),
                # Optimizer keywords (NOTE Must match keywords in minimize! method).
                precond = nothing,
                autodiff::Bool = false,
                maxiter::Integer = default_maxiter,
                maxeval::Integer = default_maxeval,
                verb::Integer = default_verb,
                output::IO = default_output,
                printer = default_printer,
                observer = default_observer)
    # Create one shot context.
    ctx = VMLMB(x; mem, lower, upper, blmvm, fmin, f2nd, dxrel, dxabs,
                epsilon, xtol, ftol, gtol, lnsrch)
    # Minimize objective finction.
    return minimize!(ctx, fg!, x; autodiff, maxiter, maxeval,
                     verb, output, printer, observer)
end

function minimize(ctx::VMLMB, fg!, x0; kwds...)
    x = copy!(similar(x0, eltype(variables_type(ctx))), x0)
    return minimize!(ctx, fg!, x; kwds...)
end

# The real worker.
function minimize!(ctx::VMLMB{T,X}, fg!, x::X;
                   precond = nothing,
                   autodiff::Bool = false,
                   maxiter::Integer = default_maxiter,
                   maxeval::Integer = default_maxeval,
                   verb::Integer = default_verb,
                   output::IO = default_output,
                   printer = default_printer,
                   observer = default_observer) where {T, X}

    # Check settings.
    @assert maxiter ≥ 0
    @assert maxeval ≥ 1

    # Work arrays.
    g = ctx.g               # gradient
    g0 = ctx.g0             # gradient at start of line-search
    x0 = ctx.x0             # variables at start of line-search
    d = ctx.d               # ascent search direction
    pg = ctx.pg             # projected gradient
    pg0 = ctx.pg0           # projected gradient at start of line search
    freevars = ctx.freevars # subset of free variables (not yet known)

    # Scalar parameters.
    f0 = T(Inf)             # function value at start of line-search
    pgnorm = T(NaN)         # Euclidean norm of the (projected) gradient
    alpha = zero(T)         # step length
    evals = 0               # number of calls to `fg`
    iters = 0               # number of iterations
    projs = 0               # number of projections onto the feasible set
    rejects = 0             # number of search direction rejections
    best_f = T(Inf)         # function value at `best_x`
    best_pgnorm = T(NaN)    # norm of projected gradient at `best_x` (< 0 if unknown)
    best_alpha = zero(T)    # step length at `best_x` (< 0 if unknown)
    best_evals = -1         # number of calls to `fg` at `best_x`
    last_evals = -1         # number of calls to `fg` at last iterate
    last_print = -1         # iteration number for last print
    last_obsrv = -1         # iteration number for last call to observer
    t0 = time()             # starting time
    dg = T(NaN)             # inner product of `d` and `g`
    gtest = T(NaN)          # threshold for convergence in the gradient
    status = UNSTARTED_ALGORITHM # algorithm is about to start

    # Algorithm stage follows that of the line-search, it is one of:
    # 0 = initially;
    # 1 = line-search in progress;
    # 2 = line-search has converged.
    stage = 0

    while true
        # Make the variables feasible.
        if is_constrained(ctx)
            # In principle, we can avoid projecting the variables whenever
            # `alpha ≤ amin` (because the feasible set is convex) but rounding
            # errors could make this wrong. It is safer to always project the
            # variables. This cost O(n) operations which are probably
            # negligible compared to, say, computing the objective function and
            # its gradient.
            project_variables!(ctx, x)
            projs += 1
        end
        # Compute the objective function and its gradient.
        if autodiff
            f = as(T, auto_differentiate!(fg!, x, g))
        else
            f = as(T, fg!(x, g))
        end
        evals += 1
        if f < best_f || evals == 1
            # Save best solution so far.
            best_f = f
            copy!(ctx.best_g, g)
            copy!(ctx.best_x, x)
            best_pgnorm = T(NaN) # must be recomputed
            best_alpha = alpha
            best_evals = evals
        end
        if stage != 0
            # Line-search in progress, check for line-search convergence.
            stage = linesearch_stage(iterate!(ctx.lnsrch, f, -dg))
            if stage == 2
                # Line-search has converged, `x` is the next iterate.
                iters += 1
                last_evals = evals
            elseif stage == 1
                # Line-search has not converged, peek next trial step.
                alpha = get_step(ctx.lnsrch) :: T
            else
                status = LINESEARCH_FAILURE
                break
            end
        end
        if stage != 1
            # Initial or next iterate after convergence of line-search.
            if is_constrained(ctx)
                # Determine the subset of free variables and compute the norm
                # of the projected gradient (needed to check for convergence).
                unblocked_variables!(freevars, ctx, x, -, g)
                lmul!(pg, Diag(freevars), g) # project gradient
                pgnorm = norm2(T, pg)
            else
                # Just compute the norm of the gradient.
                pgnorm = norm2(T, g)
            end
            if evals == best_evals
                # Now we know the norm of the (projected) gradient at the best
                # solution so far.
                best_pgnorm = pgnorm
            end
            # Check for algorithm convergence or termination.
            if evals == 1
                # Compute value for testing the convergence in the gradient.
                gtest = max(zero(T), ctx.gatol, ctx.grtol*pgnorm)
            end
            if pgnorm ≤ gtest
                # Convergence in gradient.
                status = GTEST_SATISFIED;
                break
            end
            if stage == 2
                # Check convergence in relative function reduction.
                if f ≤ ctx.fatol || abs(f - f0) ≤ ctx.frtol*max(abs(f), abs(f0))
                    status = FTEST_SATISFIED;
                    break
                end
                # Compute the effective change of variables.
                combine!(ctx.s, x, -, x0)
                snorm = norm2(T, ctx.s)
                # Check convergence in variables.
                if snorm ≤ ctx.xatol || snorm ≤ ctx.xrtol*norm2(T, x)
                    status = XTEST_SATISFIED
                    break
                end
            end
            if iters ≥ maxiter
                status = TOO_MANY_ITERATIONS
                break
            end
        end
        if evals ≥ maxeval
            status = TOO_MANY_EVALUATIONS
            break
        end
        if stage != 1
            # Initial iteration or line-search has converged.
            call_printer = (verb > 0 && (iters % verb) == 0)
            if observer !== nothing || call_printer
                stats = Stats(; iters, evals, rejects, status, fval = f,
                              gnorm = pgnorm, step = alpha, seconds = time() - t0)
                # Call user defined observer.
                if observer !== nothing
                    observer(x, stats)
                    last_obsrv = iters
                end
                # Possibly print iteration information.
                if call_printer
                    printer(output, x, stats)
                    last_print = iters
                end
            end
            if stage != 0
                # At least one step has been performed, L-BFGS approximation
                # can be updated. Using `d` as a temporary workspace to store
                # the gradient change (`d` is recomputed next).
                let y = d
                    if is_blmvm(ctx)
                        combine!(y, pg, -, pg0)
                    else
                        combine!(y, g, -, g0)
                    end
                    update!(ctx.lbfgs, ctx.s, y)
                end
            end
            # Use L-BFGS approximation to compute a search direction.
            np = if is_blmvm(ctx)
                apply!(d, ctx.lbfgs, pg; precond, freevars)
            elseif is_constrained(ctx)
                apply!(d, ctx,lbfgs, g; precond, freevars);
            else # unconstrained L-BGFS
                apply!(d, ctx.lbfgs, g; precond)
            end
            # Check whether `d` is an acceptable ascent direction.
            # Parameter `dir` is set to:
            #   0 if `d` is not a search direction,
            #   1 if `d` is an unscaled steepest ascent,
            #   2 if `d` is a scaled sufficient ascent.
            dir = 2 # assume no rescaling needed
            dg = inner(T, d, g) # FIXME not like in ref. implementation
            # FIXME  dg = as(T, NaN) # value of inner(d, g) not yet known
            if np < 1
                # No exploitable curvature information, `d` is the opposite of
                # the unscaled steepest descent feasible direction, that is the
                # projected gradient.
                dir = 1 # rescaling needed
            else
                # Some exploitable curvature information were available.
                if dg ≤ zero(dg)
                    # L-BFGS approximation does not yield an ascent direction.
                    # Normally this can only occurs for constrained
                    # optimization. For unconstrained optimization, this
                    # indicates that the preconditioner is not positive
                    # definite or that rounding errors make the L-BFGS
                    # approximation non-positive definite.
                    dir = 0 # discard search direction
                    if !is_constrained(ctx) && precond !== nothing
                        status = NOT_POSITIVE_DEFINITE
                        break
                    end
                elseif ctx.epsilon > 0 && dg < ctx.epsilon*norm2(T, d)*pgnorm
                    # A more restrictive criterion has been specified for
                    # accepting an ascent direction.
                    dir = 0 # discard search direction
                end
            end
            if dir == 0
                # No exploitable information about the Hessian is available or
                # the direction computed using the L-BFGS approximation failed
                # to be a sufficient ascent direction. Take the steepest
                # feasible ascent direction.
                if is_constrained(ctx)
                    lmul!(d, Diag(freevars), g) # take the projected gradient
                else
                    copy!(d, g) # take the gradient
                end
                dg = pgnorm^2
                dir = 1 # rescaling needed
            end
            if dir != 2 && iters > 0
                # L-BFGS search direction has been rejected.
                rejects += 1
            end
            # Determine the length `alpha` of the initial step along `d`.
            if dir == 2
                # The search direction is already scaled.
                alpha = one(T)
            else
                # Find a suitable step size along the steepest feasible
                # direction `d`. Note that `pgnorm`, the Euclidean norm of the
                # (projected) gradient, is also that of `d` in that case.
                alpha = steepest_descent_step(ctx, x, pgnorm, f)
            end
            if is_constrained(ctx)
                # Safeguard the step to avoid searching in a region where
                # all bounds are overreached.
                alpha = min(alpha, linesearch_stepmax(ctx, x, -, d))
            end
            # Initialize line-search.
            stage = linesearch_stage(start!(ctx.lnsrch, f, -dg, alpha))
            if stage != 1
                status = LINESEARCH_ERROR
                break
            end
            # Save iterate at start of line-search.
            f0 = f
            copy!(g0, g)
            copy!(x0, x)
            if is_blmvm(ctx)
                copy!(pg0, pg)
            end
        end
        # Compute next iterate.
        combine!(x, 1, x0, -alpha, d)
    end

    # In case of abnormal termination, some progresses may have been made since
    # the start of the line-search. In that case, we restore the best solution
    # so far.
    if best_f < f
        f = best_f;
        copy!(g, ctx.best_g)
        copy!(x, ctx.best_x)
        alpha = best_alpha
        if !isnan(best_pgnorm)
            # No needs to recompute the norm of the (projected) gradient.
            pgnorm = best_pgnorm
        elseif is_constrained(ctx)
            # Recompute the projected gradient and its norm.
            unblocked_variables!(freevars, ctx, x, -, g)
            lmul!(pg, Diag(freevars), g)
            pgnorm = norm2(T, pg)
        else
            # Recompute the gradient norm.
            pgnorm = norm2(T, g)
        end
        if f < f0
            # Some progresses since last iterate, pretend that one more
            # iteration has been performed.
            iters += 1
        end
    end
    stats = Stats(; iters, evals, rejects, status, fval = f,
                    gnorm = pgnorm, step = alpha, seconds = time() - t0)
    if observer !== nothing && iters > last_obsrv
        observer(x, stats)
    end
    if verb > 0
        if iters > last_print
            printer(output, x, stats)
        end
        println(output, "# Termination: ", get_reason(status))
    end
    return x, stats
end

# Traits.
for trait in (:scalar_type, :variables_type)
    @eval $trait(ctx::VMLMB) = $trait(typeof(ctx))
end
scalar_type(::Type{<:VMLMB{T,X}}) where {T,X} = T
variables_type(::Type{<:VMLMB{T,X}}) where {T,X} = X

# Determine suitable floating-point type to work with variables `x`.
scalar_type(x::AbstractArray) = scalar_type(typeof(x))
scalar_type(::Type{T}) where {T<:AbstractArray} =
    promote_type(Float64, real_type(eltype(T)))


is_lbfgs(ctx::VMLMB) = ctx.variant === :LBFGS
is_blmvm(ctx::VMLMB) = ctx.variant === :BLMVM
is_vmlmb(ctx::VMLMB) = ctx.variant === :VMLMB
is_unconstrained(ctx::VMLMB) = is_lbfgs(ctx)
is_constrained(ctx::VMLMB) = !is_unconstrained(ctx)
is_bounded(Ω::BoundedSet) = is_bounded_below(Ω.lower) | is_bounded_above(Ω.upper)
is_bounded(ctx::VMLMB) = is_bounded(ctx.bounds)

LineSearches.steepest_descent_step(ctx::VMLMB, x, d, fx) =
    steepest_descent_step(scalar_type(ctx), x, d, fx;
                          f2nd = ctx.f2nd, fmin = ctx.fmin,
                          dxrel = ctx.dxrel, dxabs = ctx.dxabs)

# Extend `NumOptBase` methods for VMLMB.
NumOptBase.linesearch_stepmax(ctx::VMLMB, x, pm::PlusOrMinus, d) =
    as(scalar_type(ctx), linesearch_stepmax(x, pm, d, ctx.bounds))

NumOptBase.project_variables!(ctx::VMLMB, x) =
    project_variables!(x, x, ctx.bounds)

NumOptBase.unblocked_variables!(ctx::VMLMB, x, pm::PlusOrMinus, d) =
    unblocked_variables!(ctx.freevars, x, pm, d, ctx.bounds)

linesearch_stage(state::Symbol) =
    state === :SEARCHING ? 1 :
    state === :CONVERGENCE || state === :WARNING ? 2 : -1

"""
    vmlmb(fg!, x0; upper=u, lower=ℓ, kwds...) -> x, stats

attempts to solve the constrained problem:

    min f(x)   subject to   x ∈ Ω   with   Ω = { x ∈ ℝⁿ | ℓ ≤ x ≤ u }

by the VMLMB method, a Variable Metric method with Limited Memory and optional
Bound constraints. Argument `fg!` implements the objective function `f(x)` and
its gradient `∇f(x)`. Argument `x0 ∈ ℝⁿ` gives an initial approximation of the
variables (its contents is left unchanged and it does not need to be feasible).
The result is a 2-tuple `(x, stats)` with `x ∈ Ω` and `stats` a structure with
information about the algorithm computations. Provided `issuccess(stats)` is
true, `x` is an approximate local minimizer of the objective function on the
feasible set `Ω`.

The caller must provide a function `fg!` to compute the value and the gradient
of the objective function as follows:

    fx = fg!(x, gx)

where `x` stores the current variables, `fx` is the value of the function at
`x` and the contents of `gx` has to be overwritten with the gradient at `x`
(when `fg!` is called, `gx` is already allocated). The best solution found so
far is returned in `x`.

Another possibility is to specify keyword `autodiff = true` and rely on
automatic differentiation to compute the gradient:

    x = vmlmb(f, x0; autodiff=true, kwds...)

where `f` is a simpler function that takes the variables `x` as a single
argument and returns the value of the objective function:

    fx = f(x)

The method [`OptimPackNextGen.auto_differentiate!`](@ref) is called to compute
the gradient of the objective function, say `f`. This method may be extended
for the specific type of `f`. An implementation of `auto_differentiate!` is
provided by `OptimPackNextGen` if the `Zygote` package is loaded.

The following keywords are available:

* `mem` specifies the amount of storage as the number of previous steps
  memorized to build an L-BFGS model of the inverse Hessian of the objective
  function. By default `mem = $default_mem`.

* `lower` and `upper` specify the lower and upper bounds for the variables. A
   bound can be a scalar to indicate that all variables have the same bound
   value. If the lower (resp. upper) bound is unspecified or set to `-∞` (resp.
   `+∞`), the variables are assumed to be unbounded below (resp. above). If no
   bounds are set, VMLMB is an unconstrained limited memory BFGS method
   (L-BFGS).

* `autodiff` is a boolean specifying whether to rely on automatic
  differentiation by calling [`OptimPackNextGen.auto_differentiate!](@ref). If
  not specified, this keyword is assumed to be `false`. You may use:

      autodiff = !applicable(fg!, x0, x0)

  to attempt to guess whether automatic differentiation is needed.

* `xtol` is a tuple of two nonnegative reals specifying respectively the
  absolute and relative tolerances for deciding convergence on the variables.
  Convergence occurs if the Euclidean norm of the the difference between
  successive iterates is less or equal `max(xtol[1], xtol[2]*norm2(x))`. By
  default, `xtol = $default_xtol`.

* `ftol` is a tuple of two nonnegative reals specifying respectively the
  absolute and relative errors desired in the function. Convergence occurs if
  the absolute error between `f(x)` and `f(xsol)` is less than `ftol[1]` or if
  the estimate of the relative error between `f(x)` and `f(xsol)`, where `xsol`
  is a local minimizer, is less than `ftol[2]`. By default, `ftol =
  $default_ftol`.

* `gtol` is a tuple of two nonnegative reals specifying the absolute and a
  relative thresholds for the norm of the gradient, convergence is assumed as
  soon as:

      ||g(x)|| ≤ hypot(gtol[1], gtol[2]*||g(x0)||)

  where `||g(x)||` is the Euclidean norm of the gradient at the current
  solution `x`, `||g(x0)||` is the Euclidean norm of the gradient at the
  starting point `x0`. By default, `gtol = $default_gtol`.

* `fmin` specifies a lower bound for the function. If provided, `fmin` is used
  to estimate the steepest desxecnt step length this value. The algorithm exits
  with a warning if `f(x) < fmin`.

* `maxiter` specifies the maximum number of iterations.

* `maxeval` specifies the maximum number of calls to `fg!`.

* `verb` specifies the verbosity level. It can be a boolean to specify whether
  to call the observer at every iteration or an integer to call the observer
  every `verb` iteration(s). The observer is never called if `verb` is less or
  equal zero. The default is `verb = false`.

* `observer` can be set with a callable object to print iteration information
  or inspect the current iterate, its signature is:

      observer(output::IO, stats, x)

  where `output` is the output stream, `stats` collects information about the
  current iteration (see below), and `x` is the current iterate.

* `output` specifies the output stream for printing information (`stdout` is
  used by default).

* `lnsrch` specifies the method to use for line searches (the default line
   search is `MoreThuenteLineSearch`).

* `blmvm` can be set true to emulate the BLMVM algorithm of Benson and Moré.
  This option has no effects for an unconstrained problem.

The `stats` object has the following properties:

* `stats.fx` is the objective function value at `x`.

* `stats.gnorm` is the norm of the gradient of the objective function at `x`.

* `stats.step` is the length of the line-search step to the current point.

* `stats.seconds` is the execution time in seconds.

* `stats.iter` is the number of iterations, `0` for the starting point.

* `stats.eval` is the number of function (and gradient) evaluations.

* `stats.rejects` is the number of times the computed direction was
  rejected and the L-BFGS recursion restarted.

* `stats.status` indicates the status of the algorithm.


### History

The VMLMB algorithm in
[OptimPackNextGen](https://github.com/emmt/OptimPackNextGen.jl) provides a pure
Julia implementation of the original method (Thiébaut, 2002) with some
improvements and the capability to emulate L-BFGS and BLMVM methods.

The limited memory BFGS method (L-BFGS) was first described by Nocedal (1980)
who dubbed it SQN. The method is implemented in MINPACK-2 (1995) by the FORTRAN
routine VMLM. The numerical performances of L-BFGS have been studied by Liu and
Nocedal (1989) who proved that it is globally convergent for uniformly convex
problems with a R-linear rate of convergence. They provided the FORTRAN code
LBFGS. The BLMVM and VMLMB algorithms were proposed by Benson and Moré (2001)
and Thiébaut (2002) to account for separable bound constraints on the
variables. These two latter methods are rather different than L-BFGS-B by Byrd
at al. (1995) which has more overheads and is slower in practice.

* J. Nocedal, "*Updating Quasi-Newton Matrices with Limited Storage*" in
  Mathematics of Computation, vol. 35, pp. 773-782 (1980).

* D.C. Liu & J. Nocedal, "*On the limited memory BFGS method for large scale
  optimization*" in Mathematical programming, vol. 45, pp. 503-528 (1989).

* R.H. Byrd, P. Lu, J. Nocedal, & C. Zhu, "*A limited memory algorithm for
  bound constrained optimization*" in SIAM Journal on Scientific Computing,
  vol. 16, pp. 1190-1208 (1995).

* S.J. Benson & J.J. Moré, "*A limited memory variable metric method in
  subspaces and bound constrained optimization problems*" in Subspaces and
  Bound Constrained Optimization Problems (2001).

* É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*" in
  Astronomical Data Analysis II, Proc. SPIE 4847, pp. 174-183 (2002).

"""
vmlmb(fg!, x0; kwds...) = vmlmb!(fg!, copy_variables(x0); kwds...)

"""
    vmlmb_CUTEst(name; kwds...) -> x, stats

yields the solution to the `CUTEst` problem `name` by the VMLMB method. This
require to have loaded the `CUTest` package.

"""
vmlmb_CUTEst(args...; kwds...) =
    error("invalid arguments or `CUTEst` package not yet loaded")

end # module
