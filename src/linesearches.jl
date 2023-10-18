"""

Module `OptimPackNextGen.LineSearches` implements line-search methods.

"""
module LineSearches

export
    ArmijoLineSearch,
    LineSearch,
    MoreThuenteLineSearch,
    MoreToraldoLineSearch,
    configure!,
    get_descr,
    get_reason,
    get_state,
    get_step,
    iterate!,
    start!,
    use_derivatives

# Imports from other packages.
using TypeUtils

# Imports from parent module.
import ..OptimPackNextGen
using OptimPackNextGen: Float, get_reason

# Dictionary associating symbolic information about the state of an iterative
# algorithm and a descriptive message.
const DESCR = Dict{Symbol,String}(
    :NOT_STARTED => "Algorithm not yet started",
    :F_LE_FMIN => "`f(x) ≤ fmin`",
    :COMPUTE_F => "Caller must compute `f(x)`",
    :COMPUTE_FG => "Caller must compute `f(x)` and `g(x)`",
    :NEW_X => "A new iterate is available for examination",
    :FINAL_X => "A solution has been found within tolerances",
    :FATOL_TEST_SATISFIED => "`fatol` test satisfied",
    :FRTOL_TEST_SATISFIED => "`frtol` test satisfied",
    :GATOL_TEST_SATISFIED => "`gatol` test satisfied",
    :GRTOL_TEST_SATISFIED => "`grtol` test satisfied",
    :FIRST_WOLFE_SATISFIED =>"First Wolfe condition satisfied",
    :STRONG_WOLFE_SATISFIED => "Strong Wolfe conditions both satisfied",
    :STP_EQ_STPMIN => "Step at lower bound",
    :STP_EQ_STPMAX => "Step at upper bound",
    :XTOL_TEST_SATISFIED => "`xtol` test satisfied",
    :ROUNDING_ERRORS => "Rounding errors prevent progress",
    :NOT_DESCENT => "Search direction is not a descent direction",
    :INIT_STP_LE_ZERO => "Initial step ≤ 0"
)

"""
    LineSearch{T}

is the super-type of objects implementing line-search methods. Type parameter
`T` is the floating-point type for the computations.

For an introduction to line-search methods, see ["*Line search methods*" by
Lihe Cao, Zhengyi Sui, Jiaqi Zhang, Yuqing Yan, and Yuhui Gu
(2021)](https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods).

For maximum flexibility, line-searches methods in `OptimPackNextGen` use
reverse communication. Assuming `SomeLineSearch` is a concrete line-search
type, a typical line-search is performed as follows:

    # Create an instance of the line-search method:
    ls = SomeLineSearch{T}(args...; kwds...)

    # Start the line-search and loop until a step satisfying some convergence
    # conditions are satisfied:
    x0 = ...            # initial variables
    f0 = func(x)        # function value at x0
    g0 = grad(x)        # gradient at x0
    d = ...             # search direction
    dtg0 = vdot(d, g0)  # directional derivative at x0
    stp1 = ...          # first step to try
    stpmin = ...        # lower bound for the step (usually zero)
    stpmax = ...        # upper bound for the step (usually a large number)
    state = start!(ls, f0, dtg0, stp1; stpmin, stpmax)
    while state == :SEARCHING
        stp = get_step(ls) # get step length to try
        x = x0 + stp*d     # compute trial point
        f = func(x)        # function value at x
        g = grad(x)        # gradient at x
        dtg = vdot(d, g)   # directional derivative at x
        state = iterate!(ls, f, dtg)
    end
    if state != :CONVERGENCE
        if state == :WARNING
            @warn get_reason(ls)
        else
            error(get_reason(ls))
        end
    end

Note that the same line-search instance may be re-used for subsequent
line-searches with the same settings. To change line-search settings call
`configure!(ls; kwds...)`.

"""
abstract type LineSearch{T<:AbstractFloat} end

"""
    get_step(ls)

yields the length of the next step to try in the line-search implemented by
`ls`.

"""
get_step(ls::LineSearch) = ls.stp

"""
    get_state(ls::LineSearch)

yields the current state in the line-search implemented by `ls`. The returned
value is one of:

- `:STARTING` until line-search is started;
- `:SEARCHING` while line-search is in progress;
- `:CONVERGENCE` when line-search has converged;
- `:WARNING` when line-search terminated with a warning;
- `:ERROR` when line-search terminated with an error.

"""
get_state(ls::LineSearch) = ls.state

"""
    get_descr(ls::LineSearch)

yields a symbolic description of the current state of the line-search
implemented by `ls`.

"""
get_descr(ls::LineSearch) = ls.descr

"""
    set_descr(sym::Symbol, str::AbstractString)

associates textual description `str` of the state of an iterative algorithm to
symbolic description `sym`.

"""
function set_descr(sym::Symbol, str::AbstractString)
    old = get(DESCR, sym, nothing)
    if old === nothing
        DESCR[sym] = str
    elseif old == str
        @warn "`:$sym` has already been associated with a textual description"
    else
        error("`:$sym` is already associated with a different description")
    end
    return nothing
end

function OptimPackNextGen.get_reason(ls::LineSearch)
    sym = get_descr(ls)
    str = get(DESCR, sym, nothing)
    return str === nothing ? "No description available for `:$sym`" : str
end

# Helper function.
function report!(ls::LineSearch, state::Symbol, descr::Symbol)
    ls.descr = descr
    ls.state = state
    return state
end

"""
    use_derivatives(ls)

yields whether the line-search instance `ls` requires the derivative of the
objective function when calling `iterate!`. Alternatively `ls` can also be the
line-search type. Note that the derivative is always needed by the `start!`
method.

"""
use_derivatives(::T) where {T<:LineSearch} = use_derivatives(T)

# The following methods are to provide the default step bounds given `stp1`,
# the first step to try. They may be extended for specific line-search methods.
default_stpmin(ls::LineSearch{T}, stp1::Real) where {T} = zero(T)
default_stpmax(ls::LineSearch{T}, stp1::Real) where {T} = typemax(T)

"""
    start!(ls, fx0, dgx0, stp1; stpmin = 0, stpmax = Inf) -> state

starts a new line-search with the method implemented by the line-search
instance `ls`, the returned value is the updated state of the algorithm (see
[`iterate!`](@ref) for the interpretation of this value). Arguments `fx0` and
`dgx0` are:

    fx0 = f(x0)
    dgx0 = d'·∇f(x0)

the value and the directional derivative of the objective function `f(x)` at
`x0`, the variables at start of the line-search, and with `d` the search
direction. Argument `stp1 > 0` is the length of the first step to try. Keywords
`stpmin` and `stpmax` can be used to specify bounds for the length of the step.

At a lower level, method:

    start!(ls::L, fx0::T, dgx0::T, stp1::T, stpmin::T, stpmax::T)

shall be specialized for line-search methods of type `L<:LineSearch{T}` and is
called with checked arguments.

"""
function start!(ls::LineSearch{T}, fx0::Real, dgx0::Real, stp1::Real;
                stpmin::Real = default_stpmin(ls, stp1),
                stpmax::Real = default_stpmax(ls, stp1)) where {T<:AbstractFloat}
    # This high-level version provides default values for keywords, checks the
    # input arguments for errors, and call the lower level specialized method
    # with converted arguments. Note that the tests fail if any value is NaN.
    zero(stpmin) ≤ stpmin ≤ stpmax || throw(ArgumentError(
        "`0 ≤ stpmin ≤ stpmax` must hold"))
    stpmin ≤ stp1 ≤ stpmax || throw(ArgumentError(
        "`stpmin ≤ stp1 ≤ stpmax` must hold"))
    dgx0 ≤ zero(dgx0) || throw(ArgumentError("Not a descent direction"))
    return start!(ls, as(T, fx0), as(T, dgx0), as(T, stp1), as(T, stpmin), as(T, stpmax))
end
@noinline function start!(ls::L, fx0::T, dgx0::T, stp1::T, stpmin::T,
                          stpmax::T) where {T<:AbstractFloat,L<:LineSearch{T}}
    error("low-level method `start!` is not implemented for line-search of type `$L`")
end

"""
    iterate!(ls, fx, dgx) -> state

performs one iteration of the line-search implemented by `ls`. Arguments `fx`
and `dgx` are:

    fx = f(x0 + α⋅d)
    dgx = d'·∇f(x0 + α⋅d)

the value and directional derivative of the objective function `f(x)` for the
step `α = get_step(ls)` and where `x0` and `d` are the variables at the start
of the line-search and the search direction.

The returned `state` is one of the following symbolic values:

* `:SEARCHING` to indicate that the serach is still in progress. The caller
  should take the new trial step, given by `get_step(ls)`, and call `iterate!`
  with the updated values of the objective function and, if needed, of its
  directional derivative.

* `:CONVERGENCE` to indicate that the line-search criterion holds.

* `:WARNING` to indicate that no further progresses are possible.

* `:ERROR` to indicate that an error has occurred.

"""
function iterate!(ls::LineSearch{T}, fx::Real, dgx::Real) where {T<:AbstractFloat}
    # This version is to convert arguments.
    return iterate!(ls, as(T, fx), as(T, dgx))
end
@noinline function iterate!(ls::L, fx::T, dgx::T) where {T<:AbstractFloat,
                                                         L<:LineSearch{T}}
    error("method `iterate!` is not implemented for line-search of type `$L`")
end

#-------------------------------------------------------------------------------

const default_armijo_tol = 1.0e-4

"""
    ArmijoLineSearch{T = $Float}(; ftol = $default_armijo_tol) -> ls

yields a line-search instance for floating-point type `T` which implements
Armijo's method. The objective of Armijo's method is to find a step `stp` that
satisfies the sufficient decrease condition (1st Wolfe condition):

    f(stp) ≤ f(0) + ftol*stp*f'(0),

where `stp` is smaller or equal the initial step (backtracking). The method
consists in reducing the step (by a factor 2) until the criterion holds.
Keyword `ftol` specifies the positive tolerance for the sufficient decrease
condition.

The method is described in:

* L. Armijo, "*Minimization of functions having Lipschitz continuous first
  partial derivatives*" in Pacific Journal of Mathematics, vol. **16**, pp. 1–3
  (1966).

"""
mutable struct ArmijoLineSearch{T<:AbstractFloat} <: LineSearch{T}
    state::Symbol
    descr::Symbol
    ftol::T
    finit::T
    ginit::T
    stp::T
    stpmin::T
    function ArmijoLineSearch{T}(
        ; ftol::Real = default_armijo_tol) where {T<:AbstractFloat}
        zero(ftol) < ftol ≤ 1//2 || throw(ArgumentError("`0 < ftol ≤ 1/2` must hold"))
        return new{T}(:STARTING, :NOT_STARTED, ftol, 0, 0, 0, 0)
    end
end

# Armijo's line-search does not use the directional derivative to refine the
# step.
use_derivatives(::Type{<:ArmijoLineSearch}) = false

# start! method for Armijo's line-search has the same implementation as the
# Moré & Toraldo line-search.

function iterate!(ls::ArmijoLineSearch{T}, f::T, g::T) where {T<:AbstractFloat}
    @assert ls.state == :SEARCHING
    # FIXME @assert stp == ls.stp
    if f ≤ ls.finit + ls.ftol*ls.stp*ls.ginit
        return report!(ls, :CONVERGENCE, :FIRST_WOLFE_CONDITION_SATISFIED)
    elseif ls.stp > ls.stpmin
        ls.stp = max(ls.stp/2, ls.stpmin)
        return report!(ls, :SEARCHING, :COMPUTE_F)
    else
        ls.stp = ls.stpmin
        return report!(ls, :WARNING, :STP_EQ_STPMIN)
    end
end

#-------------------------------------------------------------------------------

const default_more_toraldo_ftol   = default_armijo_tol,
const default_more_toraldo_gamma  = (0.1, 0.5)
const default_more_toraldo_strict = false

"""
    MoreToraldoLineSearch{T=$Float}(; kwds...) -> ls

yields a backtracking line-search proposed by Moré & Toraldo (1991) and which
is based on a safeguarded quadratic interpolation. Parameter `T` is the
floating-point type for the computations.

Keyword `ftol` can be used to specify a nonnegative tolerance for the
sufficient decrease condition. By default, `ftol = $default_more_toraldo_ftol`.

Keyword `gamma` can be used to specify a 2-tuple of values to safeguard the
quadratic interpolation of the step and such that `0 < gamma[1] < gamma[2] <
1`). In Moré & Toraldo (1991, GPCG algorithm) `gamma = (0.01,0.5)` while in
Birgin et al. (2000, SPG2 algorithm) `gamma = (0.1,0.9)`. The default settings
are `gamma = $default_more_toraldo_gamma`.

If keyword `strict` is set to `false`, a variant of the method is used which
takes a bissection step whenever the curvature of the model is not striclty
positive. By default, `strict` is `$default_more_toraldo_strict`.


## Description

The objective of Moré & Toraldo method is to find a step `stp` that satisfies
the sufficient decrease condition (1st Wolfe condition):

    f(stp) ≤ f(0) + ftol*stp*f'(0),

where `stp` is smaller or equal the initial step (backtracking). The principle
of the method is to reduce the step by a safeguarded quadratic interpolation
(or by a bissection if the minimum of the quadratic interpolation falls outside
the current search interval) until the criterion is met.

The new step is computed as follows:

    newstp = clamp(fact, gamma[1], gamma[2])*stp

where `fact` is the factor by which the current step should be multiplied to
reach the minimum of the quadratic interpolation of the objective function.


## References

* J.J. Moré & G. Toraldo, "*On the Solution of Large Quadratic Programming
  Problems with Bound Constraints*", SIAM J. Optim., vol. 1, pp. 93–113 (1991).

* E.G. Birgin, J.M. Martínez & M. Raydan, "*Nonmonotone Spectral Projected
  Gradient Methods on Convex Sets*", SIAM J. Optim., vol. 10, pp. 1196–1211
  (2000).

"""
mutable struct MoreToraldoLineSearch{T<:AbstractFloat} <: LineSearch{T}
    state::Symbol
    descr::Symbol
    gamma1::T
    gamma2::T
    ftol::T
    finit::T
    ginit::T
    stp::T
    stpmin::T
    strict::Bool
    function MoreToraldoLineSearch{T}(
        ; ftol::Real = default_more_toraldo_ftol,
        strict::Bool = default_more_toraldo_strict,
        gamma::NTuple{2,Real} = default_more_toraldo_gamma) where {T<:AbstractFloat}
        return configure!(new{T}(:STARTING, :NOT_STARTED,
                                 NaN, NaN, NaN, NaN, NaN, NaN, NaN, false);
                          ftol, gamma, strict)
    end
end

function configure!(ls::MoreToraldoLineSearch{T};
                    ftol::Real = ls.ftol,
                    strict::Bool = ls.strict,
                    gamma::NTuple{2,Real} = ls.gamma) where {T<:AbstractFloat}
    0 < ftol ≤ 1//2 || throw(ArgumentError("`0 < ftol ≤ 1/2` must hold"))
    0 < gamma[1] < gamma[2] < 1 || throw(ArgumentError(
        "`0 < gamma[1] < gamma[2] < 1` must hold"))
    ls.ftol   = ftol
    ls.gamma1 = gamma[1]
    ls.gamma2 = gamma[2]
    ls.strict = strict
    return ls
end
# Backtracking line-search does not use the directional derivative to refine
# the step but it does use the initial derivative.
use_derivatives(::Type{<:MoreToraldoLineSearch}) = false

function start!(ls::Union{MoreToraldoLineSearch{T},ArmijoLineSearch{T}},
                f0::T, g0::T, stp::T, stpmin::T, stpmax::T) where {T<:AbstractFloat}

    # Store objective function and directional derivative at start of line
    # search.
    ls.finit = f0
    ls.ginit = g0
    ls.stp = stp
    ls.stpmin = stpmin
    return report!(ls, :SEARCHING, :COMPUTE_F)
end

function iterate!(ls::MoreToraldoLineSearch{T}, f::T, g::T) where {T<:AbstractFloat}
    @assert ls.state == :SEARCHING
    # FIXME @assert stp == ls.stp
    if f ≤ ls.finit + ls.ftol*ls.stp*ls.ginit
        return report!(ls, :CONVERGENCE, :FIRST_WOLFE_CONDITION_SATISFIED)
    elseif ls.stp > ls.stpmin
        # FIXME type-stability
        q = -ls.ginit*ls.stp
        r = f - (ls.finit - q)
        if (ls.strict ? r > 0 && q > 0 : r != 0)
            # Safeguarded quadratic interpolation.
            stp = clamp(q/(r + r), ls.gamma1, ls.gamma2)*ls.stp
        else
            # Bissection.
            stp = ls.stp/2
        end
        ls.stp = max(stp, ls.stpmin)
        return report!(ls, :SEARCHING, :COMPUTE_F)
    else
        ls.stp = ls.stpmin
        return report!(ls, :WARNING, :STP_EQ_STPMIN)
    end
end

#-------------------------------------------------------------------------------

# A few constants for Moré Thuente line-search method.
const XTRAPL = 1.1
const XTRAPU = 4.0
const STPMAX = 1e20 # default upper step bound is STPMAX*stp1
const default_more_thuente_ftol = 0.001
const default_more_thuente_gtol = 0.9
const default_more_thuente_xtol = 0.1

"""
    MoreThuenteLineSearch{T=$Float}(; kwds...)

yields an object for Moré & Thuente line-search.

Line-search methods which need to compute a cubic step via `cstep!` need to
store some parameters in an instance of `LineSearchInterval` which has the
following members:

* `lower` and `upper` contain respectively a lower and an upper bound for the
  step;

* `stx`, `fx`, `dx` contain the values of the step, function, and derivative at
  the best step so far;

* `sty`, `fy`, `dy` contain the value of the step, function, and derivative at
  the other endpoint`sty`;

* `brackt` indicates whether a minimum is bracketed in the interval
  `(stx,sty)`.

At the start of the line-search, `brackt = false`, `stx = sty = 0` while `fx =
fy = f0` and `dx = dy = g0` the value of the function and its derivative for
the step `stp = 0`. Then `cstep!` can be called (typically in the `iterate!`
method) to compute a new step (based on cubic or quadratic interpolation) and
to maintain the interval of search.

"""
mutable struct MoreThuenteLineSearch{T<:AbstractFloat} <: LineSearch{T}
    state::Symbol
    descr::Symbol
    ftol::T
    gtol::T
    xtol::T
    stpmin::T
    stpmax::T
    finit::T
    ginit::T
    stp::T
    stx::T
    fx::T
    gx::T
    sty::T
    fy::T
    gy::T
    lower::T     # current lower bound for the step
    upper::T     # current upper bound for the step
    width::T     # current width of the interval
    oldwidth::T  # previous width of the interval
    stage::Int
    brackt::Bool # minimum has been bracketed?
    function MoreThuenteLineSearch{T}(;
                                      ftol::Real = default_more_thuente_ftol,
                                      gtol::Real = default_more_thuente_gtol,
                                      xtol::Real = default_more_thuente_xtol,
                                      ) where {T<:AbstractFloat}
        return configure!(
            new{T}(:STARTING, :NOT_STARTED,
                   NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN,
                   NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN,
                   0, false); xtol, ftol, gtol)
    end
end

function configure!(ls::MoreThuenteLineSearch{T};
                    ftol::Real = ls.ftol,
                    gtol::Real = ls.gtol,
                    xtol::Real = ls.xtol) where {T<:AbstractFloat}
    # FIXME are these checks sufficient?
    ftol > zero(ftol) || throw(ArgumentError("`ftol` must be positive"))
    gtol > zero(gtol) || throw(ArgumentError("`gtol` must be positive"))
    xtol ≥ zero(xtol) || throw(ArgumentError("`xtol` must be nonnegative"))
    ls.ftol = ftol
    ls.gtol = gtol
    ls.xtol = xtol
    return ls
end

# Provide default floating-point type for line-search constructors.
for func in (:ArmijoLineSearch, :MoreToraldoLineSearch, :MoreThuenteLineSearch,)
    @eval $func(args...; kwds...) = $func{Float}(args...; kwds...)
end

# Moré & Thuente line-search does use the directional derivative to refine the
# step.
use_derivatives(::Type{<:MoreThuenteLineSearch}) = true

default_stpmax(ls::MoreThuenteLineSearch{T}, stp1::Real) where {T} =
    as(T, STPMAX*stp1)

# Low-level `start!` method, arguments have been checked for errors.
function start!(ls::MoreThuenteLineSearch{T}, fx0::T, dgx0::T, stp1::T,
                stpmin::T, stpmax::T) where {T<:AbstractFloat}
    # Initialize local variables.
    # The variables STX, FX, GX contain the values of the step,
    # function, and derivative at the best step.
    # The variables STY, FY, GY contain the value of the step,
    # function, and derivative at STY.
    # The variables STP, F, G contain the values of the step,
    # function, and derivative at STP.
    ls.stpmin    = stpmin
    ls.stpmax    = stpmax
    ls.finit     = fx0
    ls.ginit     = dgx0
    ls.stp       = stp1
    ls.stx       = 0
    ls.fx        = fx0
    ls.gx        = dgx0
    ls.sty       = 0
    ls.fy        = fx0
    ls.gy        = dgx0
    ls.lower     = 0
    ls.upper     = (XTRAPU + one(XTRAPU))*stp1
    ls.width     = stpmax - stpmin
    ls.oldwidth  = 2*(stpmax - stpmin)
    ls.stage     = 1
    ls.brackt    = false
    report!(ls, :SEARCHING, :COMPUTE_FG)
end

function iterate!(ls::MoreThuenteLineSearch{T}, f::T, g::T) where {T<:AbstractFloat}

    @assert ls.state == :SEARCHING
    # FIXME @assert stp == ls.stp

    P66 = T(0.66)
    HALF = T(0.5)
    ZERO = zero(T)

    # If psi(stp) ≤ 0 and f'(stp) ≥ 0 for some step, then the algorithm enters
    # the second stage.
    gtest = ls.ftol*ls.ginit
    ftest = ls.finit + ls.stp*gtest
    if ls.stage == 1 && f ≤ ftest && g ≥ ZERO
        ls.stage = 2
    end

    # Test for termination: convergence or warnings.
    if f ≤ ftest && abs(g) ≤ -ls.gtol*ls.ginit
        return report!(ls, :CONVERGENCE, :STRONG_WOLFE_SATISFIED)
    end
    if ls.stp == ls.stpmin && (f > ftest || g ≥ gtest)
        return report!(ls, :WARNING, :STP_EQ_STPMIN)
    end
    if ls.stp == ls.stpmax && f ≤ ftest && g ≤ gtest
        return report!(ls, :WARNING, :STP_EQ_STPMAX)
    end
    if ls.brackt
        if ls.upper - ls.lower ≤ ls.xtol*ls.upper
            return report!(ls, :WARNING, :XTOL_TEST_SATISFIED)
        end
        if ls.stp ≤ ls.lower || ls.stp ≥ ls.upper
            return report!(ls, :WARNING, :ROUNDING_ERRORS)
        end
    end

    # A modified function is used to predict the step during the first stage if
    # a lower function value has been obtained but the decrease is not
    # sufficient.

    if ls.stage == 1 && f ≤ ls.fx && f > ftest

        # Call CSTEP to update STX, STY, and to compute the new step for the
        # modified function and its derivatives.
        info, ls.brackt,
        ls.stx, fxm, gxm,
        ls.sty, fym, gym,
        ls.stp = cstep(ls.brackt, ls.lower, ls.upper,
                    ls.stx, ls.fx - ls.stx*gtest, ls.gx - gtest,
                    ls.sty, ls.fy - ls.sty*gtest, ls.gy - gtest,
                    ls.stp, f - ls.stp*gtest, g - gtest)

        # Reset the function and derivative values for F.
        ls.fx = fxm + ls.stx*gtest
        ls.fy = fym + ls.sty*gtest
        ls.gx = gxm + gtest
        ls.gy = gym + gtest

    else

        # Call CSTEP to update STX, STY, and to compute the new step.
        info, ls.brackt,
        ls.stx, ls.fx, ls.gx,
        ls.sty, ls.fy, ls.gy,
        ls.stp = cstep(ls.brackt, ls.lower, ls.upper,
                    ls.stx, ls.fx, ls.gx,
                    ls.sty, ls.fy, ls.gy,
                    ls.stp, f, g)

    end

    # Decide if a bisection step is needed.
    if ls.brackt
        wcur = abs(ls.sty - ls.stx)
        if wcur ≥ P66*ls.oldwidth
            ls.stp = ls.stx + HALF*(ls.sty - ls.stx)
        end
        ls.oldwidth = ls.width
        ls.width = wcur
    end

    # Set the minimum and maximum steps allowed for STP.
    if ls.brackt
        if ls.stx ≤ ls.sty
            ls.lower = ls.stx
            ls.upper = ls.sty
        else
            ls.lower = ls.sty
            ls.upper = ls.stx
        end
    else
        ls.lower = ls.stp + XTRAPL*(ls.stp - ls.stx)
        ls.upper = ls.stp + XTRAPU*(ls.stp - ls.stx)
    end

    # Force the step to be within the bounds STPMAX and STPMIN.
    ls.stp = clamp(ls.stp, ls.stpmin, ls.stpmax)

    # If further progress is not possible, let STP be the best point obtained
    # during the search.
    if (ls.brackt && (ls.stp ≤ ls.lower || ls.stp ≥ ls.upper ||
                      ls.upper - ls.lower ≤ ls.xtol*ls.upper))
        ls.stp = ls.stx
    end

   # Obtain another function and derivative.
    return report!(ls, :SEARCHING, :COMPUTE_FG)
end

"""
# Compute a safeguarded cubic step.

    cstep(brackt, stpmin, stpmax,
          stx, fx, dx,
          sty, fy, dy,
          stp, fp, dp) ->  info, brackt, stx,fx,dx, sty,fy,dy, stp

The function `cstep` computes a safeguarded step for a search procedure and
updates an interval that contains a step that satisfies a sufficient decrease
and a curvature condition.  The algorithm is described in:

* J.J. Moré and D.J. Thuente, "*Line search algorithms with guaranteed
  sufficient decrease*" in ACM Transactions on Mathematical Software, vol. 20,
  pp. 286–307 (1994).

The parameter `stx` contains the step with the least function value.  If
`brackt` is set to true (*i.e.,* non-zero) then a minimizer has been bracketed
in an interval with endpoints `stx` and `sty`.  The parameter `stp` contains
the current step.  The subroutine assumes that if `brackt` is true then:

     min(stx,sty) < stp < max(stx,sty),

and that the derivative at `stx` is negative in the direction of the step.

On output, the updated parameters are returned.

Parameter `brackt` specifies if a minimizer has been bracketed.  Initially
`brackt` must be set to false.  The returned value of `brackt` indicates if a
minimizer has been bracketed.

Parameters `stpmin` and `stpmax` specify the lower and the upper bounds for the
step.

Parameters `stx`, `fx` and `dx` specify the step, the function and the
derivative at the best step obtained so far.  The derivative must be negative
in the direction of the step, that is, `dx` and `stp - stx` must have opposite
signs.  On return, these parameters are updated appropriately.

Parameters `sty`, `fy` and `dy` specify the step, the function and the
derivative at the other endpoint of the interval of uncertainty.  On return,
these parameters are updated appropriately.

Parameters `stp`, `fp` and `dp` specify the step, the function and the
derivative at the current step.  If `brackt` is true, then `stp` must be
between `stx` and `sty`.  On return, these parameters are updated
appropriately.  The returned value of `stp` is the next trial step.

The returned value `info` indicates which case occured for computing the new
step.

"""
function cstep(brackt::Bool, args::Real...)
    T = float(promote_type(map(typeof, args)...))
    return cstep(brackt, map(as(T), args)...)
end

function cstep(brackt::Bool, stpmin::T, stpmax::T,
               stx::T, fx::T, dx::T,
               sty::T, fy::T, dy::T,
               stp::T, fp::T, dp::T) where {T<:AbstractFloat}

    ZERO = zero(T)
    ONE = one(T)
    TWO = convert(T, 2)
    THREE = convert(T, 3)
    P66 = convert(T, 0.66)
    local theta::T, gamma::T, p::T, q::T, r::T, s::T, t::T

    info = 0

    # Check the input parameters for errors.
    if brackt && (stp ≤ min(stx, sty) || stp ≥ max(stx, sty))
        throw(ArgumentError("STP outside bracket (STX,STY)"))
    end
    if dx*(stp - stx) ≥ ZERO
        throw(ArgumentError("descent condition violated"))
    end
    if stpmax < stpmin
        throw(ArgumentError("STPMAX < STPMIN"))
    end

    # Determine if the derivatives have opposite sign.
    opposite = ((dx < ZERO && dp > ZERO) || (dx > ZERO && dp < ZERO))

    if fp > fx
        # First case.  A higher function value.  The minimum is bracketed.  If
        # the cubic step is closer to STX than the quadratic step, the cubic
        # step is taken, otherwise the average of the cubic and quadratic steps
        # is taken.
        info = 1
        theta = THREE*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)^2 - (dx/s)*(dp/s))
        if stp < stx
            gamma = -gamma
        end
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p/q
        stpc = stx + r*(stp - stx)
        stpq = stx + ((dx/((fx - fp)/(stp - stx) + dx))/TWO)*(stp - stx)
        if abs(stpc - stx) < abs(stpq - stx)
            stpf = stpc
        else
            stpf = stpc + (stpq - stpc)/TWO
        end
        brackt = true
    elseif opposite
        # Second case.  A lower function value and derivatives of opposite
        # sign.  The minimum is bracketed.  If the cubic step is farther from
        # STP than the secant (quadratic) step, the cubic step is taken,
        # otherwise the secant step is taken.
        info = 2
        theta = THREE*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)^2 - (dx/s)*(dp/s))
        if stp > stx
            gamma = -gamma
        end
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p/q
        stpc = stp + r*(stx - stp)
        stpq = stp + (dp/(dp - dx))*(stx - stp)
        if abs(stpc - stp) > abs(stpq - stp)
            stpf = stpc
        else
            stpf = stpq
        end
        brackt = true
    elseif abs(dp) < abs(dx)
        # Third case.  A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative decreases.  The cubic step is
        # computed only if the cubic tends to infinity in the direction of the
        # step or if the minimum of the cubic is beyond STP.  Otherwise the
        # cubic step is defined to be the secant step.
        info = 3
        theta = THREE*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        # The case GAMMA = 0 only arises if the cubic does not tend to infinity
        # in the direction of the step.
        t = (theta/s)^2 - (dx/s)*(dp/s)
        gamma = (t > ZERO ? s*sqrt(t) : ZERO)
        if stp > stx
            gamma = -gamma
        end
        p = (gamma - dp) + theta
        #q = ((gamma - dp) + gamma) + dx
        q = (gamma + (dx - dp)) + gamma
        r = p/q
        if r < ZERO && gamma != ZERO
            stpc = stp + r*(stx - stp)
        elseif stp > stx
            stpc = stpmax
        else
            stpc = stpmin
        end
        stpq = stp + (dp/(dp - dx))*(stx - stp)
        if brackt
            # A minimizer has been bracketed.  If the cubic step is closer to
            # STP than the secant step, the cubic step is taken, otherwise the
            # secant step is taken.
            stpf = abs(stpc - stp) < abs(stpq - stp) ? stpc : stpq
            t = stp + P66*(sty - stp)
            if stp > stx ? stpf > t : stpf < t
                stpf = t
            end
        else
            # A minimizer has not been bracketed. If the cubic step is farther
            # from stp than the secant step, the cubic step is taken, otherwise
            # the secant step is taken.
            stpf = abs(stpc - stp) > abs(stpq - stp) ? stpc : stpq
            stpf = max(stpmin, min(stpf, stpmax))
        end
    else
        # Fourth case.  A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative does not decrease.  If the
        # minimum is not bracketed, the step is either STPMIN or STPMAX,
        # otherwise the cubic step is taken.
        info = 4
        if brackt
            theta = THREE*(fp - fy)/(sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            t = theta/s
            gamma = s*sqrt((theta/s)^2 - (dy/s)*(dp/s))
            if stp > sty
                gamma = -gamma
            end
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p/q
            stpc = stp + r*(sty - stp)
            stpf = stpc
        elseif stp > stx
            stpf = stpmax
        else
            stpf = stpmin
        end
    end

    # Update the interval which contains a minimizer and guess for next step.
    if fp > fx
        sty = stp
        fy = fp
        dy = dp
    else
        if opposite
            sty = stx
            fy = fx
            dy = dx
        end
        stx = stp
        fx = fp
        dx = dp
    end
    stp = stpf
    return (info, brackt, stx,fx,dx, sty,fy,dy, stp)
end

end # module
