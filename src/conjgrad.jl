#
# conjgrad.jl --
#
# Linear conjugate gradients for OptimPack.
#
# -----------------------------------------------------------------------------
#
# This file is part of OptimPackNextGen.jl which is licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2015-2017, Éric Thiébaut.
#

doc"""
# Linear Conjugate Gradient Algorithm

This method iteratively solves the linear system:

    A⋅x = b

where `A` is the so-called left-hand-side operator (which must be positive
definite) of the normal equations, `b` is the right-hand-side vector of the
normal equations and `x` are the unknowns.

To use this method call:

    x = conjgrad(A, b)

where the left-hand-side operator is provided as a function (or any callable
object) which takes two arguments, a destination `dst` and a source `src` and
which stores in `dst` the result of `A` applied to `src` and which returns
`dst`:

    A(dst, src) -> dst

By default, the method starts with all variables set to zero, another initial
value, say `x0`, for the variables may be provided:

    x = conjgrad(A, b, x0)

The initial variables may be used to store the final result:

    conjgrad!(A, b, x) -> x

will overwrite the contents of the initial variables `x` with the final result
and will return it.  If no initial variables are specified, the default is to
start with all variables set to zero.

There are several keywords to control the algorithm:

* `ftol` specifies the function tolerance for convergence.  The convergence is
  assumed as soon as the variation of the objective function between two
  successive iterations is less or equal `ftol` times the largest variation so
  far.  By default, `ftol = 1e-7`.

* `gtol` specifies the gradient tolerances for convergence, it is a tuple of
  two values `(gatol, grtol)` where `gatol` and `grtol` are the absolute and
  relative tolerances.  Convergence occurs when the Euclidean norm of the
  residuals (which is that of the gradient of the associated objective
  function) is less or equal the largest of `gatol` and `grtol` times the
  Eucliden norm of the initial residuals.  By default, `gtol = (0.0, 0.0)`.

* `maxiter` specifies the maximum number of iterations.

"""
conjgrad{T}(A, b::T; kwds...) =
    conjgrad!(A, b, vfill!(vcreate(b), 0); kwds...)

conjgrad{T}(A, b::T, x0::T; kwds...) =
    conjgrad!(A, b, vcopy(x0); kwds...)

function conjgrad!{T}(A, b::T, x::T;
                      ftol::Real = 1e-7,
                      gtol::NTuple{2,Real} = (0.0,0.0),
                      maxiter::Integer = typemax(Int),
                      verb::Bool = false,
                      io::IO = STDOUT)
    # Initialization.
    @assert ftol ≥ 0
    @assert gtol[1] ≥ 0
    @assert gtol[2] ≥ 0
    if vdot(x,x) > 0 # cheap trick to check whether x is non-zero
        r = vcreate(x)
        A(r, x)
        vcombine!(r, 1, b, -1, r)
    else
        r = vcopy(b)
    end
    local rho::Float = vdot(r, r)
    const ftest = Float(ftol)
    const gtest = Float(max(gtol[1], gtol[2]*sqrt(rho)))
    local alpha::Float = 0
    local gamma::Float = 0
    local psi::Float = 0
    local psimax::Float = 0
    local oldrho::Float = 0

    # Conjugate gradient iterations.
    k = 0
    n = length(b)
    while true
        if verb
            if k == 0
                @printf(io, "# %s\n# %s\n",
                        "Iter.  Delta f(x)    ||∇f(x)||",
                        "-------------------------------")
            end
            @printf(io, "%6d %12.4e %12.4e\n",
                    k, Cdouble(psi), Cdouble(sqrt(rho)))
        end
        k += 1
        if sqrt(rho) ≤ gtest
            # Normal convergence.
            break
        elseif k > maxiter
            warn("too many ($(maxiter)) conjugate gradient iterations")
            break
        end
        if rem(k, n) == 1
            # First iteration or restart.
            if k == 1
                p = vcopy(r)
                q = vcreate(x)
            else
                vcopy!(p, r)
            end
        else
            vcombine!(p, 1, r, (rho/oldrho), p)
        end
        A(q, p)
        gamma = vdot(p, q)
        if gamma ≤ 0
            error("left-hand-side operator A is not positive definite")
        end
        alpha = rho/gamma
        vupdate!(x, +alpha, p)
        psi = rho*alpha
        psimax = max(psi, psimax)
        if psi ≤ ftest*psimax
            # Normal convergence.
            break
        end
        vupdate!(r, -alpha, q)
        oldrho = rho
        rho = vdot(r, r)
    end
    return x
end

@doc @doc(conjgrad) conjgrad!
