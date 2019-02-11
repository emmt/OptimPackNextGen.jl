#
# gqtpar.jl --
#
# Julia implementation of GQTPAR algorithm for "Computing A Trust Region Step"
# by Moré, J.J. & Sorensen, D.C. (SIAM J. Sci. Stat. Comp., 1983, vol. 4,
# pp. 553-572).
#
#-----------------------------------------------------------------------------
#
# Original FORTRAN code dgqt.f and destsv.f from MINPACK-2 project.
#
# Copyright (C) 1993, Brett M. Averick and Jorge J. Moré (MINPACK-2 Project,
# Argonne National Laboratory and University of Minnesota).
#
# Copyright (C) 1994, Brett M. Averick, Richard Carter and Jorge J. Moré
# (MINPACK-2 Project, Argonne National Laboratory and University of
# Minnesota).
#
# Julia version of GQTPAR is part of OptimPackNextGen package licensed under
# the MIT "Expat" License.
#
# Copyright (C) 2008-2018, Éric Thiébaut
# <https://github.com/emmt/OptimPackNextGen.jl>.
#

module MoreSorensen

export gqtpar, gqtpar!

using Compat
#import Compat.LinearAlgebra.BLAS
import Compat.LinearAlgebra.BLAS: trsv!, nrm2
import Compat.LinearAlgebra.BLAS: BlasInt, BlasReal, BlasFloat, BlasComplex
import Compat.LinearAlgebra.BLAS: libblas, @blasfunc

# We need to apply BLAS TRSV to a submatrix.  BEWARE that no bound check is
# performed.
for (fname, elty) in ((:dtrsv_,:Float64),
                      (:strsv_,:Float32),
                      (:ztrsv_,:ComplexF64),
                      (:ctrsv_,:ComplexF32))
    @eval begin
                #       SUBROUTINE DTRSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)
                #       .. Scalar Arguments ..
                #       INTEGER INCX,LDA,N
                #       CHARACTER DIAG,TRANS,UPLO
                #       .. Array Arguments ..
                #       DOUBLE PRECISION A(LDA,*),X(*)
        function _trsv!(uplo::Char, trans::Char, diag::Char, n::Integer,
                        A::StridedMatrix{$elty}, lda::Integer,
                        x::StridedVector{$elty}, incx::Integer = stride(x,1))
            ccall((@blasfunc($fname), libblas), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                 uplo, trans, diag, n, A, lda, x, incx)
        end
    end
end

function _trsv!(uplo::Char, trans::Char, diag::Char, n::Integer,
                A::StridedMatrix{T}, x::StridedVector{T},
                incx::Integer = stride(x,1)) where {T<:BlasFloat}
    _trsv!(uplo, trans, diag, n, A, max(1,stride(A,2)), x, incx)
end

"""
# Computing a Trust Region Step

**GQTPAR** is Moré & Sorensen method for computing a trust region step.

The possible calling sequences are:

```julia
gqtpar(A, uplo, b, delta, rtol, atol, itmax, par) -> info, par, x, fx, iter
```

or:

```julia
gqtpar!(A, uplo, b, delta,
        rtol, atol, itmax, par, x [, z, d, w]) -> info, par, x, fx, iter
```

Given an `n×n` symmetric matrix `A`, an `n`-vector `b`, and a positive number
`delta`, this subroutine determines a vector `x` which approximately minimizes
the quadratic function:

    f(x) = (1/2)*x'*A*x + b'*x

subject to the Euclidean norm constraint

    norm(x) ≤ delta.

This subroutine computes an approximation `x` and a Lagrange multiplier `par`
such that either `par` is zero and

    norm(x) ≤ (1+rtol)*delta,

or `par` is positive and

    abs(norm(x) - delta) ≤ rtol*delta.

If `xsol` is the solution to the problem, the approximation `x` satisfies

    f(x) ≤ ((1 - rtol)**2)*f(xsol)

The arguments are:

* `A` is a 2D floating-point array of size `(n,n)`.  On entry the full upper or
   lower triangle of `A` must contain the full upper triangle of the symmetric
   matrix `A`.  On exit the array contains the matrix `A`.

* `uplo` specifies which part of the symmetric matrix `A` is stored in array
  `A` on entry: 'U' for the full upper triangle of `A`, 'L' for the full
  lower triangle of `A`.

* `b` is a 1D floating-point array of length `n` which specifies the linear
  term in the quadratic fucntion.  The contents of `b` is left unchanged.

* `delta` is a bound on the Euclidean norm of `x`.

* `rtol` is the relative accuracy desired in the solution. Convergence occurs
  if `f(x) ≤ ((1 - rtol)^2)*f(xsol)`.

* `atol` is the absolute accuracy desired in the solution. Convergence occurs
  when `norm(x) ≤ (1 + rtol)*delta` or `max(-f(x),-f(xsol)) ≤ atol`.

* `itmax` specifies the maximum number of iterations.

* `par` is an initial estimate of the Lagrange multiplier for the constraint
  `norm(x) ≤ delta`.

For `gqtpar!`:

* `x` is a 1D floating-point array of length `n` to store the approximate
  solution.

* `z`, `d` and `w` are optional work arrays of lenght `n`.

The returned values are:

* `info` is an integer code:

  - `info = 1` if the function value `f(x)` has the relative accuracy specified
    by `rtol`.

  - `info = 2` if the function value `f(x)` has the absolute accuracy specified
    by `atol`.

  - `info = 3` if rounding errors prevent further progress.  On exit `x` is the
    best available approximation.

  - `info = 4` indicates a failure to converge after `itmax` iterations.  On
    exit `x` is the best available approximation.

* `par` is the final estimate of the Lagrange multiplier.

* `x` is the final estimate of the solution.

* `fx` is `f(x)` at the output `x`.

* `iter` is the number of necessary iterations.


## References

* J. J. Moré & D. C. Sorensen, "*Computing a Trust Region Step*", SIAM
  J. Sci. Stat. Comp., vol. **4**, pp. 553-572 (1983).

* MINPACK-2 Project. July 1994.
  Argonne National Laboratory and University of Minnesota.
  Brett M. Averick, Richard Carter, and Jorge J. Moré.

"""
function gqtpar(A::StridedMatrix{T},
                uplo::Char,
                b::StridedVector{T},
                delta::Real, rtol::Real, atol::Real,
                itmax::Integer, par::Real) where {T<:BlasReal}
    return gqtpar!(A, uplo, b, delta, rtol, atol, itmax, par,
                   Array{T}(undef, lenght(b)))
end

function gqtpar!(A::StridedMatrix{T},
                 uplo::Char,
                 b::StridedVector{T},
                 delta::Real, rtol::Real, atol::Real,
                 itmax::Integer, par::Real,
                 x::StridedVector{T},
                 z::StridedVector{T} = Array{T}(undef, length(b)),
                 d::StridedVector{T} = Array{T}(undef, length(b)),
                 w::StridedVector{T} = Array{T}(undef, length(b))) where {T<:BlasReal}
    return gqtpar!(A, uplo, b, T(delta), T(rtol), T(atol),
                   Int(itmax), T(par), x, z, d, w)
end

@doc @doc(gqtpar) gqtpar!

function gqtpar!(A::StridedMatrix{T},
                 uplo::Char,
                 b::StridedVector{T},
                 delta::T,
                 rtol::T,
                 atol::T,
                 itmax::Int,
                 par::T,
                 x::StridedVector{T},
                 z::StridedVector{T},
                 d::StridedVector{T},
                 w::StridedVector{T}) where {T<:BlasReal}

    @inbounds begin

        n = length(b)
        @assert size(A) == (n, n)
        @assert length(x) == n
        @assert length(z) == n
        @assert length(d) == n
        @assert length(w) == n
        lda = max(1, stride(A, 2))

        # Local constants and variables.
        ZERO  = convert(T, 0.0)
        ONE   = convert(T, 1.0)
        HALF  = convert(T, 0.5)
        SMALL = convert(T, 0.001)
        local alpha::T
        local anorm::T
        local bnorm::T
        local prod::T
        local rxnorm::T
        local rznorm::T
        local xnorm::T
        local parc::T
        local parf::T
        local parl::T
        local pars::T
        local paru::T
        local temp::T
        local temp1::T
        local temp2::T
        local temp3::T

        # Initialization.
        if ! isfinite(par) || par < ZERO
            par = ZERO
        end
        parf = ZERO
        xnorm = ZERO
        rxnorm = ZERO
        alpha = ZERO
        rznorm = ZERO  # avoids compiler warnings
        vzero!(x)
        vzero!(z)
        if uplo == 'U'
            # Copy upper triangular part of A in its lower triangular part and
            # save its diagonal.
            for j in 1:n
                d[j] = A[j,j]
                @simd for i in j+1:n
                    A[i,j] = A[j,i]
                end
            end
        elseif uplo == 'L'
            # Copy lower triangular part of A in its upper triangular part and
            # save its diagonal.
            for j in 1:n
                d[j] = A[j,j]
                @simd for i in j+1:n
                    A[j,i] = A[i,j]
                end
            end
        else
            error("invalid value for uplo='$uplo' (expecting 'U' or 'L')")
        end

        # Calculate the l1-norm of A, the Gershgorin row sums,
        # and the l2-norm of b.
        anorm = ZERO
        for j in 1:n
            temp = ZERO
            @simd for i in 1:n
                temp += abs(A[i,j])
            end
            anorm = max(anorm, temp)
            w[j] = temp - abs(d[j])
        end
        bnorm = nrm2(b)

        # Calculate a lower bound, PARS, for the domain of the problem.
        # Also calculate an upper bound, PARU, and a lower bound, PARL,
        # for the Lagrange multiplier.
        pars = parl = paru = -anorm
        for j in 1:n
            pars = max(pars, -d[j])
            parl = max(parl, w[j] + d[j])
            paru = max(paru, w[j] - d[j])
        end
        temp = bnorm/delta
        parl = max(ZERO, temp - parl, pars)
        paru = max(ZERO, temp + paru)

        # If the input PAR lies outside of the interval (PARL,PARU),
        # set PAR to the closer endpoint.
        par = min(max(par, parl), paru)

        # Special case: parl = paru.
        paru = max(paru, (ONE + rtol)*parl)

        # Beginning of an iteration.
        info = 0
        iter = 0
        while true

            # Safeguard PAR.
            if par ≤ pars && paru > ZERO
                par = max(sqrt(parl/paru), SMALL)*paru
            end

            # Copy the lower triangle of A into its upper triangle and
            # compute A + par*I.  FIXME: not the first time
            for j in 1:n
                A[j,j] = d[j] + par
                @simd for i in j+1:n
                    A[j,i] = A[i,j]
                end
            end

            # Attempt the Cholesky factorization of A without referencing
            # the lower triangular part.
            indef = Int(potrf!('U', A)[2])
            rednc = false

            if indef == 0

                ###############################################
                ##  Case 1: A + par⋅I is positive definite.  ##
                ###############################################

                # Compute an approximate solution x and save the
                # last value of par with A + par*I positive definite.
                parf = par
                copyto!(w, b)
                trsv!('U', 'T', 'N', A, w)
                rxnorm = nrm2(w)
                trsv!('U', 'N', 'N', A, w)
                @simd for j in 1:n
                    x[j] = -w[j]
                end
                xnorm = nrm2(x)

                # Test for convergence.
                if (abs(xnorm - delta) ≤ rtol*delta
                    || (par == ZERO && xnorm ≤ (ONE + rtol)*delta))
                    # RTOL test satisfied.
                    info = 1
                end

                # Compute a direction of negative curvature and use this
                # information to improve pars.
                rznorm = estsv!(A, z)
                pars = max(pars, par - rznorm*rznorm)

                # Compute a negative curvature solution of the form
                # x + alpha*z where norm(x + alpha*z) = delta.
                if xnorm < delta

                    # Compute ALPHA.
                    prod = DOT(z, x)/delta
                    temp = ((delta + xnorm)/delta)*(delta - xnorm)
                    alpha = temp/(abs(prod) + sqrt(prod*prod + temp/delta))
                    alpha = copysign(alpha, prod)

                    # Test to decide if the negative curvature step
                    # produces a larger reduction than with z = 0.
                    rznorm *= abs(alpha)
                    temp1 = (rznorm/delta)^2
                    rednc = (temp1 + par*(xnorm/delta)^2 ≤ par)

                    # Test for convergence.
                    temp2 = par + (rxnorm/delta)^2
                    if HALF*temp1 ≤ rtol*(ONE - HALF*rtol)*temp2
                        info = 1
                    elseif info == 0 && HALF*temp2 ≤ (atol/delta)/delta
                        info = 2
                    elseif xnorm == ZERO
                        info = 1
                    end
                end

                # Compute the Newton correction PARC to PAR.
                if xnorm == ZERO
                    parc = -par
                else
                    temp = ONE/xnorm
                    vscale!(w, temp, x)
                    trsv!('U', 'T', 'N', A, w)
                    temp = nrm2(w)
                    parc = (((xnorm - delta)/delta)/temp)/temp
                end

                # Update PARL or PARU.
                if xnorm > delta && parl < par
                    parl = par
                end
                if xnorm < delta && paru > par
                    paru = par
                end

            else

                ###################################################
                ##  Case 2: A + par⋅I is not positive definite.  ##
                ###################################################

                # Use the rank information from the Cholesky
                # decomposition to update PAR.
                if indef > 1
                    # Restore column indef to A + par*I.
                    @simd for j in 1:indef-1
                        A[j,indef] = A[indef,j]
                    end
                    A[indef,indef] = d[indef] + par

                    # Compute PARC.
                    @simd for j in 1:indef-1
                        w[j] = A[j,indef]
                    end
                    _trsv!('U', 'T', 'N', indef-1, A, w)
                    A[indef,indef] -= nrm2(indef-1, w, stride(w,1))^2
                    _trsv!('U', 'N', 'N', indef-1, A, w)
                end
                w[indef] = -ONE
                temp = nrm2(indef, w, stride(w,1))
                parc = -(A[indef,indef]/temp)/temp
                pars = max(pars, par, par + parc)

                # If necessary, increase PARU slightly.  This is needed
                # because in some exceptional situations PARU is the optimal
                # value of PAR.
                paru = max(paru, (ONE + rtol)*pars)
            end

            # Use PARS to update PARL.
            parl = max(parl, pars)

            # Test for termination.
            iter += 1
            if info == 0
                if paru == ZERO
                    info = 2
                elseif paru ≤ (ONE + HALF*rtol)*pars
                    info = 3
                elseif iter ≥ itmax
                    info = 4
                end
            end

            # If exiting, store the best approximation and restore
            # the upper triangle of A.
            if info != 0

                # Compute the best current estimates for x and f.
                par = parf
                if rednc
                    fx = -HALF*((rxnorm^2 + par*delta^2) - rznorm^2)
                    if alpha != ZERO
                        @simd for j in 1:n
                            x[j] += alpha*z[j]
                        end
                    end
                else
                    fx = -HALF*(rxnorm^2 + par*xnorm^2)
                end

                # Restore the upper triangle of A.
                for j in 1:n
                    A[j,j] = d[j]
                    @simd for i in j+1:n
                        A[j,i] = A[i,j]
                    end
                end

                return info, par, x, fx, iter
            end

            # Compute an improved estimate for PAR.
            par = max(par + parc, parl)

        end # End of an iteration.
    end
end


"""

```julia
estsv!(R, z) -> svmin
```

Given an `n×n` upper triangular matrix `R`, this subroutine estimates the
smallest singular value and the associated singular vector of `R`.

In the algorithm a vector `e` is selected so that the solution `y` to the
system `R'*y = e` is large. The choice of sign for the components of `e` cause
maximal local growth in the components of `y` as the forward substitution
proceeds.  The vector `z` is the solution of the system `R*z = y`, and the
estimate `svmin` is `norm(y)/norm(z)` in the Euclidean norm.

On exit `svmin` is returned and `z` contains a singular vector associated with
the estimate `svmin` such that `norm(R*z) = svmin` and `norm(z) = 1` in the
Euclidean norm.


## References

* MINPACK-2 Project. October 1993.
  Argonne National Laboratory.
  Brett M. Averick and Jorge J. Moré.

"""
function estsv!(R::StridedMatrix{T},
                z::StridedVector{T}) where {T<:BlasReal}

    # Local constants and variables.
    n = length(z)
    @assert size(R) = (n,n)
    ZERO  = convert(T, 0.0)
    ONE   = convert(T, 1.0)
    SMALL = convert(T, 0.01)
    local e::T, s::T, sm::T, temp::T, w::T, wm::T, ynorm::T, znorm::T

    @inbounds begin

        # Clear vector z.
        @simd for i in 1:n
            z[i] = ZERO
        end

        # This choice of e makes the algorithm scale invariant.
        e = abs(R[1,1])
        if e == ZERO
            z[1] = ONE
            return ZERO
        end

        # Solve R'*y = e.
        for i in 1:n

            # Scale y. The factor of 0.01 reduces the number of scalings.
            e = copysign(e, -z[i])
            if abs(e - z[i]) > abs(R[i,i])
                temp = min(abs(R[i,i]/(e - z[i])), SMALL)
                vscale!(z, temp)
                e *= temp
            end

            # Determine the two possible choices of y[i].
            if R[i,i] == ZERO
                w = ONE
                wm = ONE
            else
                w  =  (e - z[i])/R[i,i]
                wm = -(e + z[i])/R[i,i]
            end

            # Choose y[i] based on the predicted value of y(j) for j > i.
            s  = abs(e - z[i])
            sm = abs(e + z[i])
            for j in i+1:n
                sm += abs(z[j] + wm*R[i,j])
            end
            @simd for j in i+1:n
                temp = z[j] + w*R[i,j]
                z[j] = temp
                s += abs(temp)
            end
            if s < sm
                temp = wm - w
                w = wm
                if temp != ZERO
                    @simd for j in i+1:n
                        z[j] += temp*R[i,j]
                    end
                end
            end
            z[i] = w
        end
        ynorm = nrm2(z)

        # Solve R**z = y.
        for j in n:-1:1
            # Scale z.
            if abs(z[j]) > abs(R[j,j])
                temp = min(abs(R[j,j]/z[j]), SMALL)
                vscale!(z, temp)
                ynorm *= temp
            end
            if R[j,j] == ZERO
                z[j] = ONE
            else
                z[j] /= R[j,j]
            end
            if (temp = z[j]) != ZERO
                @simd for i in 1:j-1
                    z[i] -= temp*R[i,j]
                end
            end
        end

        # Normalize z and return svmin.
        znorm = ONE/nrm2(z)
        vscale!(z, znorm)
        return ynorm*znorm
    end
end

end # module
