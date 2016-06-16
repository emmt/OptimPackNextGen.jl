#
# weights.jl --
#
# Implement methods to compute statistical weights for TiPi.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2015-2016, Éric Thiébaut, Jonathan Léger & Matthew Ozon.
# This file is part of TiPi.  All rights reserved.
#

doc"""
# Compute statistical weights for counting data

    compute_weights!(wgt, dat, alpha, beta; bad=...) -> wgt

store in `wgt` the statistical weights for the data `dat` assuming the
following simple model for the variance of the data:

    Var(dat) = alpha*max(data,0) + beta                      (1)

where `alpha ≥ 0` and `beta > 0` are the parameters of the noise model.
The computed weights are:

    wgt[i] = 1/Var(dat[i])    if dat[i] is finite and not a NaN
           = 0                else

and thus account for valid data which must have a finite value and not be a
NaN.  The rationale is that saturations may be marked with an infinite value
while bad data are marked by a NaN.  Note that the weights are guaranteed to be
nonnegative and that, with `alpha = 0`, uniform variance is assumed.

An error is thrown if it is found that there are no valid data.

If keyword `bad` is specified with a real value, the invalid data will be
marked with this value (and the corresponding weights will be set to zero).
Although it may result in cahnging the data, this keyword is useful to avoid
taking care of special data values for further processing (e.g., use `bad = 0`
or some other finite value).  Without this keyword, the data `dat` are left
unchanged.

The following methods allocate the weights:

    wgt = compute_weights(dat, alpha, beta)
    wgt = compute_weights!(dat, alpha, beta; bad=...)

The first one does not change the input data, while the second does with `bad =
NaN` by default.  Finally:

    wgt = default_weights(dat)

is the same as `compute_weights(dat, 0, 1)`, i.e., it yields weights which are
equal to one for valid data and to zero otherwise.


## Explanations

For a signal based on counts (for instance, photo-electrons), the variance of
the data should be given by:

    Var(dat) = (E(adu*dat) + sigma^2)/adu^2

with `adu` the conversion factor of the detector (for instance in e- per
digital level) such that `adu*dat` is the measured data in count units,
`E(adu*dat)` is the expected number of counts (which is also the variance of
the counts assuming Poisson statistics) and `sigma` is the standard deviation
(rms value) of the detector noise in counts (per pixel per frame).  Expanding
this expression yields:

    Var(dat) = alpha*E(dat) + beta

with `alpha = 1/adu` and `beta = (sigma/adu)^2`.  Finally, the following
approximation:

    E(dat) ≈ max(dat, 0)

leads to Eq. (1).


## References

* L. M. Mugnier, T. Fusco & J.-M. Conan, "MISTRAL: a myopic edge-preserving
  image restoration method, with application to astronomical
  adaptive-optics-corrected long-exposure images", J. Opt. Soc. Am. A, vol. 21,
  pp.1841-1854 (2004).

* A. Foi, M. Trimeche, V. Katkovnik & K. Egiazarian, "Practical
  Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data",
  IEEE Transactions on Image Processing, vol. 17, pp. 1737-1754 (2008).

""" function compute_weights! end

function compute_weights!{T<:AbstractFloat,N}(wgt::Array{T,N}, dat::Array{T,N},
                                              alpha::Real, beta::Real;
                                              bad::Union{Real,Void}=nothing)
    alpha::T = alpha
    beta::T = beta
    @assert size(wgt) == size(dat)
    @assert alpha ≥ 0
    @assert beta > 0
    cnt = 0
    const wmax = one(T)/beta
    if is(bad, nothing)
        # Bad data will be left unchanged.
        if alpha > 0
            @inbounds begin
                for i in 1:length(dat)
                    if isinf(dat[i]) || isnan(dat[i])
                        wgt[i] = zero(T)
                    elseif wgt[i] > zero(T)
                        wgt[i] = (dat[i] > zero(T) ?
                                  one(T)/(alpha*dat[i] + beta) :
                                  wmax)
                        cnt += 1
                    end
                end
            end
        else
            @inbounds begin
                for i in 1:length(dat)
                    if isinf(dat[i]) || isnan(dat[i])
                        wgt[i] = zero(T)
                    elseif wgt[i] > zero(T)
                        wgt[i] = wmax
                        cnt += 1
                    end
                end
            end
        end
    else
        # Bad data will be marked with the specified value.
        bad::T = bad
        if alpha > 0
            @inbounds begin
                for i in 1:length(dat)
                    if isinf(dat[i]) || isnan(dat[i])
                        dat[i] = bad
                        wgt[i] = zero(T)
                    elseif wgt[i] > zero(T)
                        wgt[i] = (dat[i] > zero(T) ?
                                  one(T)/(alpha*dat[i] + beta) :
                                  wmax)
                        cnt += 1
                    end
                end
            end
        else
            @inbounds begin
                for i in 1:length(dat)
                    if isinf(dat[i]) || isnan(dat[i])
                        dat[i] = bad
                        wgt[i] = zero(T)
                    elseif wgt[i] > zero(T)
                        wgt[i] = wmax
                        cnt += 1
                    end
                end
            end
        end
    end
    if cnt == 0
        error("noo valid data!")
    end
    return wgt
end

function compute_weights!{T<:AbstractFloat,N}(dat::Array{T,N},
                                              alpha::Real, beta::Real;
                                              bad::Real=NaN)
    compute_weights!(Array(T, size(dat)), dat, alpha, beta; bad=bad)
end

function compute_weights{T<:AbstractFloat,N}(dat::Array{T,N},
                                             alpha::Real, beta::Real)
    compute_weights!(Array(T, size(dat)), dat, alpha, beta)
end

function default_weights{T<:AbstractFloat,N}(dat::Array{T,N})
    compute_weights(dat, 0, 1)
end

@doc @doc(compute_weights!) compute_weights
@doc @doc(compute_weights!) default_weights

#------------------------------------------------------------------------------
