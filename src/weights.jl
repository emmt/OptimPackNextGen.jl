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

const ALLOW_NEGATIVE_WEIGHTS   = (1 << 0)
const ALLOW_NONFINITE_WEIGHTS  = (1 << 1)
const FORBID_NONFINITE_DATA    = (1 << 2)
const FORBID_NO_VALID_DATA     = (1 << 3)

doc"""

    fix_weighted_data(wgt, dat) -> wgt, dat

check the validity of the data `dat` with their corresponding weights `wgt` and
may apply some corrections.  Input weights may be `nothing` or simply omitted
as in:

    fix_weighted_data(dat) -> wgt, dat

which is the same as assuming that all input weights are equal to one.

The returned value is a tuple of weights and data after corrections have been
applied if necessary.  Input parameters `wgt` and `dat` are left unchanged so
there is no side effects.  The returned values are either the input ones (if no
changes were necessary) or new arrays otherwise.  If the input weights are
unspecifed, the ouput weights are `nothing` if no bad data were found or a new
array of the same type and size as `dat` and with ones where data are valid and
zero elsewhere.

The purpose of this routine is to provide weighted data suitable for fast
numerical processing.  To that end, the following rules are checked according
to the value of keyword `policy` (which is zero by default):

* `! isfinite(wgt[i])` yields an error unless the bit
  `TiPi.ALLOW_NONFINITE_WEIGHTS` is set in `policy`;

* `wgt[i] < 0` yields an error unless the bit `TiPi.ALLOW_NEGATIVE_WEIGHTS` is
  set in `policy`;

* `! isfinite(dat[i])` while `wgt[i] > 0` yields an error if the bit
  `TiPi.FORBID_NONFINITE_DATA` is set in `policy`;

If no such errors occur, weights and data may be fixed so that all weights are
finite and nonnegative with a zero value indicating invalid data and the value
of nonfinite data are replaced by the value specified by the keyword `bad`,
defaulting to zero.  The keyword `bad` may take any finite value.

Finally, an error is raised if there are no valid data and bit
`TiPi.FORBID_NO_VALID_DATA` of `policy` is set.

The variant

    fix_weighted_data!(wgt, dat) -> cnt

performs the same operations but modifies the input arguments if corrections
are needed.  The number of valid data is returnd.

"""
function fix_weighted_data end

function fix_weighted_data{T<:Real,N}(wgt::AbstractArray{T,N},
                                      dat::AbstractArray{T,N};
                                      bad::Real=zero(T),
                                      policy::Integer=0)
    # Basic check.
    if ! isfinite(bad)
        throw(ArgumentError("\"bad\" value must be finite"))
    end
    if size(wgt) != size(dat)
        throw(ArgumentError("weights and data must have the same size"))
    end

    # First pass for errors.
    policy = Int(policy)
    fixdat = false
    fixwgt = false
    cnt = 0
    @inbounds for i in eachindex(wgt, dat)
        valid = true
        if ! isfinite(wgt[i])
            if (policy & ALLOW_NONFINITE_WEIGHTS) == 0
                throw(ArgumentError("non-finite weights are forbidden"))
            end
            fixwgt = true
            valid = false
        elseif wgt[i] ≤ zero(T)
            if wgt[i] < zero(T)
                if (policy & ALLOW_NEGATIVE_WEIGHTS) == 0
                    throw(ArgumentError("weights must be nonnegative"))
                end
                fixwgt = true
            end
            valid = false
        end
        if ! isfinite(dat[i])
            if valid
                if (policy & FORBID_NONFINITE_DATA) != 0
                    throw(ArgumentError("non-finite data are forbidden"))
                end
                fixwgt = true
            end
            fixdat = true
        elseif valid
            cnt += 1
        end
    end
    if cnt == 0 && (policy & FORBID_NO_VALID_DATA) != 0
        throw(ArgumentError("no-valid data"))
    end

    # Second pass to fix the data and their weights.
    retdat = (fixdat ? Array(T, size(dat)) : dat)
    retwgt = (fixwgt ? Array(T, size(wgt)) : wgt)
    if fixdat
        bad = T(bad)
        if fixwgt
            @inbounds for i in eachindex(dat, wgt, retdat, retwgt)
                if isfinite(wgt[i]) && wgt[i] ≥ zero(T) && isfinite(dat[i])
                    retdat[i] = dat[i]
                    retwgt[i] = wgt[i]
                else
                    retdat[i] = bad
                    retwgt[i] = zero(T)
                end
            end
        else
            @inbounds for i in eachindex(dat, wgt, retdat)
                retdat[i] = wgt[i] > zero(T) && isfinite(dat[i]) ? dat[i] : bad
            end
        end
    elseif fixwgt
        @inbounds for i in eachindex(dat, wgt, retwgt)
            retwgt[i] = isfinite(wgt[i]) && wgt[i] ≥ zero(T) ? wgt[i] : zero(T)
        end
    end

    return (retwgt, retdat)
end

fix_weighted_data{T<:Real,N}(::Void, dat::AbstractArray{T,N}; kws...) =
    fix_weighted_data(dat; kws...)

# No weights given.
function fix_weighted_data{T<:Real,N}(dat::AbstractArray{T,N};
                                      bad::Real=zero(T),
                                      policy::Integer=0)
    # Basic check.
    if isnan(bad) || isinf(bad)
        throw(ArgumentError("\"bad\" value must be finite"))
    end

    # First pass to decide whether to fix data or not.
    fixdat = false
    @inbounds for i in eachindex(dat)
        if ! isfinite(dat[i])
            if (policy & FORBID_NONFINITE_DATA) != 0
                throw(ArgumentError("non-finite data are forbidden"))
            end
            fixdat = true
            break
        end
    end

    # Second pass to fix data and set weights.
    if fixdat
        bad = T(bad)
        cnt = 0
        retdat = Array(T, size(dat))
        retwgt = Array(T, size(dat))
        @inbounds for i in eachindex(dat, retdat, retwgt)
            if isfinite(dat[i])
                retdat[i] = dat[i]
                retwgt[i] = one(T)
                cnt += 1
            else
                retdat[i] = bad
                retwgt[i] = zero(T)
            end
        end
        if cnt == 0 && (policy & FORBID_NO_VALID_DATA) != 0
            throw(ArgumentError("no-valid data"))
        end
    else
        retdat = dat
        retwgt = nothing
    end

    return (retwgt, retdat)
end

function fix_weighted_data!{T<:Real,N}(wgt::AbstractArray{T,N},
                                       dat::AbstractArray{T,N};
                                       bad::Real=zero(T),
                                       policy::Integer=0)
    if isnan(bad) || isinf(bad)
        throw(ArgumentError("bad value must be finite"))
    end
    if size(wgt) != size(dat)
        throw(ArgumentError("weights and data must have the same size"))
    end

    bad = T(bad)
    policy = Int(policy)
    cnt = 0
    @inbounds for i in eachindex(wgt, dat)
        if isinf(wgt[i]) || isnan(wgt[i])
            if (policy & ALLOW_NONFINITE_WEIGHTS) == 0
                throw(ArgumentError("non-finite weights are forbidden"))
            end
            dat[i] = bad
            wgt[i] = zero(T)
        elseif wgt[i] <= zero(T)
            if wgt[i] < zero(T)
                if (policy & ALLOW_NEGATIVE_WEIGHTS) == 0
                    throw(ArgumentError("weights must be nonnegative"))
                end
                wgt[i] = zero(T)
            end
            dat[i] = bad
        elseif isinf(dat[i]) || isnan(dat[i])
            if (policy & FORBID_NONFINITE_DATA) != 0
                throw(ArgumentError("non-finite data are forbidden"))
            end
            dat[i] = bad
            wgt[i] = zero(T)
        else
            cnt += 1
        end
    end
    if cnt == 0 && (policy & FORBID_NO_VALID_DATA) != 0
        throw(ArgumentError("no-valid data"))
    end
    return cnt
end

#------------------------------------------------------------------------------
