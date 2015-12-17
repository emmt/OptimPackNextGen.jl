# TiPi.jl

**TiPi** is a **T**oolkit for **I**nverse **P**roblems and **I**maging in
Julia.  One of the main objectives of TiPi is to solve image reconstruction
problems, so TiPi is designed to deal with large number of unknowns
(*e.g.*, as many as pixels in an image).


## Cost functions

Many inverse problems are solved by minimizing a cost function.  **TiPi**
defines a common interface for the cost functions that can be used in that
context.  There are currently two kinds of cost functions that can be used in
TiPi:

* **differentiable cost functions** for which the value of the cost
  function and its gradient can be computed;

* **nondifferentiable cost functions** for which the value of the cost
  function may be computed but not that of the gradient, these functions
  are dealt with by means of their proximal operator.


### Differentiable cost functions

Differentiable cost functions must implement the following methods:
```julia
fx = cost(alpha::Real, param::ParamType, x::VariableType)
fx = cost!(alpha::Real, param::ParamType, x::VariableType,
           gx::VariableType, clr::Bool=false)
```
where `alpha` is the multiplier of the cost (it is guaranteed that `alpha` is
nonnegative), `param` is anything needed by the cost function, `ParamType` is
a subtype of the abstract type `CostParam`, `x` is the array of variables,
`VariableType` is the type of the variables (e.g., `Array{Cdouble,2}` for
images), `gx` is an array of same type and dimensions as `x` to store (or
integrate) the gradient of the cost function (times `alpha`), `clr` tells
whether to clear (fill with zeros) the values of the gradient or to simply add
the computed gradient to the existing values, the returned value is
`alpha*f(x)` that is the value of the cost function at `x` times the
multiplier `alpha`.

Thanks to the dispatching rules of Julia, the types `ParamType` and
`VariableType` of the cost function parameters and of the variables of the
problem are used to identify the actual cost function code that is called.

The multiplier and the clear flag arguments are intended for building
composite cost functions as a weighted sum of cost functions without
sacrifying efficiency.

For instance, we implement the parameters of a *maximum a posteriori* (MAP)
cost function as:
```julia
type MAPCostParam <: CostParam
    mu::Cdouble     # regularization weight
    lkl::CostParam  # parameters of the likelihood term
    rgl::CostParam  # parameters of the regularization term
end
```
Then, computing the cost is as simple as:
```julia
function cost{T}(alpha::Real, param::MAPCostParam, x::T)
    alpha == 0 && return 0.0
    cost(alpha, param.lkl, x) + cost(alpha*param.mu, param.rgl, x)
end
```
and computing the cost and the gradient is implemented by:
```julia
function cost!{T}(alpha::Real, param::MAPCostParam, x::T, gx::T, clr::Bool=false)
    if alpha == 0
        clr && fill!(gx, 0)
        return 0.0
    else
        return (cost(alpha,          param.lkl, x, gx, clr) +
                cost(alpha*param.mu, param.rgl, x, gx, false))
    end
end
```
Note the specific way the *clear* flag `clr` is managed to preserve the good
properties of the interface and allow our implementation of the MAP cost
function to be mixed itself with other functions.  As the `fill!` method is
used here to set all components of the gradient to zero, this method must
exists for the type `T` of the variables.


### Proximal operators

The proximal operator of the cost function `f(x)` (times the multiplier `α`)
is defined by:
```
    prox(α, f, x) = argmin_y { α f(y) + (1/2) ||x - y||² }
```

where `||...||` denotes the usual Eclidean norm.

The proximal operator of `f(x)` is implemented by the following method:
```julia
px = prox(alpha::Real, param::ParamType, x::VariableType)
```
where `alpha` is the multiplier (guaranteed to be nonnegative), `param` is an
intance of `ParamType` and `x` are the variables.  The argument `param`
carries all parameters needed by the specific cost fonction and also serves as
a signature to identity the cost function (as explained above).


### Rationale

There is no such thing as:
```julia
    abstract DifferentiableCostParam <: CostParam
```
The rationale is that rather than implementing a sub-type of `CostParam` for
differentiable cost functions, it is sufficient to have such specific cost
functions implement the `cost!` method.  Cost functions which are not
differentiable shall not implement the `cost!` method.  The same applies for
other specificities, for instance a cost function may have an associated
proximal operator or not.  Since Julia does not allow for multiple heritage,
we distinguish various flavor of cost functions by the methods they implement
(that's more Julia style).

