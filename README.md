# TiPi.jl

**TiPi** is a **T**oolkit for **I**nverse **P**roblems and **I**maging in
Julia.  One of the objective of TiPi is to solve image reconstruction
problems, so TiPi is designed to dela with large number of unknowns.


## Cost functions

Many inverse problems are solved by minimizing a cost function.  **TiPi**
defines a common interface for the cost functions that can be used in that
context.  The are currently two kind of cost functions that can be used in
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
where `alpha` is the weight of the cost (it is guaranteed that `alpha` is
nonnegative), `param` is anything needed by the cost function, `ParamType`
is a subtype of the abstract type `CostParam`, `x` is the array of
variables, `VariableType` is the type of the variables (e.g.,
`Array{Cdouble,2}` for images), , `gx` is an array of same type and
dimensions as `x` to store (or integrate) the gradient of the cost function
(times `alpha`), `clr` tells whether to clear (fill with zeros) the values
of the gradient or to simply add the computed gradient to the existing
values, the retuend value is `alpha*f(x)` that is the value of the cost
function at `x` times the weight `alpha`.

Thanks to the dispatching rules of Julia, the types `ParamType` and
`VariableType` of the cost function parameters and of the variables of the
problem are used to identify the actual cost function code that is called.

The weight and the clear flag arguments are intended for building composite
cost functions as a weighted sum of cost functions.

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
function cost{T<:Real}(alpha::Real, param::MAPCostParam, x::Array{T})
    alpha == 0 && return 0.0
	cost(alpha, param.lkl, x) + cost(alpha*param.mu, param.rgl, x)
end
```
and computing the cost and the gradient is implemented by:
```julia
function cost!{T<:Real}(alpha::Real, param::MAPCostParam, x::Array{T},
                        gx::Array{T}, clr::Bool=false)
    if alpha == 0
	   clr && gx[:] = 0
	   return 0.0
	else
	   return (cost(alpha,          param.lkl, x, gx, clr) +
	           cost(alpha*param.mu, param.rgl, x, gx, false))
	end
end
```
Note the specific way the *clear* flag `clr` is managed to preserve the
good properties of the interface and allow our implementation of the MAP
cost function to be mixed itself with other functions.


### Proximal operators

