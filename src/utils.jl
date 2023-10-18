"""
    objfun = ObjectiveFunction(args...)

yields a callable object built from arguments `args...` and suitable for
optimizers in `OptimPackNextGen` and it can be used as:

    fx = objfun(x)
    fx = objfun(x, gx)

to compute the value `fx` of the objective function and, possibly, overwrite
`gx` with gradient of the objective function.

"""
struct ObjectiveFunction{T}
    parent::T
end
Base.parent(objfun::ObjectiveFunction) = getfield(objfun, :parent)

"""
    OptimPackNextGen.auto_differentiate!(f, x, g) -> fx

yields `fx = f(x)` for a given function `f` and variables `x` and overwrites
the contents of `g` with the gradient of the function at `x`.

This method may be extended to compute the gradient and function value for
specific `typeof(f)` or to automatically compute the gradient as can be done by
the `Zygote` package if it is loaded.

"""
auto_differentiate!(arg...; kwds...) =
    error("invalid arguments or `Zygote` package not yet loaded")

"""
     OptimPackNextGen.copy_variables(x)

yields a copy of the variables `x`. The result is an array *similar* to `x` but
with guaranteed floating-point element type.

"""
copy_variables(x::AbstractArray) = copyto!(similar(x, float(eltype(x))), x)

"""
    OptimPackNextGen.get_tolerances(T, name, atol, rtol) -> atol, rtol

yields absolute and relative tolerances `atol` and `rtol` for a parameter named
`name` converted to floating-point type `T` throwing an `ArgumentError`
exception if these settings have invalid values.

"""
function get_tolerances(::Type{T}, name::AbstractString,
                        atol::Real, rtol::Real) where {T<:AbstractFloat}
    atol ≥ zero(atol) ||
        throw(ArgumentError("absolute tolerance for $name must be nonnegative"))
    zero(rtol) ≤ rtol ≤ one(rtol) ||
        throw(ArgumentError("relative tolerance for $name must be in [0,1]"))
    return as(T, atol), as(T, rtol)
end

"""
    OptimPackNextGen.get_reason(alg) -> str

yields a textual description of the current state of the iterative algorithm
implemented by `alg`.

"""
function get_reason end
