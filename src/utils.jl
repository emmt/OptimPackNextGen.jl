"""

`Float` is the type of all floating point scalars, it is currently an alias to
`Cdouble` which is itself an alias to `Float64`.

"""
const Float = Cdouble

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
    configure!(ctx; kwds...) -> ctx

change settings in the context `ctx` according to keywords `kwds...` and
returns the modified context.

"""
function configure! end

"""
    minimize(ctx, f, x0; kwds...) -> x, stats

attempt to minimize objective function `f` with initial variables `x0`
and given the context `ctx`.

"""
function minimize end

"""
    minimize!(ctx, f, x0; kwds...) -> x, stats

attempt to minimize objective function `f` with initial variables `x0` and
given the context `ctx`. This is the *in-place* version of `minimize`: input
variables `x0` are overwriten by the solution and `x === x0` holds on output.

"""
function minimize! end

"""
    OptimPackNextGen.scalar_type(obj) -> T

yields the type of scalar computations with structured object `obj`.

"""
function scalar_type end

"""
    OptimPackNextGen.variables_type(obj) -> T

yields the type of variables for computations with structured object `obj`..

"""
function variables_type end

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
    OptimPackNextGen.get_reason(alg) -> str

yields a textual description of the current state of the iterative algorithm
implemented by `alg`.

"""
function get_reason end

"""
     OptimPackNextGen.is_positive(x) -> bool

yields whether `x` is strictly positive.

"""
is_positive(x) = x > zero(x)
# FIXME unused: is_negative(x) = x < zero(x)
