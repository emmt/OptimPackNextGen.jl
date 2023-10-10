"""
    OmptimPackNextGen.auto_differentiate!(f, x, g) -> fx

yields `fx = f(x)` for a given function `f` and variables `x` and overwrites
the contents of `g` with the gradient of the function at `x`.

This method may be extended to compute the gradient and function value for
specific `typeof(f)` or to automatically compute the gradient as can be done by
the `Zygote` package if it is loaded.

"""
auto_differentiate!(arg...; kwds...) =
    error("`Zygote` package must be loaded first")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include(
            "../ext/OptimPackNextGenZygoteExt.jl")
    end
end
