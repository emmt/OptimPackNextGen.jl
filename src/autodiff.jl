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
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        function auto_differentiate!(f, x, g)
            vcopy!(g, Zygote.gradient(f, x)[1]);
            return f(x)
        end
    end
end
