module OptimPackNextGenZygoteExt

if isdefined(Base, :get_extension)
    using Zygote
    using OptimPackNextGen
else
    using ..Zygote
    using ..OptimPackNextGen
end

function OptimPackNextGen.auto_differentiate!(f, x, g)
    OptimPackNextGen.NumOptBase.copy!(g, Zygote.gradient(f, x)[1]);
    return f(x)
end

end # module
