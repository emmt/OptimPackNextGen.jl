module OptimPackNextGenZygoteExt

using Zygote

if isdefined(Base, :get_extension)
    using OptimPackNextGen: OptimPackNextGen, vcopy!
else
    using ..OptimPackNextGen: OptimPackNextGen, vcopy!
end

function OptimPackNextGen.auto_differentiate!(f, x, g)
    vcopy!(g, Zygote.gradient(f, x)[1]);
    return f(x)
end

end # module
