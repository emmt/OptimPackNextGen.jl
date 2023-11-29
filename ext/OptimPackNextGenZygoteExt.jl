module OptimPackNextGenZygoteExt

if isdefined(Base, :get_extension)
    using Zygote
    using OptimPackNextGen: OptimPackNextGen, vcopy!
else
    using ..Zygote
    using ..OptimPackNextGen: OptimPackNextGen, vcopy!
end

function OptimPackNextGen.auto_differentiate!(f, x, g)
    m = Zygote.withgradient(f,x)
    vcopy!(g,m[:grad][1] );
    return m[:val]
end

end # module
