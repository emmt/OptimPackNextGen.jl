module OptimPackNextGenCUTEstExt

if isdefined(Base, :get_extension)
    using NLPModels
    using CUTEst
    using OptimPackNextGen
    import OptimPackNextGen.SPG: spg, spg_CUTEst
else
    using ..NLPModels
    using ..CUTEst
    using ..OptimPackNextGen
    import ..OptimPackNextGen.SPG: spg, spg_CUTEst
end

function spg_CUTEst(name::AbstractString, m::Integer; kwds...)
    model = CUTEstModel(name)
    try
        return spg(model, m; kwds...)
    finally
        finalize(model)
    end
end

function spg(model::CUTEstModel, m::Integer; kwds...)
    bound_constrained(model) || unconstrained(model) || error(
        "SPG method can only solve bound constrained or unconstrained problems ($name)")
    x0 = get_x0(model)
    objfun = ObjectiveFunction(model)
    proj = Projector(model)
    return spg(objfun, proj, x0, m; kwds...)
end

end # module
