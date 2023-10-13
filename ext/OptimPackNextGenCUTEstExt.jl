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
    nlp = CUTEstModel(name)
    try
        return spg(nlp, m; kwds...)
    finally
        finalize(nlp)
    end
end

"""
    spg(nlp::CUTEstModel, m::Integer; kwds...) -> x

yields the solution to the `CUTEst` problem `nlp` by the SPG method with a
memory of `m` previous steps.

"""
function spg(nlp::CUTEstModel, m::Integer;
             verb::Integer = false,
             io::IO        = stdout,
             kwds...)
    bound_constrained(nlp) || unconstrained(nlp) || error(
        "SPG method can only solve bound constrained or unconstrained problems ($name)")
    x0 = get_x0(nlp)
    objfun = ObjectiveFunction(nlp)
    proj = Projector(nlp)
    if verb > zero(verb)
        println(io, "#\n# Solving CUTEst ",
                (bound_constrained(nlp) ? "bound constrained" : "unconstrained"),
                " problem $(get_name(nlp)) by SPG method\n",
                "# with n = $(get_nvar(nlp)) variable(s) and m = $m memorized step(s).\n#")
    end
    return spg(objfun, proj, x0, m; verb, kwds...)
end

end # module
