module OptimPackNextGenCUTEstExt

if isdefined(Base, :get_extension)
    using NLPModels
    using CUTEst
    using OptimPackNextGen
    import OptimPackNextGen.SPG: spg, spg_CUTEst
    import OptimPackNextGen.QuasiNewton: vmlmb, vmlmb_CUTEst
else
    using ..NLPModels
    using ..CUTEst
    using ..OptimPackNextGen
    import ..OptimPackNextGen.SPG: spg, spg_CUTEst
    import ..OptimPackNextGen.QuasiNewton: vmlmb, vmlmb_CUTEst
end

function vmlmb_CUTEst(name::AbstractString; kwds...)
    nlp = CUTEstModel(name)
    try
        return vmlmb(nlp; kwds...)
    finally
        finalize(nlp)
    end
end

"""
    vmlmb(nlp::CUTEstModel; kwds...) -> x

yields the solution to the `CUTEst` problem `nlp` by the VMLMB method with
settings specified by keywords `kwds..`.

"""
function vmlmb(nlp::CUTEstModel;
               mem::Integer  = 5,
               blmvm::Bool = false,
               verb::Integer = false,
               output::IO    = stdout,
               kwds...)
    bound_constrained(nlp) || unconstrained(nlp) || error(
        "VMLMB method can only solve bound constrained or unconstrained problems ($name)")
    x0 = get_x0(nlp)
    objfun = ObjectiveFunction(nlp)
    if verb > zero(verb)
        println(output, "#\n# Solving CUTEst ",
                bound_constrained(nlp) ? "bound constrained" : "unconstrained",
                " problem $(get_name(nlp)) by ", blmvm ? "BLMVM" : "VMLMB", " method\n",
                "# with n = $(get_nvar(nlp)) variable(s) and m = $mem memorized step(s).\n#")
    end
    if bound_constrained(nlp)
        lower = get_lvar(nlp)
        upper = get_uvar(nlp)
        return vmlmb(objfun, x0; mem, blmvm, verb, output, lower, upper, kwds...)
    else
        return vmlmb(objfun, x0; mem, blmvm, verb, output, kwds...)
    end
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
                bound_constrained(nlp) ? "bound constrained" : "unconstrained",
                " problem $(get_name(nlp)) by SPG method\n",
                "# with n = $(get_nvar(nlp)) variable(s) and m = $m memorized step(s).\n#")
    end
    return spg(objfun, proj, x0, m; verb, io, kwds...)
end

end # module
