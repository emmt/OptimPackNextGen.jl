module BenchmarkingBrentFmin

using Printf
using BenchmarkTools

using OptimPackNextGen.Brent
using .Brent: fmin_atol, fmin_rtol

include("fmin-funcs.jl")
using .FminTestFunctions

function run(;
             T::Type{<:AbstractFloat}=Float64,
             prec::Bool=false, # tests with high precision
             )
    if prec
        atol = 1e-100
        rtol = 1e-15
    else
        atol = fmin_atol(T)
        rtol = fmin_rtol(T)
    end
    for _f in (brent_2, brent_3, brent_4, brent_5,
               michalewicz_1, michalewicz_2,
               ampgo_2, ampgo_3, ampgo_4, ampgo_5, ampgo_6, ampgo_7,
               ampgo_8, ampgo_9, ampgo_10, ampgo_11, ampgo_12, ampgo_13,
               ampgo_14, ampgo_15, ampgo_18, ampgo_20, ampgo_21, ampgo_22,
               gsl_fmin_1, gsl_fmin_2, gsl_fmin_3, gsl_fmin_4)
        f = TestFunc{T}(_f)
        a = local_lower(f)
        b = local_upper(f)
        x0 = xsol(f)
        f0 = fsol(f)
        reset_nevals()
        (xm, fm, lo, hi) = fmin(f, a, b; rtol=rtol, atol=atol)
        nevals = get_nevals()
        println("Problem $(ident(f)) (T=$T):")
        @printf(" ├─ nevals = %d\n", nevals)
        if prec
            @printf(" ├─ x0 = %.13f, f0 = %.13f\n", xm, fm)
            @printf(" └─ prec = %.3e\n", (hi - lo)/2)
        else
            @printf(" ├─")
            @btime fmin($f, $a, $b);
            @printf(" ├─ xm     = %.15f ± %.3e\n", xm, (hi - lo)/2)
            @printf(" ├─ x0     = %.15f\n", x0)
            @printf(" ├─ x_err  = %.3e\n", abs(xm - x0))
            @printf(" ├─ f(xm)  = %.15f\n", fm)
            @printf(" ├─ f(x0)  = %.15f\n", f0)
            @printf(" └─ f_err  = %.3e\n", abs(fm - f0))
        end
    end
end

end # module
