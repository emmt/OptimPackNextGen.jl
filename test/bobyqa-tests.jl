module BobyqaTests

using Printf
using OptimPackNextGen.Powell

function runtests()
    # The test function.
    function f(x::DenseVector{Cdouble})
        fx = 0.0
        n = length(x)
        for i in 4:2:n
            for j in 2:2:i-2
                tempa = x[i - 1] - x[j - 1]
                tempb = x[i] - x[j]
                temp = tempa*tempa + tempb*tempb
                temp = max(temp,1e-6)
                fx += 1.0/sqrt(temp)
            end
        end
        return fx
    end

    # Run the tests.
    bdl = -1.0
    bdu =  1.0
    rhobeg = 0.1
    rhoend = 1e-6
    for m in (5,10)
        q = 2.0*pi/m
        n = 2*m
        x = Array{Cdouble}(undef, n)
        xl = Array{Cdouble}(undef, n)
        xu = Array{Cdouble}(undef, n)
        for i in 1:n
            xl[i] = bdl
            xu[i] = bdu
        end
        for jcase in 1:2
            if jcase == 2
                npt = 2*n + 1
            else
                npt = n + 6
            end
            @printf("\n\n     2D output with M =%4ld,  N =%4ld  and  NPT =%4ld\n",
                    m, n, npt)
            for j in 1:m
                temp = q*j
                x[2*j - 1] = cos(temp)
                x[2*j]     = sin(temp)
            end
            fx = bobyqa!(f, x, xl, xu, rhobeg, rhoend, npt=npt,
                         verbose=2, maxeval=500000)[3]
            @printf("\n***** least function value: %.15e\n", fx)
        end
    end
end

end # module
