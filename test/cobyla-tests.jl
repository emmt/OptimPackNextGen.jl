module CobylaTests

using Printf
using OptimPackNextGen.Powell

function runtests(;revcom::Bool = false, scale::Real = 1.0)
    # Beware that order of operations may affect the result (whithin
    # rounding errors).  I have tried to keep the same ordering as F2C
    # which takes care of that, in particular when converting expressions
    # involving powers.
    prt(s) = println("\n       "*s)
    for nprob in 1:10
        if nprob == 1
            # Minimization of a simple quadratic function of two variables.
            prt("Output from test problem 1 (Simple quadratic)")
            n = 2
            m = 0
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = -1.0
            xopt[2] = 0.0
            ftest = (x::DenseVector{Cdouble}) -> begin
                r1 = x[1] + 1.0
                r2 = x[2]
                fc = 10.0*(r1*r1) + (r2*r2)
                return fc
            end
        elseif nprob == 2
            # Easy two dimensional minimization in unit circle.
            prt("Output from test problem 2 (2D unit circle calculation)")
            n = 2
            m = 1
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = sqrt(0.5)
            xopt[2] = -xopt[1]
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = x[1]*x[2]
                con[1] = 1.0 - x[1]*x[1] - x[2]*x[2]
                return fc
            end
        elseif nprob == 3
            # Easy three dimensional minimization in ellipsoid.
            prt("Output from test problem 3 (3D ellipsoid calculation)")
            n = 3
            m = 1
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = 1.0/sqrt(3.0)
            xopt[2] = 1.0/sqrt(6.0)
            xopt[3] = -0.33333333333333331
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = x[1]*x[2]*x[3]
                con[1] = 1.0 - (x[1]*x[1]) - 2.0*(x[2]*x[2]) - 3.0*(x[3]*x[3])
                return fc
            end
        elseif nprob == 4
            # Weak version of Rosenbrock's problem.
            prt("Output from test problem 4 (Weak Rosenbrock)")
            n = 2
            m = 0
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = -1.0
            xopt[2] = 1.0
            ftest = (x::DenseVector{Cdouble}) -> begin
                r2 = x[1]
                r1 = r2*r2 - x[2]
                r3 = x[1] + 1.0
                fc = r1*r1 + r3*r3
                return fc
            end
        elseif nprob == 5
            # Intermediate version of Rosenbrock's problem.
            prt("Output from test problem 5 (Intermediate Rosenbrock)")
            n = 2
            m = 0
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = -1.0
            xopt[2] = 1.0
            ftest = (x::DenseVector{Cdouble}) -> begin
                r2 = x[1]
                r1 = r2*r2 - x[2]
                r3 = x[1] + 1.0
                fc = r1*r1*10.0 + r3*r3
                return fc
            end
        elseif nprob == 6
            # This problem is taken from Fletcher's book Practical Methods
            # of Optimization and has the equation number (9.1.15).
            prt("Output from test problem 6 (Equation (9.1.15) in Fletcher)")
            n = 2
            m = 2
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = sqrt(0.5)
            xopt[2] = xopt[1]
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = -x[1] - x[2]
                r1 = x[1]
                con[1] = x[2] - r1*r1
                r1 = x[1]
                r2 = x[2]
                con[2] = 1.0 - r1*r1 - r2*r2
                return fc
            end
        elseif nprob == 7
            # This problem is taken from Fletcher's book Practical Methods
            # of Optimization and has the equation number (14.4.2).
            prt("Output from test problem 7 (Equation (14.4.2) in Fletcher)")
            n = 3
            m = 3
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = 0.0
            xopt[2] = -3.0
            xopt[3] = -3.0
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = x[3]
                con[1] = x[1]*5.0 - x[2] + x[3]
                r1 = x[1]
                r2 = x[2]
                con[2] = x[3] - r1*r1 - r2*r2 - x[2]*4.0
                con[3] = x[3] - x[1]*5.0 - x[2]
                return fc
            end
        elseif nprob == 8
            # This problem is taken from page 66 of Hock and Schittkowski's
            # book Test Examples for Nonlinear Programming Codes. It is
            # their test problem Number 43, and has the name Rosen-Suzuki.
            prt("Output from test problem 8 (Rosen-Suzuki)")
            n = 4
            m = 3
            xopt = Array{Cdouble}(undef, n)
            xopt[1] = 0.0
            xopt[2] = 1.0
            xopt[3] = 2.0
            xopt[4] = -1.0
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                fc = (r1*r1 + r2*r2 + r3*r3*2.0 + r4*r4 - x[1]*5.0
                      - x[2]*5.0 - x[3]*21.0 + x[4]*7.0)
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                con[1] = (8.0 - r1*r1 - r2*r2 - r3*r3 - r4*r4 - x[1]
                          + x[2] - x[3] + x[4])
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                r4 = x[4]
                con[2] = (10.0 - r1*r1 - r2*r2*2.0 - r3*r3 - r4*r4*2.0
                          + x[1] + x[4])
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                con[3] = (5.0 - r1*r1*2.0 - r2*r2 - r3*r3 - x[1]*2.0
                          + x[2] + x[4])
                return fc
            end
        elseif nprob == 9
            # This problem is taken from page 111 of Hock and
            # Schittkowski's book Test Examples for Nonlinear Programming
            # Codes. It is their test problem Number 100.
            prt("Output from test problem 9 (Hock and Schittkowski 100)")
            n = 7
            m = 4
            xopt = Array{Cdouble}(undef, n)
            xopt[1] =  2.330499
            xopt[2] =  1.951372
            xopt[3] = -0.4775414
            xopt[4] =  4.365726
            xopt[5] = -0.624487
            xopt[6] =  1.038131
            xopt[7] =  1.594227
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                r1 = x[1] - 10.0
                r2 = x[2] - 12.0
                r3 = x[3]
                r3 *= r3
                r4 = x[4] - 11.0
                r5 = x[5]
                r5 *= r5
                r6 = x[6]
                r7 = x[7]
                r7 *= r7
                fc = (r1*r1 + r2*r2*5.0 + r3*r3 + r4*r4*3.0
                      + r5*(r5*r5)*10.0 + r6*r6*7.0 + r7*r7
                      - x[6]*4.0*x[7] - x[6]*10.0 - x[7]*8.0)
                r1 = x[1]
                r2 = x[2]
                r2 *= r2
                r3 = x[4]
                con[1] = (127.0 - r1*r1*2.0 - r2*r2*3.0 - x[3]
                          - r3*r3*4.0 - x[5]*5.0)
                r1 = x[3]
                con[2] = (282.0 - x[1]*7.0 - x[2]*3.0 - r1*r1*10.0
                          - x[4] + x[5])
                r1 = x[2]
                r2 = x[6]
                con[3] = (196.0 - x[1]*23.0 - r1*r1 - r2*r2*6.0
                          + x[7]*8.0)
                r1 = x[1]
                r2 = x[2]
                r3 = x[3]
                con[4] = (r1*r1*-4.0 - r2*r2 + x[1]*3.0*x[2]
                          - r3*r3*2.0 - x[6]*5.0 + x[7]*11.0)
                return fc
            end
        elseif nprob == 10
            # This problem is taken from page 415 of Luenberger's book
            # Applied Nonlinear Programming. It is to maximize the area of
            # a hexagon of unit diameter.
            prt("Output from test problem 10 (Hexagon area)")
            n = 9
            m = 14
            xopt = fill!(Array{Cdouble}(undef, n), 0.0)
            ftest = (x::DenseVector{Cdouble}, con::DenseVector{Cdouble}) -> begin
                fc = -0.5*(x[1]*x[4] - x[2]*x[3] + x[3]*x[9] - x[5]*x[9]
                           + x[5]*x[8] - x[6]*x[7])
                r1 = x[3]
                r2 = x[4]
                con[1] = 1.0 - r1*r1 - r2*r2
                r1 = x[9]
                con[2] = 1.0 - r1*r1
                r1 = x[5]
                r2 = x[6]
                con[3] = 1.0 - r1*r1 - r2*r2
                r1 = x[1]
                r2 = x[2] - x[9]
                con[4] = 1.0 - r1*r1 - r2*r2
                r1 = x[1] - x[5]
                r2 = x[2] - x[6]
                con[5] = 1.0 - r1*r1 - r2*r2
                r1 = x[1] - x[7]
                r2 = x[2] - x[8]
                con[6] = 1.0 - r1*r1 - r2*r2
                r1 = x[3] - x[5]
                r2 = x[4] - x[6]
                con[7] = 1.0 - r1*r1 - r2*r2
                r1 = x[3] - x[7]
                r2 = x[4] - x[8]
                con[8] = 1.0 - r1*r1 - r2*r2
                r1 = x[7]
                r2 = x[8] - x[9]
                con[9] = 1.0 - r1*r1 - r2*r2
                con[10] = x[1]*x[4] - x[2]*x[3]
                con[11] = x[3]*x[9]
                con[12] = -x[5]*x[9]
                con[13] = x[5]*x[8] - x[6]*x[7]
                con[14] = x[9]
                return fc
            end
        else
            error("bad problem number ($nprob)")
        end

        x = Array{Cdouble}(undef, n)
        for icase in 1:2
            fill!(x, 1.0)
            rhobeg = 0.5
            rhoend = (icase == 2 ? 1e-4 : 0.001)
            if revcom
                # Test the reverse communication variant.
                c = Array{Cdouble}(undef, max(m, 0))
                ctx = Cobyla.Context(n, m, rhobeg, rhoend;
                                     verbose = 1, maxeval = 2000)
                status = getstatus(ctx)
                while status == Cobyla.ITERATE
                    if m > 0
                        # Some constraints.
                        fx = ftest(x, c)
                        status = iterate(ctx, fx, x, c)
                    else
                        # No constraints.
                        fx = ftest(x)
                        status = iterate(ctx, fx, x)
                    end
                end
                if status != Cobyla.SUCCESS
                    println("Something wrong occured in COBYLA: ",
                            getreason(status))
                end
            elseif scale == 1
                cobyla!(ftest, x, m, rhobeg, rhoend;
                        verbose = 1, maxeval = 2000)
            else
                Cobyla.minimize!(ftest, x, m, rhobeg/scale, rhoend/scale;
                                 scale = fill!(Array{Cdouble}(undef, n), scale),
                                 verbose = 1, maxeval = 2000)
            end
            if nprob == 10
                tempa = x[1] + x[3] + x[5] + x[7]
                tempb = x[2] + x[4] + x[6] + x[8]
                tempc = 0.5/sqrt(tempa*tempa + tempb*tempb)
                tempd = tempc*sqrt(3.0)
                xopt[1] = tempd*tempa + tempc*tempb
                xopt[2] = tempd*tempb - tempc*tempa
                xopt[3] = tempd*tempa - tempc*tempb
                xopt[4] = tempd*tempb + tempc*tempa
                for i in 1:4
                    xopt[i + 4] = xopt[i]
                end
            end
            temp = 0.0
            for i in 1:n
                r1 = x[i] - xopt[i]
                temp += r1*r1
            end
            @printf("\n     Least squares error in variables =%16.6E\n", sqrt(temp))
        end
        @printf("  ------------------------------------------------------------------\n")
    end
end

end # module
