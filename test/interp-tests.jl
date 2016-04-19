module TiPiTests
const PLOTTING = true
if PLOTTING
    using PyCall; pygui(:gtk); # can be: :wx, :gtk, :qt
    using LaTeXStrings
    import PyPlot; const plt = PyPlot
end
include("../src/TiPi.jl")
#importall TiPi

function interp_tests()
    z = [0.5, 0.3, 0.1, 0.0, -0.2, -0.7, -0.7, 0.0, 1.7]
    #z = [0, 0, 0, 1, 0, 0, 0]

    # 1-D example
    t = linspace(-3,14,2000);
    op1 = TiPi.Interpolator(TiPi.Kernels.triangle, 1:length(z), t)
    op2 = TiPi.Interpolator(TiPi.Kernels.catmull_rom, 1:length(z), t)
    if PLOTTING
        plt.figure(2)
        plt.clf()
        plt.plot(t, op1(z), color="darkgreen",
                 linewidth=1.0, linestyle="-");
        plt.plot(t, op2(z+0.1), color="orange",
                 linewidth=2.0, linestyle="-");
        plt.plot(t, op1(z+0.1), color="darkred",
                 linewidth=2.0, linestyle="-");
    end

    # Test conversion to a sparse matrix.
    sm1 = sparse(op1)
    sm2 = sparse(op2)
    println("max. error 1: ", maximum(abs(op1(z) - sm1*z)))
    println("max. error 2: ", maximum(abs(op2(z) - sm2*z)))

    # 2-D example
    t = reshape(linspace(-3,14,20*21), (20,21));
    op = TiPi.Interpolator(TiPi.Kernels.catmull_rom, 1:length(z), t)
    if PLOTTING
        plt.figure(3)
        plt.clf()
        plt.imshow(op(z));
    end
end
interp_tests()
end # module
