module TiPiTests

const PLOTTING = true
if PLOTTING
    using PyCall; pygui(:gtk); # can be: :wx, :gtk, :qt
    using LaTeXStrings
    import PyPlot; const plt = PyPlot
end
include("../src/kernels.jl")

function runtests()
    for (name, ker) in (("Box", Kernels.box),
                        ("Box (type)", Kernels.BoxKernel{Float32}),
                        ("Catmull-Rom",  Kernels.catmull_rom),
                        ("Mitchell-Netravali",  Kernels.mitchell_netravili),
                        ("cardinal Mitchell-Netravali",
                         Kernels.MitchellNetraviliKernel(Float16, 0, 1)),
                        ("Duff's tensioned B-spline",
                         Kernels.MitchellNetraviliKernel(Float32, 0.5, 0)))
        println(name," kernel:")
        println(" - support: ", length(ker))
        println(" - normalized: ", Kernels.isnormalized(ker))
        println(" - cardinal: ", Kernels.iscardinal(ker))
        println(" - ker(0): ", ker(0)) # test conversion
    end

    plt.figure(1)
    plt.clf()
    x = linspace(-5,5,1000);
    plt.plot(x, Kernels.box(x), color="darkgreen",
             linewidth=2.0, linestyle="-");
    plt.plot(x, Kernels.triangle(x), color="darkblue",
             linewidth=2.0, linestyle="-");
    plt.plot(x, Kernels.quadratic(x), color="darkcyan",
             linewidth=2.0, linestyle="-");
    plt.plot(x, Kernels.cubic(x), color="firebrick",
             linewidth=2.0, linestyle="-");
    plt.plot(x, Kernels.catmull_rom(x), color="orange",
             linewidth=2.0, linestyle="-");
    plt.plot(x, Kernels.mitchell_netravili(x), color="violet",
             linewidth=2.0, linestyle="-");
    plt.title("Some kernel functions");
end

runtests()

end # module
