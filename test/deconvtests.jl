include("../src/TiPi.jl")

function deconvtest(test::ASCIIString="conjgrad")
    dir = "../data/"
    y = TiPi.MDA.read(dir*"saturn.mda")
    h = TiPi.MDA.read(dir*"saturn_psf.mda")
    wgt = TiPi.MDA.read(dir*"saturn_wgt.mda")
    T = eltype(y)
    if test == "conjgrad"
        if true
            x = fill(zero(T), (640,640))
            prob = TiPi.Deconv.init(h, y, wgt, size(x), [5e-3,5e-3];
                                    normalize=true, verbose=true)
        else
            x = fill(zero(T), size(y))
            prob = TiPi.Deconv.init(h, y, size(x), [5e-3,5e-3];
                                    normalize=true, verbose=true)
        end
        @time TiPi.conjgrad!(prob, x, 0.0, 1000);
        TiPi.MDA.write(x,"/tmp/tipideconvtest.mda")
    else
        x = fill(zero(T), (640,640))
        lkl = TiPi.Deconv.deconvparam(h, y, size(x);
                                      normalize=true, verbose=true)
        rgl1 = TiPi.HyperbolicEdgePreserving(1.0, (1.0,1.0))
        rgl2 = TiPi.QuadraticSmoothness{2}()
        function fg!(x, g)
            if true
                return (TiPi.cost!(1, lkl, x, g, true) +
                        TiPi.cost!(0.8, rgl1, x, g, false))

            else
                return (TiPi.cost!(1, lkl, x, g, true) +
                        TiPi.cost!(5e-3, rgl2, x, g, false))
            end
        end
        dom = TiPi.ConvexSets.ScalarLowerBound(zero(T))
        if test == "vmlmb"
            f = TiPi.Optimization.vmlmb!(fg!, x, 3, dom, maxeval=500, verb=1,
                                         gtol=(0.0,0.0))
            TiPi.MDA.write(x,"/tmp/tipideconvtest-vmlmb.mda")
        elseif test == "blmvm"
            f = TiPi.Optimization.blmvm!(fg!, x, 3, dom, maxeval=500, verb=1,
                                         gtol=(0.0,0.0))
            #x = vmlm(fg!, x, 5, verb=true, maxiter=100)
            TiPi.MDA.write(x,"/tmp/tipideconvtest-blmvm.mda")
        end
    end
end

nothing
