include("../src/TiPi.jl")

using OptimPack

function deconvtest1()
    dir = "../data"
    y = MDA.read(dir*"saturn.mda")
    psf = ifftshift(zeropad(MDA.read(dir*"saturn_psf.mda"), size(y)))
    mtf = rfft(ifftshift(zeropad(psf, size(y))))
end

function deconvtest2()
    dir = "../data/"
    y = TiPi.MDA.read(dir*"saturn.mda")
    h = TiPi.MDA.read(dir*"saturn_psf.mda")
    wgt = TiPi.MDA.read(dir*"saturn_wgt.mda")
    if true
        x = fill(zero(eltype(y)), (640,640))
        prob = TiPi.Deconv.init(h, y, wgt, size(x), [5e-3,5e-3];
                                normalize=true, verbose=true)
    else
        x = fill(zero(eltype(y)), size(y))
        prob = TiPi.Deconv.init(h, y, size(x), [5e-3,5e-3];
                                normalize=true, verbose=true)
    end
    @time TiPi.conjgrad!(prob, x, 0.0, 1000);
    TiPi.MDA.write(x,"/tmp/tipideconvtest.mda")
end

function deconvtest3()
    dir = "../data/"
    y = TiPi.MDA.read(dir*"saturn.mda")
    h = TiPi.MDA.read(dir*"saturn_psf.mda")
    wgt = TiPi.MDA.read(dir*"saturn_wgt.mda")
    T = eltype(y)
    x = fill(zero(T), (640,640))
    #for i in 1:length(x)
    #    x[i] = convert(T,i)
    #end
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
    f = TiPi.BLMVM.blmvm!(fg!, x, 3, dom, maxiter=100, verb=1)
    #x = vmlm(fg!, x, 5, verb=true, maxiter=100)
    TiPi.MDA.write(x,"/tmp/tipideconvtest-new.mda")
end

nothing
