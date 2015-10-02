include("../src/TiPi.jl")

function deconvtest1()
    dir = "../data"
    y = MDA.read(dir*"saturn.mda")
    psf = ifftshift(zeropad(MDA.read(dir*"saturn_psf.mda"), size(y)))
    mtf = rfft(ifftshift(zeropad(psf, size(y))))
end

function deconvtest2()
    dir = "../data/"
    data = TiPi.MDA.read(dir*"saturn.mda")
    psf = TiPi.MDA.read(dir*"saturn_psf.mda")
    x = fill(zero(eltype(data)), (640,640))
    q = TiPi.Deconv.init(psf, data, size(x), [1e-2,1e-2])
    TiPi.conjgrad!(q, x, 0.0, 50);
    TiPi.MDA.write(x,"/tmp/tipideconvtest.mda")
end

nothing
