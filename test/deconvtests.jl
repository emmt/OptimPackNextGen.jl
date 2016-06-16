module Test

using Base.Test
using TiPi

function runtests()
    n1, n2, n3 = 384, 128, 32
    for dims in ((n1,), (n1,n2), (n1,n2,n3))
        println("\nTesting $(length(dims))-D convolution with dims = $dims")
        println(" * Real convolution:")
        x = rand(dims)
        h = rand(dims)
        H = CirculantConvolution(h, flags=FFTW.MEASURE)
        y = H*x
        z = real(ifft(fft(h).*fft(x)))
        err = maximum(abs(y - z))/maximum(abs(z))
        println("    > max. err. with fft:  $err")
        z = irfft(rfft(h).*rfft(x), n1)
        err = maximum(abs(y - z))/maximum(abs(z))
        println("    > max. err. with rfft: $err")
        n = length(x)
        m = max(1, round(Int, 1e9/(n*log2(n))))
        println("    > timings for $m convolutions ($(round(Int,m*n*log2(n))) operations):")
        print("       - simple code with fft:   ")
        @time for k in 1:m; z = real(ifft(fft(h).*fft(x))); end
        print("       - simple code with rfft:  ")
        @time for k in 1:m; z = irfft(rfft(h).*rfft(x), n1); end
        print("       - TiPi operator (apply):  ")
        @time for k in 1:m; y = H*x; end
        print("       - TiPi operator (apply!): ")
        @time for k in 1:m; apply!(y, H, x); end

        println(" * Complex convolution:")
        x = complex(rand(dims),rand(dims))
        h = complex(rand(dims),rand(dims))
        H = CirculantConvolution(h, flags=FFTW.MEASURE)
        y = H*x
        z = ifft(fft(h).*fft(x))
        err = maximum(abs(y - z))/maximum(abs(z))
        println("    > max. err. with fft:  $err")
        println("    > timings for $m convolutions ($(round(Int,m*n*log2(n))) operations):")
        print("       - simple code with fft:   ")
        @time for k in 1:m; z = ifft(fft(h).*fft(x)); end
        print("       - TiPi operator (apply):  ")
        @time for k in 1:m; y = H*x; end
        print("       - TiPi operator (apply!): ")
        @time for k in 1:m; apply!(y, H, x); end
    end

end

function deconvtest(test::ASCIIString="conjgrad"; single::Bool=false)
    dir = "../data/"
    y = TiPi.MDA.read(dir*"saturn.mda")
    h = TiPi.MDA.read(dir*"saturn_psf.mda")
    wgt = TiPi.MDA.read(dir*"saturn_wgt.mda")
    T = single ? Cfloat : Cdouble
    y = convert(Array{T}, y)
    h = convert(Array{T}, h)
    wgt = convert(Array{T}, wgt)
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
        @time TiPi.Algebra.conjgrad!(prob, x, 0.0, 1000);
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
        if test == "vmlmb"
            TiPi.QuasiNewton.vmlmb!(fg!, x, mem=3, lower=0, maxeval=500,
                                    verb=true, gtol=(0.0,0.0))
            TiPi.MDA.write(x,"/tmp/tipideconvtest-vmlmb.mda")
        elseif test == "blmvm"
            TiPi.QuasiNewton.vmlmb!(fg!, x, mem=3, lower=0, maxeval=500,
                                    verb=true, gtol=(0.0,0.0), blmvm=true)
            TiPi.MDA.write(x,"/tmp/tipideconvtest-blmvm.mda")
        elseif test == "lbfgs"
            TiPi.QuasiNewton.vmlmb!(fg!, x, mem=3, maxeval=500,
                                    verb=true, gtol=(0.0,0.0))
            TiPi.MDA.write(x,"/tmp/tipideconvtest-lbfgs.mda")
        end
    end
end

nothing

end # module
