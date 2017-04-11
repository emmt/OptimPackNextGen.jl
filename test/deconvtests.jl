module Test

using Base.Test
using TiPi
import Base: DFT, FFTW

function runtests{T<:AbstractFloat}(::Type{T}=Float64)
    n1, n2, n3 = 384, 128, 32
    for dims in ((n1,), (n1,n2), (n1,n2,n3))
        println("\nTesting $(length(dims))-D convolution with dims = $dims")
        println(" * Real convolution:")
        x = rand(T, dims)
        h = rand(T, dims)
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
        x = complex(rand(T, dims),rand(T, dims))
        h = complex(rand(T, dims),rand(T, dims))
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

function correltests{T<:AbstractFloat}(::Type{T}=Float64)
    dim = 3*32
    xdims = (dim, dim)
    zdims = (div(dim,2) + 1, dim)

    #planning = FFTW.ESTIMATE
    planning = FFTW.MEASURE

    n = 100000
    t = 1
    FFTW.set_num_threads(t)

    println("\nTesting 2D convolution with dims = $xdims, $n times, $t thread(s)")

    println(" * Real convolution:")

    x1 = Array(T, xdims)
    x2 = Array(T, xdims)
    x3 = Array(T, xdims)
    z1 = Array(Complex{T}, zdims)
    z2 = Array(Complex{T}, zdims)
    z3 = Array(Complex{T}, zdims)

    forward = plan_rfft(x1; flags=(planning | FFTW.PRESERVE_INPUT))
    backward = plan_brfft(z1, xdims[1]; flags=(planning | FFTW.DESTROY_INPUT))

    rand!(x1)
    rand!(x2)

    # Warm-up and compile
    for k in 1:max(10,div(n,100))
        A_mul_B!(z1, forward, x1)
        A_mul_B!(z2, forward, x2)
        @inbounds @simd for i in eachindex(z1, z2, z3)
            z3[i] = z1[i]*z2[i]
        end
        A_mul_B!(x3, backward, z3)
    end

    @time for k in 1:n
        A_mul_B!(z1, forward, x1)
        A_mul_B!(z2, forward, x2)
        @inbounds @simd for i in eachindex(z1, z2, z3)
            z3[i] = z1[i]*z2[i]
        end
        A_mul_B!(x3, backward, z3)
    end
end

function deconvtest(test::String="conjgrad"; single::Bool=false)
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
