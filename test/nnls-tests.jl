module NonLinearLeastSquaresTests

using YPlot
using OptimPackNextGen

f(p,x) = p[1] + p[2].*x

n = 20
x = linspace(-1,1,n)
y = 0.8 - 1.2*x + 0.1*randn(n)
w = ones(n)
plt.plot(x,y,"ro")

p, c = OptimPackNextGen.nllsq(y,f,[1,1],x)
plt.plot(x,f(p,x),"green")

p, c = OptimPackNextGen.nllsq(w,y,f,[1,1],x)
plt.plot(x,f(p,x),"orange")

end
