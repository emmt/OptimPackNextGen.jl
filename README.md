# OptimPackNextGen.jl

[![License][license-img]][license-url]
[![Build Status][travis-img]][travis-url]
[![Appveyor][appveyor-img]][appveyor-url]
[![Coveralls][coveralls-img]][coveralls-url]
[![Codecov.io][codecov-img]][codecov-url]


`OptimPackNextGen` is a [Julia](http://julialang.org/) package for numerical
optimization with particular focus on large scale problems.


## Large scale problems

* [Quasi-Newton methods](doc/quasinewton.md) can be used to solve nonlinear
  large scale optimization problems. Optionally, bounds on the variables can be
  taken into account. The objective function must be differentiable and the
  caller must provide means to compute the objective function and its gradient.
  If the [`Zygote`](https://github.com/FluxML/Zygote.jl) is loaded, the
  gradient of the objective function may be computed by means of
  automatic-differentiation.

* *Spectral Projected Gradient* (SPG) method is provided for large-scale
  optimization problems with a differentiable objective function and convex
  constraints. The caller of `spg` (or `spg!`) shall provide a couple of
  functions to compute the objective function and its gradient and to project
  the variables on the feasible set. If the
  [`Zygote`](https://github.com/FluxML/Zygote.jl) is loaded, the gradient of
  the objective function may be computed by means of automatic-differentiation.

* [Line searches methods](doc/linesearches.md) are used to approximately
  minimize the objective function along a given search direction.


## Univariate functions

The following methods are provided for univariate functions:

* `Brent.fzero` implements van Wijngaarden–Dekker–Brent method for finding a
  zero of a function (Brent, 1973).

* `Brent.fmin` implements Brent's method for finding a minimum of a function
  (Brent, 1973).

* `Bradi.minimize` (resp. `Bradi.maximize`) implements the BRADI ("Bracket"
  then "Dig"; Soulez *et al.*, 2014) method for finding the global minimum
  (resp. maximum) of a function on an interval.

* `Step.minimize` (resp. `Step.maximize`) implements the STEP method (Swarzberg
  *et al.*, 1994) for finding the global minimum (resp. maximum) of a function
  on an interval. The objective function `f(x)` and the variable `x` may have
  units.


## Installation

The easiest way to install `OptimPackNextGen` is via Julia registry
[`EmmtRegistry`](https://github.com/emmt/EmmtRegistry):

```julia
using Pkg
pkg"registry add General"  # if not yet any registries
pkg"registry add https://github.com/emmt/EmmtRegistry"
pkg"add OptimPackNextGen"
```


## Rationale and related software

Related software are the [`OptimPack`](https://github.com/emmt/OptimPack)
library which implements the C version of the algorithms and the
[`OptimPack.jl`](https://github.com/emmt/OptimPack.jl) Julia package which is a
wrapper of this library for Julia. Compared to `OptimPack.jl`, the new
`OptimPackNextGen.jl` provide pure Julia implementation of most algorithms and
does not rely on foreign libraries. The exception are derivative-free Powell's
methods which are now provided by
[`PowellMethods.jl`](https://github.com/emmt/PowellMethods.jl) and, better, by
[`PRIMA.jl`](https://github.com/libprima/PRIMA.jl). The rationale is to
facilitate the integration of exotic types of variables for optimization
problems in Julia. Eventually, `OptimPackNextGen.jl` will become the next
version of `OptimPack.jl` but, until then, it is more flexible to have two
separate modules and avoid coping with compatibility and design issues.


## References

* S.J. Benson & J.J. Moré, "*A limited memory variable metric method in
  subspaces and bound constrained optimization problems*", in Subspaces and
  Bound Constrained Optimization Problems, (2001).

* E.G. Birgin, J.M. Martínez & M. Raydan, "*Nonmonotone Spectral Projected
  Gradient Methods on Convex Sets*," SIAM J. Optim. **10**, 1196-1211 (2000).

* R.P. Brent, "*Algorithms for Minimization without Derivatives*,"
  Prentice-Hall, Inc. (1973).

* W.W. Hager & H. Zhang, "*A New Conjugate Gradient Method with Guaranteed
  Descent and an Efficient Line Search*," SIAM J. Optim., Vol. 16, pp. 170-192
  (2005).

* W.W. Hager & H. Zhang, "*A survey of nonlinear conjugate gradient methods*,"
  Pacific Journal of Optimization, Vol. 2, pp. 35-58 (2006).

* M.R. Hestenes & E. Stiefel, "*Methods of Conjugate Gradients for Solving
  Linear Systems*," Journal of Research of the National Bureau of Standards 49,
  pp. 409-436 (1952).

* D. Liu and J. Nocedal, "*On the limited memory BFGS method for large scale
  optimization*", Mathematical Programming B **45**, 503-528 (1989).

* J.J. Moré & D.C. Sorensen, "*Computing a Trust Region Step*," SIAM J. Sci.
  Stat. Comp. **4**, 553-572 (1983).

* J.J. Moré and D.J. Thuente, "*Line search algorithms with guaranteed
  sufficient decrease*" in ACM Transactions on Mathematical Software (TOMS)
  Volume 20, Issue 3, Pages 286-307 (1994).

* F. Soulez, É. Thiébaut, M. Tallon, I. Tallon-Bosc & P. Garcia, "*Optimal a
  posteriori fringe tracking in optical interferometry*" in Proc. SPIE 9146
  "*Optical and Infrared Interferometry IV*", 91462Y (2014);
  [doi:10.1117/12.2056590](http://dx.doi.org/10.1117/12.2056590).

* T. Steihaug, "*The conjugate gradient method and trust regions in large scale
  optimization*", SIAM Journal on Numerical Analysis, vol. **20**, pp. 626-637
  (1983).

* S. Swarzberg, G. Seront & H. Bersini, "*S.T.E.P.: the easiest way to optimize
  a function*" in IEEE World Congress on Computational Intelligence,
  Proceedings of the First IEEE Conference on Evolutionary Computation, vol.
  **1**, pp. 519-524 (1994).

* É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*," in
  Astronomical Data Analysis II, SPIE Proc. **4847**, 174-183 (2002).

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/OptimPackNextGen.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/OptimPackNextGen.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.org/emmt/OptimPackNextGen.jl.svg?branch=master
[travis-url]: https://travis-ci.org/emmt/OptimPackNextGen.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/OptimPackNextGen.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/OptimPackNextGen-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/OptimPackNextGen.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/OptimPackNextGen.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/OptimPackNextGen.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/OptimPackNextGen.jl?branch=master
