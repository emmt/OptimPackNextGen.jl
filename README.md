[![Build Status](https://travis-ci.org/emmt/OptimPackNextGen.jl.svg?branch=master)](https://travis-ci.org/emmt/OptimPackNextGen.jl)

# OptimPackNextGen.jl

**OptimPackNextGen** is a [Julia](http://julialang.org/) package for numerical
optimization with particular focus on large scale problems.


## Large scale problems

* [Quasi-Newton methods](doc/quasinewton.md) can be used to solve nonlinear
  large scale optimization problems. Optionally, bounds on the variables can be
  taken into account.  The objective function must be differentiable and the
  caller must provide means to compute the objective function and its gradient.

* [Line searches methods](doc/linesearches.md) are used to approximately
  minimize the objective function along a given search direction.

* [Algebra](doc/algebra.md) describes operations on "vectors" (that is to say
  the "variables" of the problem to solve).


## Small to moderate size problems

For problems of small to moderate size, **OptimPackNextGen** provides:

* Mike Powell's **COBYLA** (see ref. [10]), **NEWUOA** (see ref. [11]), and
  **BOBYQA** (see ref. [12]) algorithms for minimizing a function of many
  variables.  These methods are *derivatives free* (only the function
  values are needed).  **NEWUOA** is for unconstrained optimization.
  **COBYLA** accounts for general inequality constraints.  **BOBYQA** accounts
  for bound constraints on the variables.


## Univariate functions

The following methods are provided for univariate functions:

* `Brent.fzero` implements van Wijngaarden–Dekker–Brent method for finding a
  zero of a function.

* `Brent.fmin` implements Brent's method for finding a minimum of a function.

* `Bradi.minimize` (resp. `Bradi.maximize`) implements the BRADI ("Bracket"
  then "Dig") method for finding the global minimum (resp. maximum) of a
  function on an interval.

* `Step.minimize` (resp. `Step.maximize`) implements the STEP method for
  finding the global minimum (resp. maximum) of a function on an interval.


## Installation

**OptimPackNextGen** is not yet an
[official Julia package](https://pkg.julialang.org/) so you have to clone the
repository and build the associated
[`OptimPack`](https://github.com/emmt/OptimPack) library:

   Pkg.clone("https://github.com/emmt/OptimPackNextGen.jl.git")
   Pkg.build("OptimPackNextGen")


## Rationale and related software

Related software are the [`OptimPack`](https://github.com/emmt/OptimPack)
library which implements the C version of the algorithms and the
[`OptimPack.jl`](https://github.com/emmt/OptimPack.jl) Julia package which is a
wrapper of this library for Julia.  Compared to `OptimPack.jl`, the new
`OptimPackNextGen.jl` implements in pure Julia the algorithms dedicated to
large scale problems but still relies on the C library for a few algorithms
(notably the Powell methods).  The rationale is to facilitate the integration
of exotic types of variables for optimization problems in Julia.  Eventually,
`OptimPackNextGen.jl` will become the next version of `OptimPack.jl` but, until
then, it is more flexible to have two separate modules and avoid coping with
compatibility and design issues.
