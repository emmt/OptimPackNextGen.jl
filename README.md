# OptimPack.jl

**OptimPack** is a package/library for numerical optimization in particular
large scale problems.


## Large scale problems

* [Quasi-Newton methods](doc/quasinewton.md) can be used to solve nonlinear
  large scale optimization problems. Optionally, bounds on the variables can be
  taken into account.  The objective function must be differentiable and the
  caller must provide means to compute the objective function and its gradient.

* [Line searches methods](doc/linesearches.md) are used to approximately
  minimize the objective function along a given serach direction.

* [Algebra](doc/algebra.md) describes operations on "vectors" (that is to say
  the "variables" of the problem to solve).


## Univariate functions

The following methods are provided for univariate functions:

* `Brent.fzero` implements van Wijngaarden–Dekker–Brent method for finding a
  zero of a function.

* `Brent.fmin` implements Brent's method for finding a minimum of a function.

* `Bradi.minimize` (resp. `Bradi.maximize`) implements the BRADI ("Bracket"
  then "Dig") method for finding a global minimum (resp. maximum) of a
  function.

* `Step.minimize` (resp. `Step.maximize`) implements the STEP method for
  finding a global minimum (resp. maximum) of a function.
