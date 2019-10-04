# Limited memory quasi-Newton methods

OptimPackNextGen provides some limited-memory quasi-Newton methods to minimize
differentiable objective functions of many variables.  Optionally, bounds on
the variables can be specified.


## Variable Metric Limited Memory with Bounds (VMLMB)

The main driver for these methods is `vmlmb` which is used as follows:

    x = vmlmb(fg!, x0; mem=..., lower=..., upper=..., ftol=..., fmin=...)

computes a local minimizer of a function of several variables by a limited
memory variable metric method.  The caller provides a function `fg!` to compute
the value and the gradient of the function as follows:

    f = fg!(x, g)

where `x` are the current variables, `f` is the value of the function at `x`
and `g` is the gradient at `x` (`g` is already allocated as `g = vcreate(x0)`).
Argument `x0` gives the initial approximation of the variables (its contents is
left unchanged).  The best solution found so far is returned in `x`.

The following keywords are available:

* `mem` specifies the amount of storage.

* `ftol` is a tuple of two nonnegative reals specifying respectively the
  absolute and relative errors desired in the function.  Convergence occurs if
  the estimate of the relative error between `f(x)` and `f(xsol)`, where `xsol`
  is a local minimizer, is less than `ftol[1]` or if the absolute error between
  `f(x)` and `f(xsol)` is less than `ftol[2]`.  By default, `ftol = (0.0,1e-8)`.

* `gtol` is a tuple of two nonnegative reals specifying the absolute and a
  relative thresholds for the norm of the gradient, convergence is assumed as
  soon as:

      ||g(x)|| <= hypot(gtol[1], gtol[2]*||g(x0)||)

  where `||g(x)||` is the Euclidean norm of the gradient at the current
  solution `x`, `||g(x0)||` is the Euclidean norm of the gradient at the
  starting point `x0`.  By default, `gtol = (0.0,1e-6)`.

* `fmin` specifies a lower bound for the function.  If provided, `fmin` is used
  to estimate the steepest desxecnt step length this value.  The algorithm
  exits with a warning if `f(x) < fmin`.

* `maxiter` specifies the maximum number of iterations.

* `maxeval` specifies the maximum number of calls to `fg!`.

* `verb` specifies whether to print iteration information (`verb = false`, by
  default).

* `printer` can be set with a user defined function to print iteration
  information, its signature is:

      printer(io::IO, iter::Integer, eval::Integer, rejects::Integer,
              f::Real, gnorm::Real, stp::Real)

  where `io` is the output stream, `iter` the iteration number (`iter = 0` for
  the starting point), `eval` is the number of calls to `fg!`, `rejects` is the
  number of times the computed direction was rejected, `f` and `gnorm` are the
  value of the function and norm of the gradient at the current point, `stp` is
  the length of the step to the current point.

* `output` specifies the output stream for printing information (`STDOUT` is
  used by default).

* `lnsrch` specifies the method to use for line searches (the default
   line search is `MoreThuenteLineSearch`).

* `lower` and `upper` specify the lower and upper bounds for the variables.
   The bound can be a scalar to indicate that all variables have the same bound
   value.  If the lower (resp. upper) bound is unspecified or set to `±∞`, the
   variables are assumed to be unbounded below (resp. above).  If no bounds are
   set, VMLMB amounts to an unconstrained limited memory BFGS method (L-BFGS).

* `blmvm` can be set true to emulate the BLMVM algorithm of Benson and Moré.
  This option has no effects for an uncostrained problem.


### In-place version

The method `vmlmb!` implements the in-place version of `vmlmb`:

     vmlmb!(fg!, x; mem=..., lower=..., upper=..., ftol=..., fmin=...) -> x

which finds a local minimizer of `f(x)` starting at `x` and stores the best
solution in `x`.


### History

The VMLMB algorithm in OptimPackNextGen.jl provides a pure Julia implementation
of the original method (Thiébaut, 2002) with some improvements and the
capability to emulate L-BFGS and BLMVM methods.

The limited memory BFGS method (L-BFGS) was first described by Nocedal (1980)
who dubbed it SQN.  The method is implemented in MINPACK-2 (1995) by the
FORTRAN routine VMLM.  The numerical performances of L-BFGS have been studied
by Liu and Nocedal (1989) who proved that it is globally convergent for
uniformly convex problems with a R-linear rate of convergence.  They provided
the FORTRAN code LBFGS.  The BLMVM and VMLMB algorithms were proposed by Benson
and Moré (2001) and Thiébaut (2002) to account for separable bound constraints
on the variables.  These two latter methods are rather different than L-BFGS-B
by Byrd at al. (1995) which has more overheads.

* J. Nocedal, "*Updating Quasi-Newton Matrices with Limited Storage*" in
  Mathematics of Computation, vol. 35, pp. 773-782 (1980).

* D.C. Liu & J. Nocedal, "*On the limited memory BFGS method for large scale
  optimization*" in Mathematical programming, vol. 45, pp. 503-528
  (1989).

* R.H. Byrd, P. Lu, J. Nocedal, & C. Zhu, "*A limited memory algorithm for
  bound constrained optimization*" in SIAM Journal on Scientific Computing,
  vol. 16, pp. 1190-1208 (1995).

* S.J. Benson & J.J. Moré, "*A limited memory variable metric method in
  subspaces and bound constrained optimization problems*" in Subspaces and
  Bound Constrained Optimization Problems (2001).

* É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*" in
  Astronomical Data Analysis II, Proc. SPIE 4847, pp. 174-183 (2002).
