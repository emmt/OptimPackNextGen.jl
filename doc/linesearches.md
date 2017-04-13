# Line search methods

The objective of line search is to (approximately) minimize:

     ϕ(α) = f(x0 + α d)

for `α > 0` and with `x0` the variables at the start of the line search, `d`
the search direction and `α` the step lenght.

Implemented line search methods are sub-types of the `LineSearch{T}`
abstract type (which is parameterized by the floating-point type `T` for
the computations).  To use a given line search method, say
`SomeLineSearch`, first create the line search object:

    lnsrch = SomeLineSearch(T; ...)

with `T` the chosen floating-point type.

At every iterate `x0` of the optimization algorithm, the line search is
initiated with:

    task = start!(lnsrch, f0, df0, α)

where `f0 = f(x0)` and `df0 = ⟨ d, ∇f(x0) ⟩` are the function value and the
directional derivative at the position `x0`, and `α` is the lenght of the
first step to take.  For subsequent steps, checking the convergence of the
line search and computing the next step length are performed by:

    task = iterate!(lnsrch, f1, df1)

where `f1 = f(x + α d)` and `df1 = ⟨ d, ∇f(x + α d) ⟩` are the function value
and the directional derivative at the position `x0 + α d`, that is for a step
length `α` which is given by:

    α = getstep(lnsrch)

The result of the `start!` and `iterate!` methods is a symbol specifying the
next task to perform.  Normally, the task is either `:SEARCH` or `:CONVERGENCE`
to indicate whether the line search is in progress or has converged.  If the
returned task is `:SEARCH`, the caller should take the proposed step, compute
the function value and, possibly the directional derivative, and then call
`iterate!` again.  If the returned task is `:CONVERGENCE`, the line search has
converged and a new iterate is available for inspection (the next step length
is equal to the current one).  Other possible tasks are `:WARNING` to indicate
that line search is terminated because no more progresses are possible,
`:ERROR` to indicate an error or `:START` when linesearch has not yet been
started.  In that case, `getstatus(lnsrch)` and `getreason(lnsrch)` can be used
to retrieve a symbolic status of a textual description of the problem.

At any time, the current pending task and the step length to take
are obtainable by:

    task = gettask(lnsrch)
    α = getstep(lnsrch)

Although the directional derivative `df0` at the start of the line search is
required, not all line search methods need the directional derivative for
refining the step lenght (i.e. when calling `iterate!`).  To check that, call:

    usederivatives(lnsrch)

which return a boolean result.

For unconstrained optimization, the directional derivative is:

    ϕ'(α) = ⟨ d, ∇f(x0 + α d) ⟩

If simple bound constraints are implemented, the function `ϕ(α)` becomes:

    ϕ(α) = f(x(α))

where `x(α) = P(x0 + α d)` and `P(...)` is the projection onto the feasible
(convex) set.  Since `ϕ(α)` is then piecewise differentiable, it is recommended
to use line search methods which do not require the derivatives like
`ArmijoLineSearch` or `MoreToraldoLineSearch`.  If you insist on using a line
search methods which does require the derivatives like `MoreThuenteLineSearch`,
you may approximate the directional derivative by:

    ϕ'(α) ≈ ⟨ (x(α) - x(0))/α, ∇f(x(α)) ⟩

where `x(0) = x0` assuming that `x0` is feasible.

## References

* L. Armijo, "*Minimization of functions having Lipschitz continuous first
  partial derivatives*" in Pacific Journal of Mathematics, vol. 16, pp. 1–3
  (1966).

* L. Grippo, F. Lampariello & S. Lucidi, "*A Nonmonotone Line Search Technique
  for Newton’s Method*" in SIAM J. Num. Anal., vol. 23, pp. 707–716 (1986).

* J.J. Moré & G. Toraldo, "*On the Solution of Large Quadratic Programming
  Problems with Bound Constraints*" in SIAM J. Optim., vol. 1, pp. 93–113
  (1991).

* J.J. Moré and D.J. Thuente, "*Line search algorithms with guaranteed
  sufficient decrease*" in ACM Transactions on Mathematical Software, vol. 20,
  pp. 286–307 (1994).

* E.G. Birgin, J.M. Martínez & M. Raydan, "*Nonmonotone Spectral Projected
  Gradient Methods on Convex Sets*" in SIAM J. Optim., vol. 10, pp. 1196–1211
  (2000).
