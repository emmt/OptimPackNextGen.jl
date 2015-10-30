Convex Sets
===========

A convex set `Ω` is such that it contains any segment whose endpoints belong to
the set `Ω`.

The abstract base type for any convex set is `ConvexSet`.

For instance the type `SimplyBoundedSet` inherits from `ConvexSet` and is
meant to serve a a base for any set consisting in separable bounds.

The following operations may be defined for a convex set `dom`.  For
efficiency reasons, the destination `dst` to store the result of the
operation is always sepcified.

```Julia
    project_variables!(dst, dom, x)
```
to store in the destination *vector* `dst` the orthogonal projection of the
variables `x` onto the domain `dom`.

```Julia
    project_gradient!(dst, dom, x, g)
```
to store in the destination *vector* `dst` the projection of the gradient
`g` at the point `x` onto the domain `dom`.  The variables `x` shall be
feasible (*i.e.* they belong to `dom`).

```Julia
    project_direction!(dst, dom, x, d)
```
to store in the destination *vector* `dst` the projection of the direction
`d` at the point `x` onto the domain `dom`.  The variables `x` shall be
feasible (*i.e.* they belong to `dom`).

In principle:
```Julia
    project_gradient!(gp, dom, x, g)
    project_direction!(dp, dom, x, -g)
```
yields a projected gradient `gp` and a projected direction `dp` such that
```Julia
    db = -gp
```
This is how the default (because it has the least specialized signature)
`project_direction!` method.


Bounds along a search direction
-------------------------------

To avoid making too large steps or exploit that a straight segment of the
search path belongs to the feasible domain, it is necessary to figure out
the bounds of the step length along a search direction.

The method:
```Julia
	amin = minimum_step(dom, x, d)
```
returns `amin`, the shortest step size for which a bound is encountered
along the search direction`d` starting at `x`.  The straight segment
`x + α*d` such that `0 ≤ α ≤ amin`, completely belongs to the feasible set
`dom`.  If there is no limits, `amin = ∞` is returned.

The method:
```Julia
    amax = maximum_step(dom, x, d)
```

returns `amax`, the longest step size for which a bound is encountered
along the search direction`d` starting at `x`.  This means that the
projection of `x + α*d` onto the feasible set `dom` yields the same result
as that of `x + amax*d` for any `α ≥ amax` and a different result otherwise
(*i.e.* for `0 ≤ α < amax`).

It is guaranteed that `0 ≤ amin ≤ amax ≤ ∞`.

The method:
```Julia
    (amin, amax) = step_bounds(dom, x, d)
```
returns the two bounds.  It may be faster to call the method `step_bounds`
rather than calling `minimum_step` and `maximum_step`, which is however
what is done by the default implementation.

Finally, the method:
```Julia
   acut = shortcut_step(alpha, dom, x, d)
```
returns:
```Julia
   acut = min(alpha, maximum_step(dom, x, d))
```
but may be implemnted more efficiently.

