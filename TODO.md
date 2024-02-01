* Implement reference line-search method.

* Cleanup URLS in `README.md` and switch to GitHub Actions.

* Brent.fzero: yield the endpoints of the bracket.

* Brent.fzero and Brent.fmin: automatic bracketting of the solution:

* Fix default tolerances to avoid excessive precision in `Brent.fmin` when the
  solution is very close to zero.

> If the interval to consider is not bounded or only left/right bounded, the
> idea is to find a suitable interval (A,B) where at least one minimum must
> exists (if the function is continue) and start Brent's algorithm with correct
> values for X, FX, ... (in order to save some function evaluations).

* Add BigFloat to the tests.

* Uniformize the tests for automatic check.

* Add verbose mode for `conjgrad`.

* Write doc. for `bobyqa`, `newuoa`, etc.

* Automatically build dependencies.

* Add `work` keyword to Powell's methods to avoid garbage collection.

* Hierarchy of types should be more consistent (`Status`?) and exported methods
  (`getreason`, `iterate`, etc.) should be shared.

* Reorganize things:
  - OptimPack.jl -> OptimPackLib.jl = Julia wrapper to use OptimPack library;
  - OptimPackNextGen.jl -> OptimPack.jl = pure Julia version of large scale
    optimization methods;
  - OptimPackNextGen.jl/Powell -> OptimPackPowell.jl = Julia wrapper to use
    Powell methods in OptimPack library;

* Uniformize convergence criteria and/or provide an optional user-defined
  function to decide for convergence.

* Automatic pre-conditioner (`k` is iteration number, `j` is variable index):
  * Uniform scaling:

    ```
    γ_k = min_γ ||γ*y_{k} - s_{k}||^2
        = β_k/α_k
    α_{k} = y_{k}'*y_{k}
    β_{k} = y_{k}'*s_{k}
    ```

    Note for VMLMB: the scalar products may or may not be computed on the free
    variables.

  * Uniform scaling with forgetting factor `0 ≤ η ≤ 1`:

    ```
    γ_k = min_γ Σ_{i≥0} η^i ||γ*y_{k-i} - s_{k-i}||^2
        = β_k/α_k
    α_{k} = η*α_{k-1} + y_{k}'*y_{k}
    β_{k} = η*β_{k-1} + y_{k}'*s_{k}
    ```

    Notes: (i) Same remark as above. (ii) When `η = 0`, the usual scaling is
    retrieved. (iii) Thanks to the forgetting factor, the memory can be longer
    than `m` the number of memorized steps.

  * Non-uniform scaling with forgetting factor (`G` is a diagonal operator):

    ```
    G_k = min_G Σ_{i≥0} η^i ||G*y_{k-i} - s_{k-i}||^2
        = b_k/a_k (elementwise)
    a_{k,j} = η*a_{k-1,j} + y_{k-i,j}^2
    b_{k,j} = η*b_{k-1,j} + y_{k-i,j}*s_{k-i,j}
    ```

* Limited Memory Symmetric Rank One (LMSR1) model of the (inverse) Hessian:
  * Define SR1 model `B_k` of Hessian based on `m` previous steps.
  * Let `M_k` be some diagonal positive definite operator (`M_{k} = inv(G_{k})`
    see above).
  * Define the step as:
    ```
    s_{k+1} = argmin_s g(x_{k})'*s + (1/2)*s'*B_{k}*s + (λ_{k}/2)*s'*M_{k}*s
            = -(B_{k} + λ_{k}*M_{k})^{-1}*g(x_{k})
    ```
    or use truncated conjugate gradients (preconditioned by `M_{k}`) to tune
    the step length.
