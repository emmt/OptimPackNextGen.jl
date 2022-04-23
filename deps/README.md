# Automatic build of Julia bindings to a library

Edit files [`generator.toml`](./generator.toml) and
[`build_deps.jl`](./build_deps.jl).  In `build_deps.jl`, it should be
sufficient to change the value of `headers` (and the name of the TOML file if
you change it).

Execute the generator:

```sh
julia build_deps.jl
```
