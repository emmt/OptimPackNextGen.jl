# Wrapping headers

This directory contains a script that can be used to automatically generate
wrappers from the C headers provided by `OptimPack_jll` artifact. This is
accomplished by `Clang.jl`.

# Usage

Either run `julia build_wrappers.jl` directly, or include it and call the
`build_wrappers()` function. Be sure to activate the project environment in
this folder (`julia --project`), which will install `Clang.jl` and
`JuliaFormatter.jl`.
