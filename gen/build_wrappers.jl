# Script to parse PRIMA headers and generate Julia wrappers.
using OptimPack_jll
using Clang
using Clang.Generators
using JuliaFormatter

function build_wrappers(
    filename::AbstractString = joinpath(
        @__DIR__, "..", "src", "wrappers.jl"))

    cd(@__DIR__)
    incdir = joinpath(OptimPack_jll.artifact_dir, "include")
    headers = map(x -> joinpath(incdir, x), ["bobyqa.h", "cobyla.h", "newuoa.h"])

    options = load_options(joinpath(@__DIR__, "generator.toml"))
    options["general"]["output_file_path"] = filename

    args = get_default_args()
    push!(args, "-I$incdir")

    ctx = create_context(headers, args, options)
    build!(ctx)

    code = readlines(filename)
    for repl in [
        # Replace the type of the client-data by `Any` to allow for having an
        # immutable object (the objective function) specified as that argument.
        r"\b(fc::Ptr\{cobyla_calcfc\}|objfun::Ptr\{(bobyqa|newuoa)_objfun\})\s*,\s*data::Ptr\{Cvoid\}\s*" => s"\1, data::Any",
        ]
        for i in eachindex(code)
            code[i] = replace(code[i], repl)
        end
    end
    open(filename, "w") do io
        foreach(line -> println(io, line), code)
    end
    format_file(filename, YASStyle())
    return nothing
end

# If we want to use the file as a script with `julia wrapper.jl`
if abspath(PROGRAM_FILE) == @__FILE__
     build_wrappers()
end
