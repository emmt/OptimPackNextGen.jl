using Libdl
using Clang.Generators
using Clang.LibClang.Clang_jll

lib_file = "libopk.$(Libdl.dlext)"
inc_file = "optimpack.h"
search_prefixes = [joinpath(ENV["HOME"], "apps"),
    "/apps", "/usr/local", "/opt",]

# We wrap everything into a function to avoid having undefined variables...
function build_deps()
    # Find the directory where header files are installed.
    inc_dir = ""
    for dir in [get(ENV, "OPK_INCDIR", ""),
                map(x -> joinpath(x, "include"), search_prefixes)...]
        if dir != "" && isfile(joinpath(dir, inc_file))
            inc_dir = dir
            break
        end
    end
    if inc_dir == ""
        @error("Header \"$inc_file\" not found.  Try to set environment variable OPK_INCDIR with the path of the directory where OptimPack header files have been installed and re-build the package.")
        error("Header \"$inc_file\" not found.")
    end

    # Find the directory where libraries are installed.
    lib_dir = ""
    for dir in [get(ENV, "OPK_LIBDIR", ""),
                map(x -> joinpath(x, "lib"), search_prefixes)...]
        if dir != "" && isfile(joinpath(dir, lib_file))
            lib_dir = dir
            break
        end
    end
    if lib_dir == ""
        @warn("Library \"$lib_file\" not found.  Try to set environment variable OPK_LIBDIR with the path of the directory where OptimPack libraries have been installed and re-build the package.")
        #error("Library \"$lib_file\" not found.")
    end

    # Header files.
    headers = map(x -> joinpath(inc_dir, x), [
        "bobyqa.h",
        "cobyla.h",
        "newuoa.h",
        "optimpack.h",
        "optimpack-linalg.h",
        "optimpack-private.h",
    ])

    # List of (unique and in order) include directories.
    include_dirs = String[]
    for dir in Iterators.map(dirname, headers)
        dir in include_dirs || push!(include_dirs, dir)
    end

    # The rest is pretty standard.
    cd(@__DIR__)
    options = load_options(joinpath(@__DIR__, "generator.toml"))
    args = get_default_args()
    for dir in include_dirs
        push!(args, "-I$dir")
    end
    #push!(args, "-DNC_FUNCTION=extern")
    push!(args, "-I.")
    ctx = create_context(headers, args, options)
    build!(ctx)

    # Rewrite destination file.
    dest_file = options["general"]["output_file_path"]
    code = readlines(dest_file)
    lib_path = (lib_dir == "" ? lib_file : joinpath(lib_dir, lib_file))
    for repl in [
        r"\s+$" => "",
        r"\b(objfun|fc)::Ptr\{Cvoid\},\s*data::Ptr\{Cvoid\}\s*" => s"\1::Ptr{Cvoid}, data::Any",
        #r"\blibopk.cobyla" => "libcobyla.cobyla",
        #r"\blibopk.newuoa" => "libnewuoa.newuoa",
        #r"\blibopk.bobyqa" => "libbobyqa.bobyqa",
        r"^(\s*const\s+OPK_STATUS_LIST_\s*=)" => s"# Skipping MacroDefinition: \1",
    ]
        for i in eachindex(code)
            code[i] = replace(code[i], repl)
        end
    end
    open(dest_file, "w") do io
        foreach(line -> println(io, line), code)
    end
end

# Run the build script.
build_deps()
