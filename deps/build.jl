using BinDeps
using Compat

const version = "3.0.1"
const unpacked_dir = "optimpack-$version"

@BinDeps.setup

optimpack = library_dependency("libopk")

provides(Sources,
         URI("https://github.com/emmt/OptimPack/releases/download/v$version/optimpack-$version.tar.gz"),
         optimpack,
         unpacked_dir=unpacked_dir)

prefix = joinpath(BinDeps.depsdir(optimpack), "usr")
srcdir = joinpath(BinDeps.depsdir(optimpack), "src", unpacked_dir)
libdir = joinpath(prefix, "lib")
name = "opk"
@compat @static if is_unix()
    libfilename = "lib$(name).so"
elseif is_apple()
    libfilename = "lib$(name).dylib"
#elseif is_windows()
#    libfilename = "$(name).dll"
else
    error("unknown architecture")
end
destlib = joinpath(libdir, libfilename)

provides(SimpleBuild,
         (@build_steps begin
             GetSources(optimpack)
             CreateDirectory(prefix)
             CreateDirectory(joinpath(prefix,"lib"))
             @build_steps begin
                 ChangeDirectory(srcdir)
                 FileRule(destlib,
                          @build_steps begin
                              `./configure --enable-shared --disable-static --prefix="$prefix"`
                              `make`
                              `make install`
                          end)
             end
         end),
         optimpack)

@BinDeps.install Dict(:libopk => :opklib)

# List installed libraries (for debugging installation).
run(`ls -l "$libdir"`)
