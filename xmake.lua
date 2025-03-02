add_rules("mode.debug", "mode.release")

set_languages("c++17")
set_optimize("fastest")

add_defines("THREDED_MATRIX_OPS")

add_requires("libsdl2")


target("Neural Engine")
    set_kind("binary")
    add_packages("libsdl2")
    add_includedirs("include")
    add_files("src/*.cpp")
    add_links("ws2_32")