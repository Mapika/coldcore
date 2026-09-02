# Explicit-incidence plugin: ABI 2 (flat sparse_* symbols, docs/PLUGIN.md
# "ABI 2: explicit incidence").  Independent of the separable-transform
# plugin: no shared translation units, no shared globals.

add_library(coldcore_sparse SHARED sparse.cu)
set_target_properties(coldcore_sparse PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)
message(STATUS "sparse plugin: ${CMAKE_CURRENT_SOURCE_DIR}/sparse.cu")
