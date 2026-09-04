# Covering-family plugin: one v1-ABI translation unit (dct_hc.cu) serving
# PROB_HAMMING (0), PROB_TORUS_LINF (1) and PROB_LEE (2).
#
# IN-TREE since 2026-08-24: the transitional arrangement (compile the
# prototype engine's source unmodified until PLUGIN.md v1 landed) is over.
# Base = verbatim sync of the prototype of 2026-08-22 ~14:15 UTC; the Lee
# problem family is an in-tree addition.  Provenance notes: VENDORED.md.

add_library(coldcore_covering SHARED dct_hc.cu)
set_target_properties(coldcore_covering PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)
message(STATUS "covering plugin: in-tree source "
               "${CMAKE_CURRENT_SOURCE_DIR}/dct_hc.cu")
