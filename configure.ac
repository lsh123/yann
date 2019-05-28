dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(YANN, 1.0)
AC_CONFIG_MACRO_DIRS([m4])

AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([subdir-objects])
LT_INIT()

AC_PROG_CXX

AM_EXTRA_RECURSIVE_TARGETS([check-funcs check-layers check-fcnn check-cnn check-custom])

AC_CONFIG_FILES(Makefile src/Makefile)
AC_OUTPUT

