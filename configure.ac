dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(YANN, 1.0)
AC_CONFIG_MACRO_DIRS([m4])

AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([foreign subdir-objects])
LT_INIT()

AC_PROG_CXX

AM_EXTRA_RECURSIVE_TARGETS([
  check-funcs 
  check-layers
  check-fcnn
  check-cnn
  check-custom
  check-word2vec
])

# Provide special options for INTEL MKL
AC_MSG_CHECKING([whether INTEL's MKL is enabled])
AC_ARG_ENABLE(mkl,
  [AS_HELP_STRING([--enable-mkl], [Use INTEL's MKL for Eigen backend (default = no)])],
  AC_MSG_RESULT([yes]),
  AC_MSG_RESULT([no])
)
if test "$enable_mkl" = "yes"; then
  AC_CHECK_HEADERS(mkl.h)
  AC_CHECK_LIB(mkl_core)

  AM_CXXFLAGS="$AM_CXXFLAGS -DEIGEN_USE_MKL_ALL=1"
  AM_LDFLAGS="$AM_LDFLAGS -lmkl_sequential -lmkl_core -lmkl_blas95_lp64 -lblas64 -lmkl_intel_lp64 -llapack64 -lmkl_intel_lp64  -lpthread -lm -ldl"
fi


AC_SUBST(AM_CXXFLAGS)
AC_SUBST(AM_LDFLAGS)
AC_CONFIG_FILES(Makefile src/Makefile)
AC_OUTPUT

