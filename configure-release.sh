#!/bin/sh

SRC_PATH=`dirname $0`

# TODO: try -mavx2 -mfma
CXXFLAGS="-O3 -DNDEBUG=1 -DBOOST_UBLAS_NDEBUG"
LDFLAGS="-O3"

$SRC_PATH/configure CFLAGS= CPPFLAGS= CXXFLAGS="$CXXFLAGS" LD_FLAGS="$LDFLAGS" 

