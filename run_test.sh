#!/bin/sh

SRC_PATH=`dirname $0`
TEST_TYPE=$1
TEST_NAME=$2

./src/test-$1 --log_level=all --run_test=$2

