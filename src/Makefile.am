#
# Targets
#
lib_LTLIBRARIES=libyann.la libyann_test.la
bin_PROGRAMS=test-funcs test-layers test-fcnn test-cnn test-custom

AM_CXXFLAGS=\
	-Wall -Werror \
	-DBOOST_ENABLE_ASSERT_HANDLER=1 \
	$(NULL)

AM_LDFLAGS= \
	-lboost_filesystem -lboost_system -lboost_thread \
	$(NULL)

#
# Extra flags
#
AM_CXXFLAGS_TEST= \
	-DBOOST_TEST_DYN_LINK \
	$(NULL)
AM_LDFLAGS_TEST= \
	-static \
	-lboost_unit_test_framework \
	$(NULL)
AM_LDADD_TEST= \
	libyann.la \
	libyann_test.la \
	$(NULL)

#
# Main YANN library
#
libyann_la_SOURCES = \
	types.cpp \
	utils.cpp \
	functions.cpp \
	random.cpp \
	nnlayer.cpp \
	nn.cpp \
	nntraining.cpp \
	layers/contlayer.cpp \
	layers/fclayer.cpp \
	layers/polllayer.cpp \
	layers/convlayer.cpp \
	layers/smaxlayer.cpp \
	fcnn.cpp \
	convnn.cpp \
	$(NULL)
libyann_la_CXXFLAGS=$(AM_CXXFLAGS)
libyann_la_LDFLAGS=$(AM_LDFLAGS)

#
# Testing utils for YANN library
#
libyann_test_la_SOURCES = \
	test/test_utils.cpp \
	test/timer.cpp \
	test/mnist-test.cpp \
	$(NULL)
libyann_test_la_CXXFLAGS=$(AM_CXXFLAGS)
libyann_test_la_LDFLAGS=$(AM_LDFLAGS)

#
# Functions unit test program
#
test_funcs_SOURCES=test/test_functions.cpp
test_funcs_CXXFLAGS= $(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_funcs_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_funcs_LDADD=$(AM_LDADD_TEST)

#
# Layers unit test program
#
test_layers_SOURCES= \
	test/test_smaxlayer.cpp \
	test/test_layers.cpp \
	test/test_contlayer.cpp \
	test/test_polllayer.cpp \
	test/test_convlayer.cpp \
	test/test_fclayer.cpp \
	$(NULL)	
test_layers_CXXFLAGS= $(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_layers_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_layers_LDADD=$(AM_LDADD_TEST)

#
# FCNN unit test program
#
test_fcnn_SOURCES=test/test_fcnn.cpp 
test_fcnn_CXXFLAGS=$(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_fcnn_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_fcnn_LDADD=$(AM_LDADD_TEST)
	
#
# CNN unit test program
#
test_cnn_SOURCES=test/test_cnn.cpp
test_cnn_CXXFLAGS=$(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_cnn_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_cnn_LDADD=$(AM_LDADD_TEST)

#
# Custom NN tests (not included in the "check"!)
#
test_custom_SOURCES=test/test_custom_nn.cpp
test_custom_CXXFLAGS=$(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_custom_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_custom_LDADD=$(AM_LDADD_TEST)

#
# Extra rules
#
check-%: test-%
	./$< --log_level=all --show_progress=no

check: check-funcs check-layers check-fcnn check-cnn 

	
