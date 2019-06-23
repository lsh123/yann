#
# Targets
#
lib_LTLIBRARIES=libyann.la libyann_word2vec.la libyann_test.la
bin_PROGRAMS=test-funcs test-updaters test-layers test-fcnn test-cnn test-rnn test-custom test-word2vec

AM_CXXFLAGS=\
	@AM_CXXFLAGS@ \
	-Wall -Werror \
	-DBOOST_ENABLE_ASSERT_HANDLER=1 \
	$(NULL)

AM_LDFLAGS= \
	@AM_LDFLAGS@ \
	 -lboost_system -lboost_thread \
	$(NULL)

#
# Extra flags
#
AM_CXXFLAGS_TEST= \
	-DBOOST_TEST_DYN_LINK \
	$(NULL)
AM_LDFLAGS_TEST= \
	-static \
	-lboost_iostreams \
	-lboost_filesystem \
	-lboost_unit_test_framework \
	-lz \
	$(NULL)
AM_LDADD_TEST= \
	libyann.la \
	libyann_word2vec.la \
	libyann_test.la \
	$(NULL)

#
# Main YANN library
#
libyann_la_SOURCES = \
	core/types.cpp \
	core/utils.cpp \
	core/random.cpp \
	core/layer.cpp \
	core/functions.cpp \
	core/updaters.cpp \
	core/training.cpp \
	core/nn.cpp \
	layers/contlayer.cpp \
	layers/fclayer.cpp \
	layers/polllayer.cpp \
	layers/convlayer.cpp \
	layers/smaxlayer.cpp \
	layers/reclayer.cpp \
	layers/lstmlayer.cpp \
	networks/fcnn.cpp \
	networks/cnn.cpp \
	$(NULL)
libyann_la_CXXFLAGS=$(AM_CXXFLAGS)
libyann_la_LDFLAGS=$(AM_LDFLAGS)

#
# Word2Vec implementation with YANN library
#
libyann_word2vec_la_SOURCES = \
	word2vec/text_parser.cpp \
	word2vec/word2vec.cpp \
	$(NULL)
libyann_word2vec_la_CXXFLAGS=$(AM_CXXFLAGS)
libyann_word2vec_la_LDFLAGS=$(AM_LDFLAGS)
libyann_word2vec_la_LIBADD= \
	libyann.la \
	$(NULL)
	
#
# Testing utils for YANN library
#
libyann_test_la_SOURCES = \
	test/test_utils.cpp \
	test/timer.cpp \
	test/mnist_test.cpp \
	$(NULL)
libyann_test_la_CXXFLAGS=$(AM_CXXFLAGS)
libyann_test_la_LDFLAGS=$(AM_LDFLAGS)
libyann_test_la_LIBADD= \
	libyann.la \
	$(NULL)

#
# Functions unit test program
#
test_funcs_SOURCES=\
	test/test_functions.cpp \
	test/test_types.cpp \
	$(NULL)
test_funcs_CXXFLAGS= $(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_funcs_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_funcs_LDADD=$(AM_LDADD_TEST)

#
# Updaters unit test program
#
test_updaters_SOURCES= \
	test/test_updaters.cpp \
	$(NULL)
test_updaters_CXXFLAGS=$(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_updaters_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_updaters_LDADD=$(AM_LDADD_TEST)

#
# Layers unit test program
#
test_layers_SOURCES= \
	test/test_layers.cpp \
	test/test_fclayer.cpp \
	test/test_contlayer.cpp \
	test/test_polllayer.cpp \
	test/test_convlayer.cpp \
	test/test_smaxlayer.cpp \
	test/test_reclayer.cpp \
	test/test_lstmlayer.cpp \
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
# RNN unit test program
#
test_rnn_SOURCES=test/test_rnn.cpp
test_rnn_CXXFLAGS=$(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_rnn_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_rnn_LDADD=$(AM_LDADD_TEST)

#
# Custom NN tests (not included in the "check"!)
#
test_custom_SOURCES=test/test_custom_nn.cpp
test_custom_CXXFLAGS=$(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_custom_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_custom_LDADD=$(AM_LDADD_TEST)

#
# Word2Vec tests
#
test_word2vec_SOURCES=test/test_word2vec.cpp
test_word2vec_CXXFLAGS=$(AM_CXXFLAGS) $(AM_CXXFLAGS_TEST)
test_word2vec_LDFLAGS=$(AM_LDFLAGS) $(AM_LDFLAGS_TEST)
test_word2vec_LDADD=$(AM_LDADD_TEST)

#
# Extra rules
#
check-%: test-%
	./$< --log_level=all --show_progress=no

check: check-funcs check-updaters check-layers check-fcnn check-cnn check-rnn check-word2vec

	
