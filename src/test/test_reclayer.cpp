//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/reclayer.h"
#include "core/training.h"
#include "core/utils.h"
#include "core/functions.h"

#include "timer.h"
#include "test_utils.h"
#include "test_layers.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;


struct RecurrentLayerTestFixture
{
  RecurrentLayerTestFixture()
  {

  }
  ~RecurrentLayerTestFixture()
  {

  }
};
// struct RecurrentLayerTestFixture

BOOST_FIXTURE_TEST_SUITE(RecurrentLayerTest, RecurrentLayerTestFixture);

BOOST_AUTO_TEST_CASE(IO_Test)
{
  BOOST_TEST_MESSAGE("*** RecurrentLayer IO test ...");

  const MatrixSize input_size = 5;
  const MatrixSize state_size = 5;
  const MatrixSize output_size = 3;
  auto one = make_unique<RecurrentLayer>(input_size, state_size, output_size);
  one->init(Layer::InitMode_Random, boost::none);

  BOOST_TEST_MESSAGE("RecurrentLayer before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << *one;
  BOOST_CHECK(!oss.fail());

  auto two = make_unique<RecurrentLayer>(input_size, state_size, output_size);
  std::istringstream iss(oss.str());
  iss >> *two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("RecurrentLayer after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** RecurrentLayer FeedForward test ...");

  const MatrixSize input_size  = 4;
  const MatrixSize state_size  = 3;
  const MatrixSize output_size = 2;
  const MatrixSize batch_size  = 2;

  Matrix ww_hh(state_size, state_size);
  Matrix ww_xh(input_size, state_size);
  Vector bb_h(state_size);
  Matrix ww_ha(state_size, output_size);
  Vector bb_a(output_size);

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  ww_hh <<
      0.2, 0.0, 0.0,
      0.0, 0.2, 0.0,
      0.0, 0.0, 0.2;
  ww_xh <<
      0.3, 0.0, 0.0,
      0.0, 0.3, 0.0,
      0.0, 0.0, 0.3,
      0.0, 0.0, 0.0;
  bb_h <<
      0.4, 0.4, 0.4;
  ww_ha <<
      0.5, 0.0,
      0.0, 0.5,
      0.0, 0.0;
  bb_a <<
      -0.6, 0.6;

  input <<
      0.1, 0.2, 0.3, 0.4,
      0.5, 0.6, 0.7, 0.8;

  auto layer = make_unique<RecurrentLayer>(input_size, state_size, output_size);
  layer->set_values(ww_hh, ww_xh, bb_h, ww_ha, bb_a);

  // identity
  layer->set_activation_functions(
      make_unique<IdentityFunction>(),
      make_unique<IdentityFunction>());
  expected_output <<
      -0.385, 0.830,
      -0.282, 0.936;
  test_layer_feedforward(*layer, input, expected_output);

  // sigmoid
  layer->set_activation_functions(
      make_unique<SigmoidFunction>(),
      make_unique<SigmoidFunction>());
  expected_output <<
      0.4263, 0.7123,
      0.4331, 0.7180;
  test_layer_feedforward(*layer, input, expected_output);
}

BOOST_AUTO_TEST_CASE(Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** RecurrentLayer backprop test ...");

  const MatrixSize input_size  = 4;
  const MatrixSize state_size  = 3;
  const MatrixSize output_size = 2;
  const MatrixSize batch_size  = 2;
  const double learning_rate = 0.5;
  const size_t epochs = 1000;


  Matrix ww_hh(state_size, state_size);
  Matrix ww_xh(input_size, state_size);
  Vector bb_h(state_size);
  Matrix ww_ha(state_size, output_size);
  Vector bb_a(output_size);

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  ww_hh <<
      0.2, 0.0, 0.0,
      0.0, 0.2, 0.0,
      0.0, 0.0, 0.2;
  ww_xh <<
      0.3, 0.0, 0.0,
      0.0, 0.3, 0.0,
      0.0, 0.0, 0.3,
      0.0, 0.0, 0.0;
  bb_h <<
      0.4, 0.4, 0.4;
  ww_ha <<
      0.5, 0.0,
      0.0, 0.5,
      0.0, 0.0;
  bb_a <<
      -0.6, 0.6;

  input <<
      0.5, 0.6, 0.7, 0.8, // expected: 0.1, 0.2, 0.3, 0.4,
      0.1, 0.2, 0.3, 0.4; // expected: 0.5, 0.6, 0.7, 0.8;

  expected_output <<
      -0.385, 0.830,
      -0.282, 0.936;

  auto layer = make_unique<RecurrentLayer>(input_size, state_size, output_size);
  layer->set_values(ww_hh, ww_xh, bb_h, ww_ha, bb_a);
  layer->set_activation_functions(
      make_unique<IdentityFunction>(),
      make_unique<IdentityFunction>());

  test_layer_backprop(
      *layer,
      input,
      boost::none,
      expected_output,
      make_unique<QuadraticCost>(),
      learning_rate,
      epochs
  );
}


BOOST_AUTO_TEST_CASE(Training_WithIdentity_Test)
{
  BOOST_TEST_MESSAGE("*** RecurrentLayer with Identity activation training test ...");

  const MatrixSize input_size  = 6;
  const MatrixSize state_size  = 5;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.01, 0.02, 0.11, 0.10, 0.13, 0.18,
      ///////////////////////////////////
      0.2, 0.6, 0.8, 0.10, 0.2, 0.3;

  expected <<
      0.6, 0.7, 0.5,
      //////////////
      0.3, 0.8, 0.1;

  // create layer
  auto layer = make_unique<RecurrentLayer>(input_size, state_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_functions(
      make_unique<IdentityFunction>(),
      make_unique<IdentityFunction>());
  layer->init(Layer::InitMode_Random, Layer::InitContext(12345));

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<QuadraticCost>(),
      0.01, // learning rate
      300  // epochs
  );
}

BOOST_AUTO_TEST_CASE(Training_WithSigmoid_Test)
{
  BOOST_TEST_MESSAGE("*** RecurrentLayer with Sigmoid activation training test ...");

  const MatrixSize input_size  = 6;
  const MatrixSize state_size  = 5;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.01, 0.02, 0.11, 0.10, 0.13, 0.18,
      ///////////////////////////////////
      0.2, 0.6, 0.8, 0.10, 0.2, 0.3;

  expected <<
      0.6, 0.7, 0.5,
      //////////////
      0.3, 0.8, 0.1;

  // create layer
  auto layer = make_unique<RecurrentLayer>(input_size, state_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_functions(
      make_unique<SigmoidFunction>(),
      make_unique<SigmoidFunction>());
  layer->init(Layer::InitMode_Random, Layer::InitContext(12345));

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<CrossEntropyCost>(),
      0.5,  // learning rate
      1000 // epochs
  );
}


BOOST_AUTO_TEST_SUITE_END()

