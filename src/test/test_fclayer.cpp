//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/fclayer.h"
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


struct FullyConnectedLayerTestFixture
{
  FullyConnectedLayerTestFixture()
  {

  }
  ~FullyConnectedLayerTestFixture()
  {

  }
};
// struct FullyConnectedLayerTestFixture

BOOST_FIXTURE_TEST_SUITE(FullyConnectedLayerTest, FullyConnectedLayerTestFixture);

BOOST_AUTO_TEST_CASE(IO_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer IO test ...");

  const MatrixSize input_size = 5;
  const MatrixSize output_size = 3;
  auto one = make_unique<FullyConnectedLayer>(input_size, output_size);
  one->init(Layer::InitMode_Random, boost::none);

  BOOST_TEST_MESSAGE("FullyConnectedLayer before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << *one;
  BOOST_CHECK(!oss.fail());

  auto two = make_unique<FullyConnectedLayer>(input_size, output_size);
  std::istringstream iss(oss.str());
  iss >> *two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("FullyConnectedLayer after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer FeedForward test ...");

  const MatrixSize input_size = 3;
  const MatrixSize output_size = 2;
  const MatrixSize batch_size = 2;

  Matrix ww(input_size, output_size);
  Vector bb(output_size);
  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  ww <<
      0.1, 0.4,
      0.2, 0.5,
      0.3, 0.6;

  bb <<
      0.003, 0.007;

  input <<
      0.1, 0.2, 0.3,
      0.4, 0.5, 0.6;

  auto layer = make_unique<FullyConnectedLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_values(ww, bb);

  // identity
  layer->set_activation_function(make_unique<IdentityFunction>());
  expected_output <<
      0.143, 0.327,
      0.323, 0.777;
  test_layer_feedforward(*layer, input, expected_output);

  // sigmoid
  layer->set_activation_function(make_unique<SigmoidFunction>());
  expected_output <<
      0.53569, 0.58103,
      0.58006, 0.68503;
  test_layer_feedforward(*layer, input, expected_output);
}

BOOST_AUTO_TEST_CASE(Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer backprop test ...");

  const MatrixSize input_size = 3;
  const MatrixSize output_size = 2;
  const MatrixSize batch_size = 2;
  const double learning_rate = 0.01;
  const size_t epochs = 1000;

  Matrix ww(input_size, output_size);
  Vector bb(output_size);
  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  ww <<
      1, 4,
      2, 5,
      3, 6;

  bb <<
      0.3, 0.7;

  input <<
      0, 1, 2,
      3, 4, 5;

  expected_output <<
      14.3, 32.7,
      32.3, 77.7;

  auto layer = make_unique<FullyConnectedLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<IdentityFunction>());
  layer->set_values(ww, bb);

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

BOOST_AUTO_TEST_CASE(Backprop_WithSampling_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer backprop with sampling test ...");

  const MatrixSize input_size = 3;
  const MatrixSize output_size = 2;
  const MatrixSize batch_size = 2;
  const double learning_rate = 0.01;
  const size_t epochs = 1000;
  const double sampling_rate = 0.5;

  Matrix ww(input_size, output_size);
  Vector bb(output_size);
  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  ww <<
      1, 4,
      2, 5,
      3, 6;

  bb <<
      0.3, 0.7;

  input <<
      0, 1, 2,
      3, 4, 5;

  expected_output <<
      14.3, 32.7,
      32.3, 77.7;

  auto layer = make_unique<FullyConnectedLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<IdentityFunction>());
  layer->set_values(ww, bb);
  layer->set_sampling_rate(sampling_rate);

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

BOOST_AUTO_TEST_CASE(RandomBackprop_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer backprop with random inputs and weights ...");

  const MatrixSize input_size  = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size  = 5;
  const double learning_rate = 0.75;
  const size_t epochs = 200;

  auto layer = make_unique<FullyConnectedLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<SigmoidFunction>());
  layer->init(Layer::InitMode_Random, Layer::InitContext(123));

  test_layer_backprop_from_random(
      *layer,
      batch_size,
      make_unique<CrossEntropyCost>(),
      learning_rate,
      epochs
  );
}

BOOST_AUTO_TEST_CASE(Training_WithIdentity_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer with Identity activation training test ...");

  const MatrixSize input_size  = 18;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.01, 0.02, 0.11, 0.10, 0.13, 0.18,
      0.03, 0.04, 0.12, 0.09, 0.14, 0.17,
      0.05, 0.06, 0.07, 0.08, 0.15, 0.16,
      ///////////////////////////////////
      0.02, 0.06, 0.08, 0.10, 0.13, 0.14,
      0.03, 0.02, 0.07, 0.09, 0.16, 0.15,
      0.04, 0.05, 0.13, 0.11, 0.16, 0.18;

  expected <<
      0.6, 0.7, 0.5,
      //////////////
      0.3, 0.8, 0.1;

  // create layer
  auto layer = make_unique<FullyConnectedLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<IdentityFunction>());
  layer->init(Layer::InitMode_Zeros, boost::none);

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<QuadraticCost>(),
      0.75, // learning rate
      2000  // epochs
  );
}


BOOST_AUTO_TEST_CASE(Training_WithIdentityAndSampling_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer with Identity activation and with sampling training test ...");

  const MatrixSize input_size  = 18;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;
  const double sampling_rate = 0.334;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.01, 0.02, 0.11, 0.10, 0.13, 0.18,
      0.03, 0.04, 0.12, 0.09, 0.14, 0.17,
      0.05, 0.06, 0.07, 0.08, 0.15, 0.16,
      ///////////////////////////////////
      0.02, 0.06, 0.08, 0.10, 0.13, 0.14,
      0.03, 0.02, 0.07, 0.09, 0.16, 0.15,
      0.04, 0.05, 0.13, 0.11, 0.16, 0.18;

  expected <<
      0.6, 0.7, 0.5,
      //////////////
      0.3, 0.8, 0.1;

  // create layer
  auto layer = make_unique<FullyConnectedLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<IdentityFunction>());
  layer->set_sampling_rate(sampling_rate);
  layer->init(Layer::InitMode_Zeros, boost::none);

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<QuadraticCost>(),
      0.75, // learning rate
      2000  // epochs
  );
}

BOOST_AUTO_TEST_CASE(Training_WithSigmoid_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer with Sigmoid activation training test ...");

  const MatrixSize input_size  = 18;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.01, 0.02, 0.11, 0.10, 0.13, 0.18,
      0.03, 0.04, 0.12, 0.09, 0.14, 0.17,
      0.05, 0.06, 0.07, 0.08, 0.15, 0.16,
      ///////////////////////////////////
      0.02, 0.06, 0.08, 0.10, 0.13, 0.14,
      0.03, 0.02, 0.07, 0.09, 0.16, 0.15,
      0.04, 0.05, 0.13, 0.11, 0.16, 0.18;

  expected <<
      0.6, 0.7, 0.5,
      //////////////
      0.3, 0.8, 0.1;

  // create layer
  auto layer = make_unique<FullyConnectedLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<SigmoidFunction>());
  layer->init(Layer::InitMode_Zeros, boost::none);

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<CrossEntropyCost>(),
      3.0,  // learning rate
      5000 // epochs
  );
}


BOOST_AUTO_TEST_CASE(Training_WithSigmoidAndSampling_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer with Sigmoid activation with sampling training test ...");

  const MatrixSize input_size  = 18;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;
  const double sampling_rate = 0.334;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.01, 0.02, 0.11, 0.10, 0.13, 0.18,
      0.03, 0.04, 0.12, 0.09, 0.14, 0.17,
      0.05, 0.06, 0.07, 0.08, 0.15, 0.16,
      ///////////////////////////////////
      0.02, 0.06, 0.08, 0.10, 0.13, 0.14,
      0.03, 0.02, 0.07, 0.09, 0.16, 0.15,
      0.04, 0.05, 0.13, 0.11, 0.16, 0.18;

  expected <<
      0.6, 0.7, 0.5,
      //////////////
      0.3, 0.8, 0.1;

  // create layer
  auto layer = make_unique<FullyConnectedLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<SigmoidFunction>());
  layer->init(Layer::InitMode_Zeros, boost::none);
  layer->set_sampling_rate(sampling_rate);

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<CrossEntropyCost>(),
      3.0,  // learning rate
      5000 // epochs
  );
}

BOOST_AUTO_TEST_CASE(RandomTraining_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer training with random inputs ...");

  const MatrixSize input_size  = 18;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 5;
  const double learning_rate = 0.5;
  const size_t epochs = 1000;

  auto layer = make_unique<FullyConnectedLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<SigmoidFunction>());

  test_layer_training_from_random(
      *layer,
      batch_size,
      make_unique<CrossEntropyCost>(),
      learning_rate,
      epochs
  );
}

BOOST_AUTO_TEST_SUITE_END()

