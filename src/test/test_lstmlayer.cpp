//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/lstmlayer.h"
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


struct LstmLayerTestFixture
{
  LstmLayerTestFixture()
  {

  }
  ~LstmLayerTestFixture()
  {

  }
};
// struct LstmLayerTestFixture

BOOST_FIXTURE_TEST_SUITE(LstmLayerTest, LstmLayerTestFixture);

BOOST_AUTO_TEST_CASE(IO_Test)
{
  BOOST_TEST_MESSAGE("*** LstmLayer IO test ...");

  const MatrixSize input_size = 5;
  const MatrixSize output_size = 3;
  auto one = make_unique<LstmLayer>(input_size, output_size);
  one->init(Layer::InitMode_Random, boost::none);

  BOOST_TEST_MESSAGE("LstmLayer before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << *one;
  BOOST_CHECK(!oss.fail());

  auto two = make_unique<LstmLayer>(input_size, output_size);
  std::istringstream iss(oss.str());
  iss >> *two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("LstmLayer after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** LstmLayer FeedForward test ...");

  const MatrixSize input_size  = 4;
  const MatrixSize output_size  = 3;
  const MatrixSize batch_size  = 2;

  Matrix ww_x[LstmLayer::Gate_Max];
  Matrix ww_h[LstmLayer::Gate_Max];
  Vector bb[LstmLayer::Gate_Max];

  for(auto ii = 0; ii < LstmLayer::Gate_Max; ++ii) {
    ww_x[ii].resize(input_size, output_size);
    ww_h[ii].resize(output_size, output_size);
    bb[ii].resize(output_size);

    ww_x[ii].setConstant((ii + 1) / (Value)10);
    ww_h[ii].setConstant((ii + 1) / (Value)100);
    bb[ii].setConstant((ii + 1) / (Value)1000);
  }

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  input <<
      0.1, 0.2, 0.3, 0.4,
      0.5, 0.6, 0.7, 0.8;

  auto layer = make_unique<LstmLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_values(ww_x, ww_h, bb);

  // identity
  layer->set_activation_functions(
      make_unique<IdentityFunction>(),
      make_unique<IdentityFunction>());
  // see Training_WithIdentity_Test test
  expected_output <<
      0.0082, 0.0082, 0.0082,
      0.1593, 0.1593, 0.1593;
  test_layer_feedforward(*layer, input, expected_output);

  // sigmoid
  layer->set_activation_functions(
      make_unique<SigmoidFunction>(),
      make_unique<SigmoidFunction>());
  // see Training_WithSigmoid_Test test
  expected_output <<
      0.3429, 0.3429, 0.3429,
      0.4756, 0.4756, 0.4756;
  test_layer_feedforward(*layer, input, expected_output);
}

BOOST_AUTO_TEST_CASE(Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** LstmLayer backprop test ...");

  const MatrixSize input_size  = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size  = 2;
  const double learning_rate = 2.0;
  const size_t epochs = 200;

  Matrix ww_x[LstmLayer::Gate_Max];
  Matrix ww_h[LstmLayer::Gate_Max];
  Vector bb[LstmLayer::Gate_Max];

  for(auto ii = 0; ii < LstmLayer::Gate_Max; ++ii) {
    ww_x[ii].resize(input_size, output_size);
    ww_h[ii].resize(output_size, output_size);
    bb[ii].resize(output_size);

    ww_x[ii].setConstant((ii + 1) / (Value)10);
    ww_h[ii].setConstant((ii + 1) / (Value)100);
    bb[ii].setConstant((ii + 1) / (Value)1000);
  }

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  input <<
      0.5, 0.6, 0.7, 0.8, // expected 0.1, 0.2, 0.3, 0.4,
      0.1, 0.2, 0.3, 0.4; // expected 0.5, 0.6, 0.7, 0.8;

  expected_output <<
      0.0082, 0.0082, 0.0082,
      0.1593, 0.1593, 0.1593;

  auto layer = make_unique<LstmLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_values(ww_x, ww_h, bb);
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

BOOST_AUTO_TEST_CASE(RandomBackprop_Test)
{
  BOOST_TEST_MESSAGE("*** LstmLayer backprop with random inputs and weights ...");

  const MatrixSize input_size  = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size  = 5;
  const double learning_rate = 2.0;
  const size_t epochs = 15000;

  auto layer = make_unique<LstmLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_functions(
      make_unique<SigmoidFunction>(),
      make_unique<TanhFunction>());
  layer->init(Layer::InitMode_Random, Layer::InitContext(123));

  test_layer_backprop_from_random(
      *layer,
      batch_size,
      make_unique<QuadraticCost>(),
      learning_rate,
      epochs
  );
}

BOOST_AUTO_TEST_CASE(Training_WithIdentity_Test)
{
  BOOST_TEST_MESSAGE("*** LstmLayer with Identity activation training test ...");

  const MatrixSize input_size  = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  // see FeedForward_Test test
  input <<
      0.1, 0.2, 0.3, 0.4,
      0.5, 0.6, 0.7, 0.8;
  expected_output <<
      0.0082, 0.0082, 0.0082,
      0.1593, 0.1593, 0.1593;

  // create layer
  auto layer = make_unique<LstmLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_functions(
      make_unique<IdentityFunction>(),
      make_unique<IdentityFunction>());
  layer->init(Layer::InitMode_Random, Layer::InitContext(1234));

  // test
  test_layer_training(
      *layer,
      input,
      expected_output,
      make_unique<QuadraticCost>(),
      0.01, // learning rate
      12000  // epochs
  );
}

BOOST_AUTO_TEST_CASE(Training_WithSigmoid_Test)
{
  BOOST_TEST_MESSAGE("*** LstmLayer with Sigmoid activation training test ...");

  const MatrixSize input_size  = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  // see FeedForward_Test test
  input <<
      0.1, 0.2, 0.3, 0.4,
      0.5, 0.6, 0.7, 0.8;
  expected_output <<
      0.3429, 0.3429, 0.3429,
      0.4756, 0.4756, 0.4756;

  // create layer
  auto layer = make_unique<LstmLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_functions(
      make_unique<SigmoidFunction>(),
      make_unique<SigmoidFunction>());
  layer->init(Layer::InitMode_Random, Layer::InitContext(12345));

  // test
  test_layer_training(
      *layer,
      input,
      expected_output,
      make_unique<CrossEntropyCost>(),
      0.5,  // learning rate
      1000 // epochs
  );
}

BOOST_AUTO_TEST_CASE(RandomTraining_Test)
{
  BOOST_TEST_MESSAGE("*** LstmLayer training with random inputs ...");

  const MatrixSize input_size  = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size  = 5;
  const double learning_rate = 5.0;
  const size_t epochs = 10000;

  auto layer = make_unique<LstmLayer>(input_size, output_size);
  BOOST_CHECK(layer);
  layer->set_activation_functions(
      make_unique<SigmoidFunction>(),
      make_unique<TanhFunction>());

  test_layer_training_from_random(
      *layer,
      batch_size,
      make_unique<QuadraticCost>(),
      learning_rate,
      epochs
  );
}


BOOST_AUTO_TEST_SUITE_END()

