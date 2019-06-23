//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/smaxlayer.h"
#include "core/functions.h"
#include "core/utils.h"

#include "timer.h"
#include "test_utils.h"
#include "test_layers.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

struct SoftmaxLayerTestFixture
{
  SoftmaxLayerTestFixture()
  {

  }
  ~SoftmaxLayerTestFixture()
  {

  }
};
// struct SoftmaxLayerTestFixture

BOOST_FIXTURE_TEST_SUITE(SoftmaxLayerTest, SoftmaxLayerTestFixture);

BOOST_AUTO_TEST_CASE(SoftmaxLayer_IO_Test)
{
  BOOST_TEST_MESSAGE("*** SoftmaxLayer IO test ...");

  const MatrixSize size = 7;
  SoftmaxLayer one(size);
  one.init(Layer::InitMode_Random, boost::none);

  BOOST_TEST_MESSAGE("SoftmaxLayer before writing to file: " << "\n" << one);
  ostringstream oss;
  oss << one;
  BOOST_CHECK(!oss.fail());

  SoftmaxLayer two(size);
  std::istringstream iss(oss.str());
  iss >> two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("SoftmaxLayer after loading from file: " << "\n" << two);

  BOOST_CHECK(one.is_equal(two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(SoftmaxLayer_FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** SoftmaxLayer FeedForward test ...");

  const MatrixSize size = 3;
  const MatrixSize batch_size = 2;

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, size);
  resize_batch(expected_output, batch_size, size);

  input <<
      10, 10, 10,
      //////////
      10, 2,  6;
  expected_output <<
      0.33333, 0.33333, 0.33333,
      /////////////////////////
      0.98169, 0.00033, 0.01798;

  auto layer = make_unique<SoftmaxLayer>(size);
  BOOST_CHECK(layer);
  test_layer_feedforward(*layer, input, expected_output);
}

BOOST_AUTO_TEST_CASE(SoftmaxLayer_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** SoftmaxLayer Backprop test ...");

  const MatrixSize size = 3;
  const MatrixSize batch_size = 2;
  const double learning_rate = 1.0;
  const size_t epochs = 1000;

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, size);
  resize_batch(expected_output, batch_size, size);

  input <<
      10, 10, 10,
      //////////
      10, 2,  6;

  expected_output <<
      0.8, 0.1, 0.1,
      /////////////
      0.2, 0.3, 0.5;

  auto layer = make_unique<SoftmaxLayer>(size);
  BOOST_CHECK(layer);
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
  BOOST_TEST_MESSAGE("*** SoftmaxLayer backprop with random inputs and weights ...");

  const MatrixSize size = 10;
  const MatrixSize batch_size = 5;
  const double learning_rate = 1.0;
  const size_t epochs = 20000;

  auto layer = make_unique<SoftmaxLayer>(size);
  BOOST_CHECK(layer);
  layer->init(Layer::InitMode_Random, Layer::InitContext(123));

  test_layer_backprop_from_random(
      *layer,
      batch_size,
      make_unique<QuadraticCost>(),
      learning_rate,
      epochs
  );
}

BOOST_AUTO_TEST_SUITE_END()

