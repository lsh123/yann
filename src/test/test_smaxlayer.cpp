//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/smaxlayer.h"
#include "training.h"
#include "utils.h"

#include "timer.h"
#include "test_utils.h"

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
  one.init(Layer::InitMode_Random);

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

  VectorBatch input, expected;
  resize_batch(input, batch_size, size);
  resize_batch(expected, batch_size, size);

  input << 10, 10, 10,
           //////////
           10, 2,  6;
  expected << 0.33333, 0.33333, 0.33333,
              /////////////////////////
              0.98169, 0.00033, 0.01798;

  SoftmaxLayer layer(size);

  // Test writing output to the internal buffer
  {
    std::unique_ptr<Layer::Context> ctx = layer.create_context(batch_size);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
    }
  }

  // Test writing output to an external buffer
  {
    VectorBatch output;
    resize_batch(output, batch_size, size);
    std::unique_ptr<Layer::Context> ctx = layer.create_context(output);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
    }
  }
}

BOOST_AUTO_TEST_CASE(SoftmaxLayer_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** SoftmaxLayer Backprop test ...");

  const MatrixSize size = 3;
  const MatrixSize batch_size = 2;
  const double learning_rate = 10.0;
  const size_t epochs = 150;

  VectorBatch input, expected;
  resize_batch(input, batch_size, size);
  resize_batch(expected, batch_size, size);

  input << 10, 10, 10,
           //////////
           10, 2,  6;

  expected << 0.8, 0.1, 0.1,
              /////////////
              0.2, 0.3, 0.5;


  SoftmaxLayer layer(size);

  auto ctx = layer.create_training_context(batch_size, make_unique<Updater_GradientDescent>());
  ctx->reset_state();

  VectorBatch gradient_input, gradient_output;
  resize_batch(gradient_input, batch_size, size);
  resize_batch(gradient_output, batch_size, size);

  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;

    for(size_t ii = 0; ii < epochs; ++ii) {
      // feed forward
      layer.feedforward(input, ctx.get());

      // backprop
      gradient_output = ctx->get_output() - expected;
      layer.backprop(gradient_output, input, optional<RefVectorBatch>(gradient_input), ctx.get());

      // update input
      input -= learning_rate * gradient_input;
    }
    BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
  }
}
BOOST_AUTO_TEST_SUITE_END()

