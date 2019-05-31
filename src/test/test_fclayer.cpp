//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/fclayer.h"
#include "training.h"
#include "utils.h"
#include "functions.h"

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
  FullyConnectedLayer one(input_size, output_size);
  one.init(InitMode_Random_01);

  BOOST_TEST_MESSAGE("FullyConnectedLayer before writing to file: " << "\n" << one);
  ostringstream oss;
  oss << one;
  BOOST_CHECK(!oss.fail());

  FullyConnectedLayer two(input_size, output_size);
  std::istringstream iss(oss.str());
  iss >> two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("FullyConnectedLayer after loading from file: " << "\n" << two);

  BOOST_CHECK(one.is_equal(two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer FeedForward test ...");

  const MatrixSize input_size = 3;
  const MatrixSize output_size = 2;
  const MatrixSize batch_size = 2;

  Matrix ww(input_size, output_size);
  Vector bb(output_size);
  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  ww << 0.1, 0.4,
        0.2, 0.5,
        0.3, 0.6;

  bb << 0.003, 0.007;

  input << 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6;

  FullyConnectedLayer layer(input_size, output_size);
  layer.set_values(ww, bb);


  // Test writing output to the internal buffer
  {
    std::unique_ptr<Layer::Context> ctx = layer.create_context(batch_size);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      // identity
      layer.set_activation_function(make_unique<IdentityFunction>());
      expected << 0.143, 0.327,
                  0.323, 0.777;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));

      // sigmoid
      layer.set_activation_function(make_unique<SigmoidFunction>());
      expected << 0.53569, 0.58103,
                  0.58006, 0.68503;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
    }
  }

  // Test writing output to an external buffer
  {
    VectorBatch output;
    resize_batch(output, batch_size, output_size);
    std::unique_ptr<Layer::Context> ctx = layer.create_context(output);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      // identity
      layer.set_activation_function(make_unique<IdentityFunction>());
      expected << 0.143, 0.327,
                  0.323, 0.777;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

      // sigmoid
      layer.set_activation_function(make_unique<SigmoidFunction>());
      expected << 0.53569, 0.58103,
                  0.58006, 0.68503;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
    }
  }

}

BOOST_AUTO_TEST_CASE(Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedLayer backprop test ...");

  const MatrixSize input_size = 3;
  const MatrixSize output_size = 2;
  const MatrixSize batch_size = 2;

  Matrix ww(input_size, output_size);
  Vector bb(output_size);
  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  ww << 1, 4,
        2, 5,
        3, 6;

  bb << 0.3, 0.7;

  input          << 1, 2, 3,
                    4, 5, 6;

  expected       << 14.3, 32.7,
                    32.3, 77.7;

  FullyConnectedLayer layer(input_size, output_size);
  layer.set_activation_function(make_unique<IdentityFunction>());
  layer.set_values(ww, bb);

  auto ctx = layer.create_training_context(batch_size, make_unique<Updater_GradientDescent>());
  ctx->reset_state();

  VectorBatch gradient_input, gradient_output;
  resize_batch(gradient_input, batch_size, input_size);
  resize_batch(gradient_output, batch_size, output_size);

  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;

    // feed forward
    layer.feedforward(input, ctx.get());

    // backprop
    gradient_output = ctx->get_output() - expected;
    layer.backprop(gradient_output, input, optional<RefVectorBatch>(gradient_input), ctx.get());

    // update input
    input -= gradient_input;
    layer.feedforward(input, ctx.get());
    BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
  }
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
  layer->init(InitMode_Zeros);

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
  layer->init(InitMode_Zeros);

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

BOOST_AUTO_TEST_SUITE_END()

