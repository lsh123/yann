//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/polllayer.h"
#include "nntraining.h"
#include "utils.h"

#include "timer.h"
#include "test_utils.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

struct PollingLayerTestFixture
{
  PollingLayerTestFixture()
  {

  }
  ~PollingLayerTestFixture()
  {

  }
};
// struct PollingLayerTestFixture

BOOST_FIXTURE_TEST_SUITE(PollingLayerTest, PollingLayerTestFixture);

BOOST_AUTO_TEST_CASE(PollingLayer_IO_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer IO test ...");

  const MatrixSize input_cols = 7;
  const MatrixSize input_rows = 3;
  const MatrixSize filter_size = 2;
  PollingLayer one(input_cols, input_rows, filter_size, PollingLayer::PollMode_Max);
  one.init(InitMode_Random_01);

  BOOST_TEST_MESSAGE("PollingLayer before writing to file: " << "\n" << one);
  ostringstream oss;
  oss << one;
  BOOST_CHECK(!oss.fail());

  PollingLayer two(input_cols, input_rows, filter_size, PollingLayer::PollMode_Max);
  std::istringstream iss(oss.str());
  iss >> two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("PollingLayer after loading from file: " << "\n" << two);

  BOOST_CHECK(one.is_equal(two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(PollingLayer_Max_FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Max FeedForward test ...");

  const MatrixSize input_cols = 6;
  const MatrixSize input_rows = 3;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input << 1, 2, 11, 10, 13, 18,
           3, 4, 12,  9, 14, 17,
           5, 6,  7,  8, 15, 16,
           /////////////////////
           2, 6,  8, 10, 13, 14,
           3, 2,  7,  9, 16, 15,
           4, 5, 13, 11, 16, 18;
  expected << 4, 12, 18,
              /////////
              6, 10, 16;

  PollingLayer layer(input_rows, input_cols, filter_size, PollingLayer::PollMode_Max);

  // Test writing output to the internal buffer
  {
    std::unique_ptr<Layer::Context> ctx = layer.create_context(batch_size);
    BOOST_VERIFY (ctx);
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
    resize_batch(output, batch_size, output_size);
    std::unique_ptr<Layer::Context> ctx = layer.create_context(output);
    BOOST_VERIFY (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
    }
  }
}

BOOST_AUTO_TEST_CASE(PollingLayer_Max_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Max Backprop test ...");

  const MatrixSize input_cols = 6;
  const MatrixSize input_rows = 3;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;

  VectorBatch input, input_expected, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(input_expected, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input << 1, 2, 11, 10, 13, 18,
           3, 4, 12,  9, 14, 17,
           5, 6,  7,  8, 15, 16,
           /////////////////////
           2, 6,  8, 10, 13, 14,
           3, 2,  7,  9, 16, 15,
           4, 5, 13, 11, 16, 18;

  expected << 14, 22, 28,
              /////////
              16, 20, 26;

  input_expected << 1,  2, 11, 10, 13, 28,
                    3, 14, 22,  9, 14, 17,
                    5,  6,  7,  8, 15, 16,
                    /////////////////////
                    2, 16,  8, 20, 13, 14,
                    3,  2,  7,  9, 26, 15,
                    4,  5, 13, 11, 16, 18;

  PollingLayer layer(input_rows, input_cols, filter_size, PollingLayer::PollMode_Max);

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
    BOOST_CHECK(input_expected.isApprox(input, TEST_TOLERANCE));
    layer.feedforward(input, ctx.get());
    BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
  }
}

BOOST_AUTO_TEST_CASE(PollingLayer_Avg_FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Avg FeedForward test ...");

  const MatrixSize input_cols = 6;
  const MatrixSize input_rows = 3;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input << 1, 2, 11, 10, 13, 18,
           3, 4, 12,  9, 14, 17,
           5, 6,  7,  8, 15, 16,
           /////////////////////
           2, 6,  8, 10, 13, 14,
           3, 2,  7,  9, 16, 15,
           4, 5, 13, 11, 16, 18;
  expected <<  2.5, 10.5, 15.5,
              /////////////////
               3.25, 8.5, 14.5;

  PollingLayer layer(input_rows, input_cols, filter_size, PollingLayer::PollMode_Avg);

  // Test writing output to the internal buffer
  {
    std::unique_ptr<Layer::Context> ctx = layer.create_context(batch_size);
    BOOST_VERIFY (ctx);
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
    resize_batch(output, batch_size, output_size);
    std::unique_ptr<Layer::Context> ctx = layer.create_context(output);
    BOOST_VERIFY (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
    }
  }
}

BOOST_AUTO_TEST_CASE(PollingLayer_Avg_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Avg Backprop test ...");

  const MatrixSize input_cols = 6;
  const MatrixSize input_rows = 3;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;

  VectorBatch input, input_expected, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(input_expected, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input << 1, 2, 11, 10, 13, 18,
           3, 4, 12,  9, 14, 17,
           5, 6,  7,  8, 15, 16,
           /////////////////////
           2, 6,  8, 10, 13, 14,
           3, 2,  7,  9, 16, 15,
           4, 5, 13, 11, 16, 18;

  expected << 3.5, 9.5, 13.5,     // +1, -1, -2
              /////////////////
              1.25, 8.5, 16.5;    // -2,  0, +2

  input_expected << 2, 3, 10,  9, 11, 16,
                    4, 5, 11,  8, 12, 15,
                    5, 6,  7,  8, 15, 16,
                    /////////////////////
                    0, 4,  8, 10, 15, 16,
                    1, 0,  7,  9, 18, 17,
                    4, 5, 13, 11, 16, 18;

  PollingLayer layer(input_rows, input_cols, filter_size, PollingLayer::PollMode_Avg);

  auto ctx = layer.create_training_context(
      batch_size, make_unique<Updater_GradientDescent>());
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
    BOOST_CHECK(input_expected.isApprox(input, TEST_TOLERANCE));
    layer.feedforward(input, ctx.get());
    BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
  }
}

BOOST_AUTO_TEST_SUITE_END()

