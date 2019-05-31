//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/contlayer.h"
#include "training.h"
#include "utils.h"

#include "timer.h"
#include "test_utils.h"
#include "test_layers.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

struct ContainerLayerTestFixture
{
  ContainerLayerTestFixture()
  {

  }
  ~ContainerLayerTestFixture()
  {

  }
  unique_ptr<BroadcastLayer> create_bcast_layer(
      const MatrixSize & input_size,
      const size_t & output_frames_num)
  {
    auto res = make_unique<BroadcastLayer>();
    for(auto ii = output_frames_num; ii > 0; --ii) {
      res->append_layer(make_unique<AvgLayer>(input_size));
    }
    return res;
  }
  unique_ptr<ParallelLayer> create_parallel_layer(
      const MatrixSize & input_size,
      const size_t & frames_num)
  {
    auto res = make_unique<ParallelLayer>(frames_num);
    for(auto ii = frames_num; ii > 0; --ii) {
      res->append_layer(make_unique<AvgLayer>(input_size));
    }
    return res;
  }
};
// struct ContainerLayerTestFixture

BOOST_FIXTURE_TEST_SUITE(ContainerLayerTest, ContainerLayerTestFixture);

BOOST_AUTO_TEST_CASE(BroadcastLayer_IO_Test)
{
  BOOST_TEST_MESSAGE("*** BroadcastLayer IO test ...");

  const MatrixSize input_size = 2;
  const MatrixSize output_frames_num = 4;
  auto one = create_bcast_layer(input_size, output_frames_num);
  one->init(InitMode_Random_01);

  BOOST_TEST_MESSAGE("BroadcastLayer before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << (*one);
  BOOST_CHECK(!oss.fail());

  auto two = create_bcast_layer(input_size, output_frames_num);
  std::istringstream iss(oss.str());
  iss >> (*two);
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("BroadcastLayer after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(BroadcastLayer_FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** BroadcastLayer FeedForward test ...");

  const MatrixSize input_size = 4;
  const MatrixSize output_frames_num = 2;
  const MatrixSize batch_size = 2;

  auto layer = create_bcast_layer(input_size, output_frames_num);
  YANN_CHECK_EQ(layer->get_input_size(), input_size); // one input frame
  YANN_CHECK_EQ(layer->get_output_size(), 1 * output_frames_num); // 1 output per AvgLayer

  VectorBatch input, expected;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(expected, batch_size, layer->get_output_size());

  input << 1, 2, 3, 4,
           //////////
           5, 6, 7, 8;
  expected << 2.5, 2.5,
              /////////
              6.5, 6.5;

  // Test writing output to the internal buffer
  {
    auto ctx = layer->create_context(batch_size);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer->feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
    }
  }

  // Test writing output to an external buffer
  {
    VectorBatch output;
    resize_batch(output, batch_size, layer->get_output_size());
    auto ctx = layer->create_context(output);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer->feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
    }
  }
}

BOOST_AUTO_TEST_CASE(BroadcastLayer_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** BroadcastLayer Backprop test ...");

  const MatrixSize input_size = 4;
  const MatrixSize output_frames_num = 2;
  const MatrixSize batch_size = 2;
  const double learning_rate = 0.75;
  const size_t epochs = 10;

  auto layer = create_bcast_layer(input_size, output_frames_num);
  YANN_CHECK_EQ(layer->get_input_size(), input_size); // one input frame
  YANN_CHECK_EQ(layer->get_output_size(), 1 * output_frames_num); // 1 output per AvgLayer

  VectorBatch input, input_expected, expected;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(input_expected, batch_size, layer->get_input_size());
  resize_batch(expected, batch_size, layer->get_output_size());

  input << 1, 2, 3, 4,
           //////////
           5, 6, 7, 8;

  expected << 3.5, 3.5,
              /////////
              8.5, 8.5;

  input_expected << 2, 3, 4, 5,
                    //////////
                    7, 8, 9, 10;

  auto ctx = layer->create_training_context(
      batch_size,make_unique<Updater_GradientDescent>());
  ctx->reset_state();

  VectorBatch gradient_input, gradient_output;
  resize_batch(gradient_input, batch_size, layer->get_input_size());
  resize_batch(gradient_output, batch_size, layer->get_output_size());

  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;

    for(size_t ii = 0; ii < epochs; ++ii) {
      // feed forward
      layer->feedforward(input, ctx.get());

      // backprop
      gradient_output = ctx->get_output() - expected;
      layer->backprop(gradient_output, input, optional<RefVectorBatch>(gradient_input), ctx.get());

      // update input
      input -= learning_rate * gradient_input;
    }
    BOOST_CHECK(input_expected.isApprox(input, TEST_TOLERANCE));
    BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
  }
}

BOOST_AUTO_TEST_CASE(ParallelLayer_IO_Test)
{
  BOOST_TEST_MESSAGE("*** ParallelLayer IO test ...");

  const MatrixSize input_size = 2;
  const MatrixSize frames_num = 3;
  auto one = create_parallel_layer(input_size, frames_num);
  one->init(InitMode_Random_01);

  BOOST_TEST_MESSAGE("ParallelLayer before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << (*one);
  BOOST_CHECK(!oss.fail());

  auto two = create_parallel_layer(input_size, frames_num);
  std::istringstream iss(oss.str());
  iss >> (*two);
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("ParallelLayer after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}


BOOST_AUTO_TEST_CASE(ParallelLayer_FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** ParallelLayer FeedForward test ...");

  const MatrixSize input_size = 4;
  const MatrixSize frames_num = 2;
  const MatrixSize batch_size = 2;

  auto layer = create_parallel_layer(input_size, frames_num);
  YANN_CHECK_EQ(layer->get_input_size(), input_size * frames_num); // 1 output per AvgLayer
  YANN_CHECK_EQ(layer->get_output_size(), 1 * frames_num); // 1 output per AvgLayer

  VectorBatch input, expected;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(expected, batch_size, layer->get_output_size());

  input <<  1,  2,  3,  4,
            5,  6,  7,  8,
            //////////////
            9, 10, 11, 12,
           13, 14, 15, 16;
  expected <<  2.5,  6.5,
              ///////////
              10.5, 14.5;

  // Test writing output to the internal buffer
  {
    auto ctx = layer->create_context(batch_size);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer->feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
    }
  }

  // Test writing output to an external buffer
  {
    VectorBatch output;
    resize_batch(output, batch_size, layer->get_output_size());
    auto ctx = layer->create_context(output);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer->feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
    }
  }
}

BOOST_AUTO_TEST_CASE(ParallelLayer_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** ParallelLayer Backprop test ...");

  const MatrixSize input_size = 4;
  const MatrixSize frames_num = 2;
  const MatrixSize batch_size = 2;

  auto layer = create_parallel_layer(input_size, frames_num);
  YANN_CHECK_EQ(layer->get_input_size(),  input_size * frames_num); // 1 input per AvgLayer
  YANN_CHECK_EQ(layer->get_output_size(), 1 * frames_num); // 1 output per AvgLayer

  VectorBatch input, input_expected, expected;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(input_expected, batch_size, layer->get_input_size());
  resize_batch(expected, batch_size, layer->get_output_size());

  input <<  1,  2,  3,  4,
            5,  6,  7,  8,
            //////////////
            9, 10, 11, 12,
           13, 14, 15, 16;

  expected <<  3.5,  8.5,
              ///////////
               4.5, 14.5;

  input_expected << 2,  3,  4,   5,
                    7,  8,  9,  10,
                    //////////////
                     3,  4,  5,  6,
                    13, 14, 15, 16;

  auto ctx = layer->create_training_context(
      batch_size, make_unique<Updater_GradientDescent>());
  ctx->reset_state();

  VectorBatch gradient_input, gradient_output;
  resize_batch(gradient_input, batch_size, layer->get_input_size());
  resize_batch(gradient_output, batch_size, layer->get_output_size());

  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;

    // feed forward
    layer->feedforward(input, ctx.get());

    // backprop
    gradient_output = ctx->get_output() - expected;
    layer->backprop(gradient_output, input, optional<RefVectorBatch>(gradient_input), ctx.get());

    // update input
    input -= gradient_input;
    BOOST_CHECK(input_expected.isApprox(input, TEST_TOLERANCE));

    // and calculate output one more time
    layer->feedforward(input, ctx.get());
    BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
  }
}


BOOST_AUTO_TEST_CASE(MergeLayer_FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** MergeLayer FeedForward test ...");

  const MatrixSize input_frames_num = 2;
  const MatrixSize input_size = 4;
  const MatrixSize batch_size = 2;

  auto layer = make_unique<MergeLayer>(input_frames_num);
  YANN_CHECK(layer);
  layer->append_layer(make_unique<AvgLayer>(input_size));
  YANN_CHECK_EQ(layer->get_input_size(), input_size * input_frames_num); //
  YANN_CHECK_EQ(layer->get_output_size(), 1 * 1); // 1 output for AvgLayer and for MergeLayer

  VectorBatch input, expected;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(expected, batch_size, layer->get_output_size());

  input <<  1,  2,  3,  4,
            5,  6,  7,  8,
            //////////////
            9, 10, 11, 12,
           13, 14, 15, 16;
  expected <<  9,
              ///
              25;

  // Test writing output to the internal buffer
  {
    std::unique_ptr<Layer::Context> ctx = layer->create_context(batch_size);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer->feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
    }
  }

  // Test writing output to an external buffer
  {
    VectorBatch output;
    resize_batch(output, batch_size, layer->get_output_size());
    std::unique_ptr<Layer::Context> ctx = layer->create_context(output);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer->feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
    }
  }
}

BOOST_AUTO_TEST_CASE(MergeLayer_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** MergeLayer Backprop test ...");

  const MatrixSize input_frames_num = 2;
  const MatrixSize input_size = 4;
  const MatrixSize batch_size = 2;
  const double learning_rate = 0.75;
  const size_t epochs = 10;

  auto layer = make_unique<MergeLayer>(input_frames_num);
  YANN_CHECK(layer);
  layer->append_layer(make_unique<AvgLayer>(input_size));
  YANN_CHECK_EQ(layer->get_input_size(), input_size * input_frames_num); //
  YANN_CHECK_EQ(layer->get_output_size(), 1 * 1); // 1 output for AvgLayer and for MergeLayer

  VectorBatch input, input_expected, expected;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(input_expected, batch_size, layer->get_input_size());
  resize_batch(expected, batch_size, layer->get_output_size());

  input <<
      1,  2,  3,  4,
      5,  6,  7,  8,
      //////////////
      9, 10, 11, 12,
     13, 14, 15, 16;

  expected <<
      17,    // + 8
      ///
      13;    // -12

  input_expected <<
      5,  6,  7,  8,  // +4
      9, 10, 11, 12,  // +4
      //////////////
      3,  4,  5,  6,  // -6
      7,  8,  9, 10;  // -6


  auto ctx = layer->create_training_context(
      batch_size, make_unique<Updater_GradientDescent>());
  ctx->reset_state();

  VectorBatch gradient_input, gradient_output;
  resize_batch(gradient_input, batch_size, layer->get_input_size());
  resize_batch(gradient_output, batch_size, layer->get_output_size());

  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;

    for(size_t ii = 0; ii < epochs; ++ii) {
      // feed forward
      layer->feedforward(input, ctx.get());

      // backprop
      gradient_output = ctx->get_output() - expected;
      layer->backprop(gradient_output, input, optional<RefVectorBatch>(gradient_input), ctx.get());

      // update input
      input -= learning_rate * gradient_input;
    }
    BOOST_CHECK(input_expected.isApprox(input, TEST_TOLERANCE));
    BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
  }
}

// TODO: add test for sequential container
// TODO: add test for mapping container


BOOST_AUTO_TEST_SUITE_END()

