//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/contlayer.h"
#include "core/training.h"
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
  one->init(Layer::InitMode_Random, boost::none);

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

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(expected_output, batch_size, layer->get_output_size());

  input <<
      1, 2, 3, 4,
      //////////
      5, 6, 7, 8;
  expected_output <<
      2.5, 2.5,
      /////////
      6.5, 6.5;

  test_layer_feedforward(*layer, input, expected_output);
}

BOOST_AUTO_TEST_CASE(BroadcastLayer_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** BroadcastLayer Backprop test ...");

  const MatrixSize input_size = 4;
  const MatrixSize output_frames_num = 2;
  const MatrixSize batch_size = 2;
  const double learning_rate = 0.1;
  const size_t epochs = 100;

  auto layer = create_bcast_layer(input_size, output_frames_num);
  YANN_CHECK_EQ(layer->get_input_size(), input_size); // one input frame
  YANN_CHECK_EQ(layer->get_output_size(), 1 * output_frames_num); // 1 output per AvgLayer

  VectorBatch input, input_expected, expected_output;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(input_expected, batch_size, layer->get_input_size());
  resize_batch(expected_output, batch_size, layer->get_output_size());

  input <<
      1, 2, 3, 4,
      //////////
      5, 6, 7, 8;

  expected_output <<
      3.5, 3.5,
      /////////
      8.5, 8.5;

  input_expected <<
      2, 3, 4, 5,
      //////////
      7, 8, 9, 10;

  test_layer_backprop(
      *layer,
      input,
      make_optional<RefConstVectorBatch>(input_expected),
      expected_output,
      make_unique<QuadraticCost>(),
      learning_rate,
      epochs
  );
}

BOOST_AUTO_TEST_CASE(ParallelLayer_IO_Test)
{
  BOOST_TEST_MESSAGE("*** ParallelLayer IO test ...");

  const MatrixSize input_size = 2;
  const MatrixSize frames_num = 3;
  auto one = create_parallel_layer(input_size, frames_num);
  one->init(Layer::InitMode_Random, boost::none);

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

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(expected_output, batch_size, layer->get_output_size());

  input <<
      1,  2,  3,  4,
      5,  6,  7,  8,
      //////////////
      9, 10, 11, 12,
      13, 14, 15, 16;
  expected_output <<
      2.5,  6.5,
      ///////////
      10.5, 14.5;

  test_layer_feedforward(*layer, input, expected_output);
}

BOOST_AUTO_TEST_CASE(ParallelLayer_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** ParallelLayer Backprop test ...");

  const MatrixSize input_size = 4;
  const MatrixSize frames_num = 2;
  const MatrixSize batch_size = 2;
  const double learning_rate = 0.75;
  const size_t epochs = 10;

  auto layer = create_parallel_layer(input_size, frames_num);
  YANN_CHECK_EQ(layer->get_input_size(),  input_size * frames_num); // 1 input per AvgLayer
  YANN_CHECK_EQ(layer->get_output_size(), 1 * frames_num); // 1 output per AvgLayer

  VectorBatch input, input_expected, expected_output;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(input_expected, batch_size, layer->get_input_size());
  resize_batch(expected_output, batch_size, layer->get_output_size());

  input <<
      1,  2,  3,  4,
      5,  6,  7,  8,
      //////////////
      9, 10, 11, 12,
     13, 14, 15, 16;

  expected_output <<
      3.5,  8.5,
      ///////////
      4.5, 14.5;

  input_expected <<
      2,  3,  4,   5,
      7,  8,  9,  10,
      //////////////
       3,  4,  5,  6,
      13, 14, 15, 16;

  test_layer_backprop(
      *layer,
      input,
      make_optional<RefConstVectorBatch>(input_expected),
      expected_output,
      make_unique<QuadraticCost>(),
      learning_rate,
      epochs
  );
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

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(expected_output, batch_size, layer->get_output_size());

  input <<
      1,  2,  3,  4,
      5,  6,  7,  8,
      //////////////
      9, 10, 11, 12,
     13, 14, 15, 16;

  expected_output <<
      9,
      ///
      25;

  test_layer_feedforward(*layer, input, expected_output);
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

  VectorBatch input, input_expected, expected_output;
  resize_batch(input, batch_size, layer->get_input_size());
  resize_batch(input_expected, batch_size, layer->get_input_size());
  resize_batch(expected_output, batch_size, layer->get_output_size());

  input <<
      1,  2,  3,  4,
      5,  6,  7,  8,
      //////////////
      9, 10, 11, 12,
     13, 14, 15, 16;

  expected_output <<
      17,    // + 8
      ///
      13;    // -12

  input_expected <<
      5,  6,  7,  8,  // +4
      9, 10, 11, 12,  // +4
      //////////////
      3,  4,  5,  6,  // -6
      7,  8,  9, 10;  // -6

  test_layer_backprop(
      *layer,
      input,
      make_optional<RefConstVectorBatch>(input_expected),
      expected_output,
      make_unique<QuadraticCost>(),
      learning_rate,
      epochs
  );
}

// TODO: add test for sequential container
// TODO: add test for mapping container


BOOST_AUTO_TEST_SUITE_END()

