//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/polllayer.h"
#include "core/functions.h"
#include "core/training.h"
#include "core/utils.h"

#include "timer.h"
#include "test_utils.h"
#include "test_layers.h"

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

  void test_polling_op(
      const VectorBatch & input, const VectorBatch & expected,
      const MatrixSize & input_rows, const MatrixSize & input_cols,
      const MatrixSize & filter_size, enum PollingLayer::Mode mode)
  {
    VectorBatch output;
    output.resizeLike(expected);

    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;
      PollingLayer::poll(input, input_rows, input_cols, filter_size, mode, output);
      BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
    }
  }

  void test_backprop_polling_op(
      const VectorBatch & input, const VectorBatch & input_expected, const VectorBatch & expected,
      const MatrixSize & input_rows, const MatrixSize & input_cols,
      const MatrixSize & filter_size, enum PollingLayer::Mode mode)
  {
    VectorBatch input2, gradient_input, output, gradient_output;
    input2.resizeLike(input);
    gradient_input.resizeLike(input);
    output.resizeLike(expected);
    gradient_output.resizeLike(expected);

    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;
      PollingLayer::poll(input, input_rows, input_cols, filter_size, mode, output);
      gradient_output = output - expected;

      PollingLayer::poll_gradient_backprop(
          gradient_output, input, input_rows, input_cols,
          filter_size, mode, gradient_input);
      input2 = input - gradient_input;

      BOOST_CHECK(input_expected.isApprox(input2, TEST_TOLERANCE));
    }
  }
};
// struct PollingLayerTestFixture

BOOST_FIXTURE_TEST_SUITE(PollingLayerTest, PollingLayerTestFixture);

BOOST_AUTO_TEST_CASE(IO_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer IO test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 7;
  const MatrixSize filter_size = 2;
  PollingLayer one(input_cols, input_rows, filter_size, PollingLayer::PollMode_Max);
  one.init(Layer::InitMode_Random, boost::none);

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

//////////////////////////////////////////////////////////////////////
//
// MAX polling layer tests
//
BOOST_AUTO_TEST_CASE(PollOp_Max_Test)
{
  BOOST_TEST_MESSAGE("*** Polling Max operation test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      1, 2, 11, 10, 13, 18,
      3, 4, 12,  9, 14, 17,
      5, 6,  7,  8, 15, 16,
      /////////////////////
      2, 6,  8, 10, 13, 14,
      3, 2,  7,  9, 16, 15,
      4, 5, 13, 11, 16, 18;

  expected <<
      4, 12, 18,
      /////////
      6, 10, 16;

  test_polling_op(
      input, expected, input_rows, input_cols,
      filter_size, PollingLayer::PollMode_Max);
}


BOOST_AUTO_TEST_CASE(PollOp_Max_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** Polling Max Backprop operation test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;

  VectorBatch input, input_expected, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(input_expected, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      1, 2, 11, 10, 13, 18,
      3, 4, 12,  9, 14, 17,
      5, 6,  7,  8, 15, 16,
      /////////////////////
      2, 6,  8, 10, 13, 14,
      3, 2,  7,  9, 16, 15,
      4, 5, 13, 11, 16, 18;

  expected <<
      14, 22,  8,  // +10, +10, -10,
      //////////
      16,  0, 26;  // +10, -10, +10,

  input_expected <<
      1,  2, 11, 10, 13,  8,
      3, 14, 22,  9, 14, 17,
      5,  6,  7,  8, 15, 16,
      /////////////////////
      2, 16,  8,  0, 13, 14,
      3,  2,  7,  9, 26, 15,
      4,  5, 13, 11, 16, 18;

  test_backprop_polling_op(
      input, input_expected, expected,
      input_rows, input_cols, filter_size, PollingLayer::PollMode_Max);
}

BOOST_AUTO_TEST_CASE(Max_FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Max FeedForward test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;
  const Value ww = 2;
  const Value bb = 0.5;

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  input <<
      1, 2, 11, 10, 13, 18,
      3, 4, 12,  9, 14, 17,
      5, 6,  7,  8, 15, 16,
      /////////////////////
      2, 6,  8, 10, 13, 14,
      3, 2,  7,  9, 16, 15,
      4, 5, 13, 11, 16, 18;

  expected_output <<
      8.5,  24.5, 36.5,
      //////////////
      12.5, 20.5, 32.5;

  auto layer = make_unique<PollingLayer>(
      input_rows,
      input_cols,
      filter_size,
      PollingLayer::PollMode_Max);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<IdentityFunction>());
  layer->set_values(ww, bb);

  test_layer_feedforward(*layer, input, expected_output);
}

BOOST_AUTO_TEST_CASE(Max_Training_WithIdentity_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Max with Identity activation training test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;
  const Value ww = 2;
  const Value bb = 0.1;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.01, 0.02, 0.11, 0.10, 0.13, 0.18,
      0.03, 0.04, 0.12, 0.09, 0.14, 0.17,
      0.05, 0.06, 0.07, 0.08, 0.15, 0.16,
      /////////////////////
      0.02, 0.06, 0.08, 0.10, 0.13, 0.14,
      0.03, 0.02, 0.07, 0.09, 0.16, 0.15,
      0.04, 0.05, 0.13, 0.11, 0.16, 0.18;

  // we expect ww=3 and bb=0.5
  expected <<
      0.62, 0.86, 1.04,  // max=0.04, 0.12, 0.18,
      0.68, 0.80, 0.98;  // max=0.06, 0.10, 0.16;

  // create layer
  auto layer = make_unique<PollingLayer>(
      input_rows, input_cols,
      filter_size, PollingLayer::PollMode_Max);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<IdentityFunction>());
  layer->set_values(ww, bb);

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<QuadraticCost>(),
      0.07,  // learning rate
      5000  // epochs
  );
}

BOOST_AUTO_TEST_CASE(Max_Training_WithSigmoid_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Max with Sigmoid activation training test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;
  const Value ww = 2;
  const Value bb = 0.1;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.01, 0.02, 0.11, 0.10, 0.13, 0.18,
      0.03, 0.04, 0.12, 0.09, 0.14, 0.17,
      0.05, 0.06, 0.07, 0.08, 0.15, 0.16,
      /////////////////////
      0.02, 0.06, 0.08, 0.10, 0.13, 0.14,
      0.03, 0.02, 0.07, 0.09, 0.16, 0.15,
      0.04, 0.05, 0.13, 0.11, 0.16, 0.18;

  // we expect ww=3 and bb=0.5
  expected <<
      0.65022, 0.70266, 0.73885,  // max=0.04, 0.12, 0.18,
      0.66374, 0.68997, 0.72711;  // max=0.06, 0.10, 0.16;

  // create layer
  auto layer = make_unique<PollingLayer>(
      input_rows, input_cols,
      filter_size, PollingLayer::PollMode_Max);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<SigmoidFunction>());
  layer->set_values(ww, bb);

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<QuadraticCost>(),
      0.5,  // learning rate
      100000  // epochs
  );
}


//////////////////////////////////////////////////////////////////////
//
// AVG polling layer tests
//
BOOST_AUTO_TEST_CASE(PollOp_Avg_Test)
{
  BOOST_TEST_MESSAGE("*** Polling Avg operation test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      1, 2, 11, 10, 13, 18,
      3, 4, 12,  9, 14, 17,
      5, 6,  7,  8, 15, 16,
      /////////////////////
      2, 6,  8, 10, 13, 14,
      3, 2,  7,  9, 16, 15,
      4, 5, 13, 11, 16, 18;

  expected <<
      2.5, 10.5, 15.5,
      /////////////////
      3.25, 8.5, 14.5;

  test_polling_op(input, expected, input_rows, input_cols,
                  filter_size, PollingLayer::PollMode_Avg);
}

BOOST_AUTO_TEST_CASE(PollOp_Avg_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** Polling Avg Backprop operation test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;

  VectorBatch input, input_expected, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(input_expected, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      1, 2, 11, 10, 13, 18,
      3, 4, 12,  9, 14, 17,
      5, 6,  7,  8, 15, 16,
      /////////////////////
      2, 6,  8, 10, 13, 14,
      3, 2,  7,  9, 16, 15,
      4, 5, 13, 11, 16, 18;

  expected <<
      3.5, 9.5, 13.5,     // +1, -1, -2
      /////////////////
      1.25, 8.5, 16.5;    // -2,  0, +2

  input_expected <<
      2, 3, 10,  9, 11, 16,
      4, 5, 11,  8, 12, 15,
      5, 6,  7,  8, 15, 16,
      /////////////////////
      0, 4,  8, 10, 15, 16,
      1, 0,  7,  9, 18, 17,
      4, 5, 13, 11, 16, 18;

  test_backprop_polling_op(
      input, input_expected, expected,
      input_rows, input_cols, filter_size, PollingLayer::PollMode_Avg);
}

BOOST_AUTO_TEST_CASE(Avg_FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Avg FeedForward test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;
  const Value ww = 2;
  const Value bb = 0.5;

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  input <<
      1, 2, 11, 10, 13, 18,
      3, 4, 12,  9, 14, 17,
      5, 6,  7,  8, 15, 16,
      /////////////////////
      2, 6,  8, 10, 13, 14,
      3, 2,  7,  9, 16, 15,
      4, 5, 13, 11, 16, 18;

  expected_output <<
      5.5, 21.5, 31.5,
      /////////////////
      7.0, 17.5, 29.5;

  auto layer = make_unique<PollingLayer>(
      input_rows,
      input_cols,
      filter_size,
      PollingLayer::PollMode_Avg);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<IdentityFunction>());
  layer->set_values(ww, bb);

  test_layer_feedforward(*layer, input, expected_output);
}

BOOST_AUTO_TEST_CASE(Avg_Training_WithIdentity_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Avg with Identity activation training test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;
  const Value ww = 2;
  const Value bb = 0.1;

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
      0.2023, 0.1442, 0.1079,
      //////////////////////
      0.1968, 0.1587, 0.1152;

  // create layer
  auto layer = make_unique<PollingLayer>(
      input_rows, input_cols,
      filter_size, PollingLayer::PollMode_Avg);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<IdentityFunction>());
  layer->set_values(ww, bb);

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<QuadraticCost>(),
      0.01,     // learning rate
      100000  // epochs
  );
}


BOOST_AUTO_TEST_CASE(Avg_Training_WithSigmoid_Test)
{
  BOOST_TEST_MESSAGE("*** PollingLayer Avg with Sigmoid activation training test ...");

  const MatrixSize input_rows = 3;
  const MatrixSize input_cols = 6;
  const MatrixSize filter_size = 2;
  const MatrixSize input_size = input_cols * input_rows;
  const MatrixSize output_size = PollingLayer::get_output_size(input_cols, input_rows, filter_size);
  const MatrixSize batch_size = 2;
  const Value ww = 2;
  const Value bb = 0.1;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.01, 0.02, 0.11, 0.10, 0.13, 0.18,
      0.03, 0.04, 0.12, 0.09, 0.14, 0.17,
      0.05, 0.06, 0.07, 0.08, 0.15, 0.16,
      /////////////////////
      0.02, 0.06, 0.08, 0.10, 0.13, 0.14,
      0.03, 0.02, 0.07, 0.09, 0.16, 0.15,
      0.04, 0.05, 0.13, 0.11, 0.16, 0.18;

  expected <<
      0.2044, 0.1408, 0.1101,
      //////////////////////
      0.1976, 0.1550, 0.1157;

  // create layer
  auto layer = make_unique<PollingLayer>(
      input_rows, input_cols,
      filter_size, PollingLayer::PollMode_Avg);
  BOOST_CHECK(layer);
  layer->set_activation_function(make_unique<SigmoidFunction>());
  layer->set_values(ww, bb);

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      make_unique<CrossEntropyCost>(),
      1.0,  // learning rate
      10000  // epochs
  );
}

BOOST_AUTO_TEST_SUITE_END()

