//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "functions.h"
#include "utils.h"
#include "random.h"
#include "layers/convlayer.h"
#include "layers/smaxlayer.h"

#include "test_utils.h"
#include "timer.h"
#include "mnist-test.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;


struct ConvolutionalLayerTestFixture
{
  ConvolutionalLayerTestFixture()
  {

  }
  ~ConvolutionalLayerTestFixture()
  {

  }

  inline MatrixSize find_max_pos(const RefConstVector & vv)
  {
    MatrixSize pos = 0;
    vv.maxCoeff(&pos);
    return pos;
  }

  class SimpleConvNetwork : public Network
  {
    typedef Network Base;

  public:
    static MatrixSize get_output_size(const MatrixSize & image_size, const MatrixSize & filter_size)
    {
      return ConvolutionalLayer::get_conv_output_size(image_size, image_size, filter_size);
    }

  public:
    SimpleConvNetwork(const MatrixSize & image_size, const MatrixSize & filter_size)
    {
      auto conv_layer = make_unique<ConvolutionalLayer>(image_size, image_size, filter_size);
      append_layer(std::move(conv_layer));
      auto smax_layer = make_unique<SoftmaxLayer>(get_output_size(image_size, filter_size));
      append_layer(std::move(smax_layer));
    }

    ConvolutionalLayer * get_conv_layer()
    {
      BOOST_VERIFY(get_layers_num() == 2); // we also have softmax layer
      return dynamic_cast<ConvolutionalLayer*>(get_layer(0));
    }
  }; // SimpleConvNetwork

  void conv_perf_test(const MatrixSize & size, const MatrixSize & filter_size, const size_t & epochs)
  {
      BOOST_TEST_MESSAGE("*** ConvOp Performance test with"
          << " size=" << size
          << ", filter_size=" << filter_size
          << ", epochs=" << epochs
      );

      Matrix input(size, size);
      Matrix filter(filter_size, filter_size);
      Matrix output(size + filter_size, size + filter_size);
      {
        Timer timer("Random generation");
        unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
        gen->generate(input);
        gen->generate(filter);
      }
      output.resize(ConvolutionalLayer::get_conv_output_rows(size, filter_size),
                    ConvolutionalLayer::get_conv_output_cols(size, filter_size));
      {
        Timer timer("Test plus_conv");
        for(auto ii = epochs; ii > 0; --ii) {
          ConvolutionalLayer::plus_conv(input, filter, output);
        }
      }
      output.resize(ConvolutionalLayer::get_full_conv_output_rows(size, filter_size),
                    ConvolutionalLayer::get_full_conv_output_cols(size, filter_size));
      {
        Timer timer("Test full_conv");
        for(auto ii = epochs; ii > 0; --ii) {
          ConvolutionalLayer::full_conv(input, filter, output);
        }
      }
  }

  void conv_perf_batch_test(const MatrixSize & batch_size, const MatrixSize & size, const MatrixSize & filter_size, const size_t & epochs)
  {
      BOOST_TEST_MESSAGE("*** ConvOp Performance batch test with"
          << " batch_size=" << batch_size
          << ", size=" << size
          << ", filter_size=" << filter_size
          << ", epochs=" << epochs
      );
      VectorBatch input, output;
      Matrix filter(filter_size, filter_size);

      resize_batch(input, batch_size, size * size);
      {
        Timer timer("Random generation");
        unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
        gen->generate(input);
        gen->generate(filter);
      }
      resize_batch(output, batch_size, ConvolutionalLayer::get_conv_output_size(size, size, filter_size));
      {
        Timer timer("Test plus_conv");
        for(auto ii = epochs; ii > 0; --ii) {
          ConvolutionalLayer::plus_conv(input, size, size, filter, output);
        }
      }
      resize_batch(output, batch_size, ConvolutionalLayer::get_full_conv_output_size(size, size, filter_size));
      {
        Timer timer("Test full_conv");
        for(auto ii = epochs; ii > 0; --ii) {
          ConvolutionalLayer::full_conv(input, size, size, filter, output);
        }
      }
  }
}; // struct ConvolutionalLayerTestFixture

BOOST_FIXTURE_TEST_SUITE(ConvolutionalLayerTest, ConvolutionalLayerTestFixture);

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_IO_Test)
{
  BOOST_TEST_MESSAGE("*** ConvolutionalLayer IO test ...");

  const MatrixSize input_cols = 5;
  const MatrixSize input_rows = 3;
  const MatrixSize filter_size = 2;
  ConvolutionalLayer one(input_cols, input_rows, filter_size);
  one.init(InitMode_Random_01);

  BOOST_TEST_MESSAGE("ConvolutionalLayer before writing to file: " << "\n" << one);
  ostringstream oss;
  oss << one;
  BOOST_CHECK(!oss.fail());

  ConvolutionalLayer two(input_cols, input_rows, filter_size);
  std::istringstream iss(oss.str());
  iss >> two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("ConvolutionalLayer after loading from file: " << "\n" << two);

  BOOST_CHECK(one.is_equal(two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_ConvOp_Test)
{
  BOOST_TEST_MESSAGE("*** Convolutional operation test ...");

  const size_t image_size = 4;
  const size_t max_filter_size = 4;
  Matrix filter(max_filter_size, max_filter_size);
  Matrix input(image_size, image_size);
  Matrix expected(image_size, image_size);
  Matrix output(image_size, image_size);

  //////////////////////////////////////////////////////////////////////
  //
  // Test different filter sizes
  //
  input <<
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12,
     13, 14, 15, 16;

  // 1x1
  filter.resize(1, 1);
  filter <<
      3;
  expected.resize(4, 4);
  expected <<
      3,  6,  9, 12,
     15, 18, 21, 24,
     27, 30, 33, 36,
     39, 42, 45, 48;
  output.resizeLike(expected);
  {
      // ensure we don't do allocations in eigen
      BlockAllocations block;
      ConvolutionalLayer::plus_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 2x1
  filter.resize(2, 1);
  filter <<
      2,
      3;
  expected.resize(3, 4);
  expected <<
      17, 22, 27, 32,
      37, 42, 47, 52,
      57, 62, 67, 72;
  output.resizeLike(expected);
  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;
    ConvolutionalLayer::plus_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 2x2
  filter.resize(2, 2);
  filter <<
      1, 2,
      3, 4;
  expected.resize(3, 3);
  expected <<
      44,  54,  64,
      84,  94, 104,
     124, 134, 144;
  output.resizeLike(expected);
  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;
    ConvolutionalLayer::plus_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 2x3
  filter.resize(2, 3);
  filter <<
      1, 2, 3,
      4, 5, 6;
  expected.resize(3, 2);
  expected <<
      106, 127,
      190, 211,
      274, 295;
  output.resizeLike(expected);
  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;
    ConvolutionalLayer::plus_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 3x3
  filter.resize(3, 3);
  filter <<
      1, 2, 3,
      4, 5, 6,
      7, 8, 9;
  expected.resize(2, 2);
  expected <<
      348, 393,
      528, 573;
  output.resizeLike(expected);
  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;
    ConvolutionalLayer::plus_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 4x4
  filter.resize(4, 4);
  filter <<
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12,
     13, 14, 15, 16;
  expected.resize(1, 1);
  expected <<
      1496;
  output.resizeLike(expected);
  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;
    ConvolutionalLayer::plus_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

}

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_ConvOp_Batch_Test)
{
  BOOST_TEST_MESSAGE("*** Convolutional operation batch test ...");

  const size_t max_filter_size = 4;
  const MatrixSize image_size = 4;
  const MatrixSize input_size = image_size * image_size;
  const MatrixSize batch_size = 2;
  Matrix filter(max_filter_size, max_filter_size);
  VectorBatch input, output, expected;

  //////////////////////////////////////////////////////////////////////
  //
  // Test different filter sizes
  //
  resize_batch(input, batch_size, input_size);
  input << 1, 0, 1, 0,
           0, 1, 0, 0,
           1, 0, 1, 0,
           0, 0, 0, 0,
           ///////////
           0, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1, 0,
           0, 0, 0, 0;

  // 1x1
  filter.resize(1, 1);
  filter << 1;
  resize_batch(expected, batch_size, 4 * 4);
  expected << 1, 0, 1, 0,
              0, 1, 0, 0,
              1, 0, 1, 0,
              0, 0, 0, 0,
              ///////////
              0, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 0;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::plus_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 2x1
  filter.resize(2, 1);
  filter << 1,
            1;
  resize_batch(expected, batch_size, 3 * 4);
  expected << 1, 1, 1, 0,
              1, 1, 1, 0,
              1, 0, 1, 0,
              ///////////
              0, 1, 0, 0,
              0, 1, 1, 0,
              0, 0, 1, 0;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::plus_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 2x2
  filter.resize(2, 2);
  filter << 1, 1,
            1, 0;
  resize_batch(expected, batch_size, 3 * 3);
  expected << 1, 2, 1,
              2, 1, 1,
              1, 1, 1,
              ////////
              0, 1, 0,
              1, 1, 1,
              0, 1, 1;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::plus_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 3x2
  filter.resize(3, 2);
  filter << 1, 1,
            1, 0,
            1, 0;
  resize_batch(expected, batch_size, 2 * 3);
  expected << 2, 2, 2,
              2, 1, 1,
              ////////
              0, 1, 1,
              1, 1, 1;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::plus_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 3x3
  filter.resize(3, 3);
  filter << 1, 1, 1,
            1, 0, 0,
            1, 0, 1;
  resize_batch(expected, batch_size, 2 * 2);
  expected << 4, 2,
              2, 1,
              /////
              1, 1,
              1, 1;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::plus_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 4x4
  filter.resize(4, 4);
  filter << 1, 1, 1, 1,
            1, 1, 0, 0,
            1, 0, 1, 0,
            1, 0, 0, 1;
  resize_batch(expected, batch_size, 1 * 1);
  expected << 5,
              //
              2;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::plus_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_FullConvOp_Test)
{
  BOOST_TEST_MESSAGE("*** Full convolutional operation test ...");

  const size_t image_size = 3;
  const size_t max_filter_size = 2;
  Matrix filter(max_filter_size, max_filter_size);
  Matrix input(image_size, image_size);
  Matrix expected(image_size + max_filter_size, image_size + max_filter_size);
  Matrix output(image_size + max_filter_size, image_size + max_filter_size);

  //////////////////////////////////////////////////////////////////////
  //
  // Test different filter sizes
  //
  input.resize(image_size, image_size);
  input <<
      1, 2, 3,
      4, 5, 6,
      7, 8, 9;

  // 1x1
  filter.resize(1, 1);
  filter <<
      3;
  expected.resize(3, 3);
  expected <<
      3,  6,  9,
     12, 15, 18,
     21, 24, 27;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::full_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 2x2
  filter.resize(2, 2);
  filter <<
      1, 2,
      3, 4;
  expected.resize(4, 4);
  expected <<
      4, 11, 18, 9,
     18, 37, 47, 21,
     36, 67, 77, 33,
     14, 23, 26, 9;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::full_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 2x3
  filter.resize(2, 3);
  filter <<
      1, 2, 3,
      4, 5, 6;
  expected.resize(4, 5);
  expected <<
      6,  17,  32, 23, 12,
     27,  58,  91, 58, 27,
     54, 106, 154, 94, 42,
     21,  38,  50, 26, 9;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::full_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 3x3
  filter.resize(3, 3);
  filter <<
      1, 2, 3,
      4, 5, 6,
      7, 8, 9;
  expected.resize(5, 5);
  expected <<
      9,  26,  50,  38,  21,
     42,  94, 154, 106,  54,
     90, 186, 285, 186,  90,
     54, 106, 154,  94,  42,
     21,  38,  50,  26,   9;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::full_conv(input, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_FullConvOp_Batch_Test)
{
  BOOST_TEST_MESSAGE("*** Full convolutional operation batch test ...");

  const size_t max_filter_size = 4;
  const MatrixSize image_size = 3;
  const MatrixSize input_size = image_size * image_size;
  const MatrixSize batch_size = 3;
  Matrix filter(max_filter_size, max_filter_size);
  VectorBatch input, output, expected;

  //////////////////////////////////////////////////////////////////////
  //
  // Test different filter sizes
  //
  resize_batch(input, batch_size, input_size);
  input << 1, 0, 1,
           0, 1, 0,
           1, 0, 0,
           ///////
           0, 0, 0,
           0, 0, 0,
           0, 0, 0,
           ///////
           1, 1, 1,
           1, 1, 1,
           1, 1, 1;

  // 1x1
  filter.resize(1, 1);
  filter << 1;
  resize_batch(expected, batch_size, 3 * 3);
  expected << 1, 0, 1,
              0, 1, 0,
              1, 0, 0,
              ///////
              0, 0, 0,
              0, 0, 0,
              0, 0, 0,
              ///////
              1, 1, 1,
              1, 1, 1,
              1, 1, 1;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::full_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 2x2
  filter.resize(2, 2);
  filter << 1, 1,
            1, 0;
  resize_batch(expected, batch_size, 4 * 4);
  expected << 0, 1, 0, 1,
              1, 1, 2, 1,
              0, 2, 1, 0,
              1, 1, 0, 0,
              //////////
              0, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 0, 0,
              //////////
              0, 1, 1, 1,
              1, 3, 3, 2,
              1, 3, 3, 2,
              1, 2, 2, 1;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::full_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 2x3
  filter.resize(2, 3);
  filter << 1, 1, 1,
            1, 0, 0;
  resize_batch(expected, batch_size, 4 * 5);
  expected << 0, 0, 1, 0, 1,
              1, 1, 2, 2, 1,
              0, 1, 2, 1, 0,
              1, 1, 1, 0, 0,
              /////////////
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              /////////////
              0, 0, 1, 1, 1,
              1, 2, 4, 3, 2,
              1, 2, 4, 3, 2,
              1, 2, 3, 2, 1;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::full_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

  // 3x3
  filter.resize(3, 3);
  filter << 1, 1, 1,
            1, 0, 0,
            1, 0, 1;
  resize_batch(expected, batch_size, 5 * 5);
  expected << 1, 0, 2, 0, 1,
              0, 1, 1, 1, 1,
              2, 1, 3, 2, 1,
              0, 1, 2, 1, 0,
              1, 1, 1, 0, 0,
              /////////////
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0,
              /////////////
              1, 1, 2, 1, 1,
              1, 1, 3, 2, 2,
              2, 3, 6, 4, 3,
              1, 2, 4, 3, 2,
              1, 2, 3, 2, 1;
  output.resizeLike(expected);
  {
     // ensure we don't do allocations in eigen
     BlockAllocations block;
     ConvolutionalLayer::full_conv(input, image_size, image_size, filter, output);
  }
  BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));

}

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_Rotate180_Test)
{
  BOOST_TEST_MESSAGE("*** Rotate180 test ...");

  const MatrixSize input0_size = 2;
  Matrix input0(input0_size, input0_size);
  input0 << 1, 2,
            3, 4;
  Matrix expected0(input0_size, input0_size);
  expected0 << 4, 3,
               2, 1;
  Matrix output0(input0_size, input0_size);
  {
       // ensure we don't do allocations in eigen
       BlockAllocations block;
       ConvolutionalLayer::rotate180(input0, output0);
  }
  BOOST_CHECK(expected0.isApprox(output0, TEST_TOLERANCE));

  const MatrixSize input1_size = 3;
  Matrix input1(input1_size, input1_size);
  input1 << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
  Matrix expected1(input1_size, input1_size);
  expected1 << 9, 8, 7,
               6, 5, 4,
               3, 2, 1;
  Matrix output1(input1_size, input1_size);
  {
       // ensure we don't do allocations in eigen
       BlockAllocations block;
       ConvolutionalLayer::rotate180(input1, output1);
  }
  BOOST_CHECK(expected1.isApprox(output1, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** ConvolutionalLayer FeedForward test ...");

  const MatrixSize image_size = 5;
  const MatrixSize input_size = image_size * image_size;
  const MatrixSize filter_size = 3;
  const MatrixSize batch_size = 2;
  const MatrixSize output_size = SimpleConvNetwork::get_output_size(image_size,
                                                              filter_size);

  Matrix ww(filter_size, filter_size);
  VectorBatch input, expected;

  ww << 1, 0, 1,
        0, 1, 0,
        1, 0, 1;

  resize_batch(input, batch_size, input_size);
  input << 1, 0, 1, 0, 0,
           0, 1, 0, 0, 0,
           1, 0, 1, 0, 0,
           0, 0, 0, 0, 0,
           0, 0, 0, 0, 1,
           /////////////
           0, 0, 1, 0, 1,
           0, 1, 0, 1, 0,
           1, 0, 1, 0, 1,
           0, 1, 0, 1, 0,
           1, 0, 1, 0, 0;

  resize_batch(expected, batch_size, output_size);
  expected << 5.5, 0.5, 2.5,
              0.5, 2.5, 0.5,
              2.5, 0.5, 2.5,
              /////////////
              4.5, 0.5, 5.5,
              0.5, 5.5, 0.5,
              5.5, 0.5, 4.5;

  ConvolutionalLayer layer(image_size, image_size, filter_size);
  layer.set_activation_function(make_unique<IdentityFunction>());
  layer.set_values(ww, 0.5);

  // Test writing output to the internal buffer
  {
    std::unique_ptr<Layer::Context> ctx = layer.create_context(batch_size);
    BOOST_VERIFY (ctx);
    {
         // ensure we don't do allocations in eigen
         BlockAllocations block;
         layer.feedforward(input, ctx.get());
    }
    RefConstVectorBatch output = ctx->get_output();
    BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
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
    }
    BOOST_CHECK(expected.isApprox(output, TEST_TOLERANCE));
  }
}

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** ConvolutionalLayer backprop test ...");

  const MatrixSize image_size = 3;
  const MatrixSize input_size = image_size * image_size;
  const MatrixSize filter_size = 2;
  const MatrixSize output_size = SimpleConvNetwork::get_output_size(image_size,
                                                              filter_size);
  const MatrixSize batch_size = 2;
  VectorBatch input, expected;

  Matrix ww(filter_size, filter_size);
  ww << 0, 0,
        0, 1;

  resize_batch(input, batch_size, input_size);
  input << 0, 0, 0,
           0, 1, 1,
           0, 1, 0,
           ////////
           0, 0, 0,
           1, 1, 0,
           1, 0, 0;

  resize_batch(expected, batch_size, output_size);
  expected << 1, 0,
              0, 0,
              ////
              0, 0,
              1, 0;

  ConvolutionalLayer layer(image_size, image_size, filter_size);
  layer.set_activation_function(make_unique<IdentityFunction>());
  layer.set_values(ww, 0.0);

  unique_ptr<Layer::Context> ctx = layer.create_training_context(batch_size);
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

      // update input and feed forward again
      input -= gradient_input;
      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
  }
}

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_Backprop_OnVector_Test)
{
  BOOST_TEST_MESSAGE("*** ConvolutionalLayer backprop on vector test ...");

  const MatrixSize shift = 4;
  const MatrixSize image_size = 3;
  const MatrixSize input_size = image_size * image_size;
  const MatrixSize filter_size = 2;
  const MatrixSize output_size = SimpleConvNetwork::get_output_size(image_size,
                                                              filter_size);
  const MatrixSize batch_size = 2;
  Vector input_buffer(shift + batch_size * input_size);
  MapMatrix input(input_buffer.data() + shift, batch_size, input_size); // this assumes RowMajor layout
  // Vector output_buffer(shift + output_size);
  VectorBatch expected;

  Matrix ww(filter_size, filter_size);
  ww << 0, 0,
        0, 1;

  input_buffer <<
           1, 2, 3, 4, // shift
           0, 0, 0,
           0, 1, 1,
           0, 1, 0,
           ////////
           0, 0, 0,
           1, 1, 0,
           1, 0, 0;

  resize_batch(expected, batch_size, output_size);
  expected << 1, 0,
              0, 0,
              ////
              0, 0,
              1, 0;

  ConvolutionalLayer layer(image_size, image_size, filter_size);
  layer.set_activation_function(make_unique<IdentityFunction>());
  layer.set_values(ww, 0.0);

  unique_ptr<Layer::Context> ctx = layer.create_training_context(batch_size);
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

      // update input and feed forward again
      input -= gradient_input;
      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected.isApprox(ctx->get_output(), TEST_TOLERANCE));
  }
}

BOOST_AUTO_TEST_CASE(ConvolutionalLayer_Training_Test)
{
  BOOST_TEST_MESSAGE("*** ConvolutionalLayer training test ...");

  const MatrixSize image_size = 3;
  const MatrixSize input_size = image_size * image_size;
  const MatrixSize filter_size = 2;
  const MatrixSize output_size = SimpleConvNetwork::get_output_size(image_size,
                                                              filter_size);
  const MatrixSize batch_size = 2;
  const double learning_rate = 5.0 / (double) batch_size;
  const size_t epochs = 100;
  VectorBatch expected0, inputs1, expected1, inputs2, expected2, outputs;
  resize_batch(expected0, batch_size, output_size);
  resize_batch(inputs1, batch_size, input_size);
  resize_batch(expected1, batch_size, output_size);
  resize_batch(inputs2, batch_size, input_size);
  resize_batch(expected2, batch_size, output_size);
  resize_batch(outputs, batch_size, output_size);

  inputs1 << 1, 1, 0,
             1, 0, 0,
             0, 0, 0,
             ///////
             0, 0, 0,
             0, 1, 1,
             0, 1, 0;
  expected1 << 1, 0,
               0, 0,
               ////
               0, 0,
               0, 1;
  inputs2 << 0, 1, 1,
             0, 1, 0,
             0, 0, 0,
             ///////
             0, 0, 0,
             1, 1, 0,
             1, 0, 0;
  expected2 << 0, 1,
               0, 0,
               ////
               0, 0,
               1, 0;

  // setup network
  Matrix ww(filter_size, filter_size);
  ww.setZero();

  SimpleConvNetwork net(image_size, filter_size);
  net.init(InitMode_Zeros);
  net.get_conv_layer()->set_values(ww, 0);

   // first calculate with zero ww's and b's: the output should be the same
  expected0.setConstant(0.25);
  std::unique_ptr<Context> ctx = net.create_context(outputs);
  BOOST_VERIFY(ctx);
  {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      net.calculate(inputs1, ctx.get());
      BOOST_CHECK(expected0.isApprox(outputs, TEST_TOLERANCE));

      net.calculate(inputs2, ctx.get());
      BOOST_CHECK(expected0.isApprox(outputs, TEST_TOLERANCE));
  }

  std::unique_ptr<TrainingContext> training_ctx = net.create_training_context(outputs);
  BOOST_VERIFY(training_ctx);
  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;

    // train
    for(size_t ii = 0; ii < epochs; ++ii) {
      training_ctx->reset_state();
      net.train(inputs1, expected1, training_ctx.get());
      net.update(training_ctx.get(), learning_rate, 0.0);
    }

    // check the results
    net.calculate(inputs1, training_ctx.get());
    for(MatrixSize ii = 0; ii < batch_size; ++ii) {
      BOOST_CHECK_EQUAL(
          find_max_pos(get_batch_const(outputs, ii)),
          find_max_pos(get_batch_const(expected1, ii))
      );
    }
    net.calculate(inputs2, training_ctx.get());
    for(MatrixSize ii = 0; ii < batch_size; ++ii) {
      BOOST_CHECK_EQUAL(
          find_max_pos(get_batch_const(outputs, ii)),
          find_max_pos(get_batch_const(expected2, ii))
      );
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// perf conv tests
//
BOOST_AUTO_TEST_CASE(PerfConvTest, * disabled())
{
  conv_perf_test(1000, 10, 100);
}

BOOST_AUTO_TEST_CASE(PerfBatchConvTest, * disabled())
{
  conv_perf_batch_test(10, 1000, 10, 10);
}

BOOST_AUTO_TEST_SUITE_END()
