//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "layers/s2slayer.h"
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


struct Seq2SeqLayerTestFixture
{
  Seq2SeqLayerTestFixture()
  {

  }
  ~Seq2SeqLayerTestFixture()
  {

  }
};
// struct Seq2SeqLayerTestFixture

BOOST_FIXTURE_TEST_SUITE(Seq2SeqLayerTest, Seq2SeqLayerTestFixture);

BOOST_AUTO_TEST_CASE(IO_Test)
{
  BOOST_TEST_MESSAGE("*** Seq2SeqLayer IO test ...");

  const MatrixSize input_size = 4;
  const MatrixSize output_size = 3;

  auto one = Seq2SeqLayer::create_lstm(input_size, output_size, make_unique<SigmoidFunction>());
  BOOST_CHECK(one);
  one->init(Layer::InitMode_Random, boost::none);

  BOOST_TEST_MESSAGE("Seq2SeqLayer before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << *one;
  BOOST_CHECK(!oss.fail());

  auto two = Seq2SeqLayer::create_lstm(input_size, output_size, make_unique<SigmoidFunction>());
  BOOST_CHECK(two);
  std::istringstream iss(oss.str());
  iss >> *two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("Seq2SeqLayer after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(FeedForward_Test)
{
  BOOST_TEST_MESSAGE("*** Seq2SeqLayer FeedForward test ...");

  const MatrixSize input_size = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;

  Matrix ww_x_enc[LstmLayer::Gate_Max], ww_x_dec[LstmLayer::Gate_Max];
  Matrix ww_h_enc[LstmLayer::Gate_Max], ww_h_dec[LstmLayer::Gate_Max];
  Vector bb_enc[LstmLayer::Gate_Max], bb_dec[LstmLayer::Gate_Max];

  for(auto ii = 0; ii < LstmLayer::Gate_Max; ++ii) {
    ww_x_enc[ii].resize(input_size, output_size);
    ww_h_enc[ii].resize(output_size, output_size);
    bb_enc[ii].resize(output_size);

    ww_x_dec[ii].resize(output_size, output_size);
    ww_h_dec[ii].resize(output_size, output_size);
    bb_dec[ii].resize(output_size);

    ww_x_enc[ii].setConstant((ii + 1) / (Value)10);
    ww_h_enc[ii].setConstant((ii + 1) / (Value)10);
    bb_enc[ii].setConstant((ii + 1) / (Value)10);

    ww_x_dec[ii].setConstant((ii + 1) / (Value)100);
    ww_h_dec[ii].setConstant((ii + 1) / (Value)100);
    bb_dec[ii].setConstant((ii + 1) / (Value)100);
  }

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  input <<
      0.1, 0.2, 0.3, 0.4,
      0.5, 0.6, 0.7, 0.8;

  // identity
  auto layer_identity = Seq2SeqLayer::create_lstm(input_size, output_size, make_unique<IdentityFunction>());
  BOOST_CHECK(layer_identity);
  auto encoder_identity = dynamic_cast<LstmLayer*>(layer_identity->get_encoder());
  auto decoder_identity = dynamic_cast<LstmLayer*>(layer_identity->get_decoder());
  BOOST_CHECK(encoder_identity);
  BOOST_CHECK(decoder_identity);
  encoder_identity->set_values(ww_x_enc, ww_h_enc, bb_enc);
  decoder_identity->set_values(ww_x_dec, ww_h_dec, bb_dec);

  expected_output <<
      0.00016147813, 0.00016147813, 0.00016147813,
      9.805934e-06, 9.805934e-06, 9.805934e-06;
  test_layer_feedforward(*layer_identity, input, expected_output);

  // sigmoid
  auto layer_sigmoid = Seq2SeqLayer::create_lstm(input_size, output_size, make_unique<SigmoidFunction>());
  BOOST_CHECK(layer_sigmoid);
  auto encoder_sigmoid = dynamic_cast<LstmLayer*>(layer_sigmoid->get_encoder());
  auto decoder_sigmoid = dynamic_cast<LstmLayer*>(layer_sigmoid->get_decoder());
  BOOST_CHECK(encoder_sigmoid);
  BOOST_CHECK(decoder_sigmoid);
  encoder_sigmoid->set_values(ww_x_enc, ww_h_enc, bb_enc);
  decoder_sigmoid->set_values(ww_x_dec, ww_h_dec, bb_dec);

  expected_output <<
      0.2979, 0.2979, 0.2979,
      0.3155, 0.3155, 0.3155;
  test_layer_feedforward(*layer_sigmoid, input, expected_output);
}

BOOST_AUTO_TEST_CASE(Backprop_Test)
{
  BOOST_TEST_MESSAGE("*** Seq2SeqLayer backprop test ...");

  const MatrixSize input_size = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;
  const double learning_rate = 10;
  const size_t epochs = 10000;

  Matrix ww_x_enc[LstmLayer::Gate_Max], ww_x_dec[LstmLayer::Gate_Max];
  Matrix ww_h_enc[LstmLayer::Gate_Max], ww_h_dec[LstmLayer::Gate_Max];
  Vector bb_enc[LstmLayer::Gate_Max], bb_dec[LstmLayer::Gate_Max];

  for(auto ii = 0; ii < LstmLayer::Gate_Max; ++ii) {
    ww_x_enc[ii].resize(input_size, output_size);
    ww_h_enc[ii].resize(output_size, output_size);
    bb_enc[ii].resize(output_size);

    ww_x_dec[ii].resize(output_size, output_size);
    ww_h_dec[ii].resize(output_size, output_size);
    bb_dec[ii].resize(output_size);

    ww_x_enc[ii].setConstant((ii + 1) / (Value)10);
    ww_h_enc[ii].setConstant((ii + 1) / (Value)10);
    bb_enc[ii].setConstant((ii + 1) / (Value)10);

    ww_x_dec[ii].setConstant((ii + 1) / (Value)100);
    ww_h_dec[ii].setConstant((ii + 1) / (Value)100);
    bb_dec[ii].setConstant((ii + 1) / (Value)100);
  }

  VectorBatch input, expected_output;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected_output, batch_size, output_size);

  input <<
      0, 0, 0, 0, // was 0.1, 0.2, 0.3, 0.4,
      0, 0, 0, 0; // was 0.5, 0.6, 0.7, 0.8;

  expected_output <<
      0.2979, 0.2979, 0.2979,
      0.3155, 0.3155, 0.3155;

  auto layer = Seq2SeqLayer::create_lstm(input_size, output_size, make_unique<SigmoidFunction>());
  BOOST_CHECK(layer);
  auto encoder = dynamic_cast<LstmLayer*>(layer->get_encoder());
  auto decoder = dynamic_cast<LstmLayer*>(layer->get_decoder());
  BOOST_CHECK(encoder);
  BOOST_CHECK(decoder);

  encoder->set_values(ww_x_enc, ww_h_enc, bb_enc);
  decoder->set_values(ww_x_dec, ww_h_dec, bb_dec);

  test_layer_backprop(
      *layer,
      input,
      boost::none,
      expected_output,
      make_unique<CrossEntropyCost>(),
      learning_rate,
      epochs
  );
}

BOOST_AUTO_TEST_CASE(RandomBackprop_Test)
{
  BOOST_TEST_MESSAGE("*** Seq2SeqLayer backprop with random inputs and weights ...");

  const MatrixSize input_size = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;
  const double learning_rate = 1.0;
  const size_t epochs = 10000;

  // create layer
  auto layer = Seq2SeqLayer::create_lstm(input_size, output_size, make_unique<SigmoidFunction>());
  BOOST_CHECK(layer);
  layer->init(Layer::InitMode_Random, Layer::InitContext(123));

  // test
  test_layer_backprop_from_random(
      *layer,
      batch_size,
      make_unique<CrossEntropyCost>(),
      learning_rate,
      epochs
  );
}

BOOST_AUTO_TEST_CASE(Training_WithSigmoid_Test)
{
  BOOST_TEST_MESSAGE("*** Seq2SeqLayer with Sigmoid activation training test ...");

  const MatrixSize input_size = 4;
  const MatrixSize output_size = 3;
  const MatrixSize batch_size = 2;
  const double learning_rate = 0.75;
  const size_t epochs = 500;

  VectorBatch input, expected;
  resize_batch(input, batch_size, input_size);
  resize_batch(expected, batch_size, output_size);

  input <<
      0.1, 0.2, 0.3, 0.4,
      0.5, 0.6, 0.7, 0.8;

  expected <<
      0.2979, 0.2979, 0.2979,
      0.3155, 0.3155, 0.3155;

  // create layer
  auto layer = Seq2SeqLayer::create_lstm(input_size, output_size, make_unique<SigmoidFunction>());
  BOOST_CHECK(layer);
  layer->init(Layer::InitMode_Random, Layer::InitContext(123));

  // test
  test_layer_training(
      *layer,
      input,
      expected,
      1, // tests num
      make_unique<CrossEntropyCost>(),
      learning_rate,
      epochs
  );
}

BOOST_AUTO_TEST_CASE(RandomTraining_Test)
{
  BOOST_TEST_MESSAGE("*** Seq2SeqLayer training with random inputs ...");

  const MatrixSize input_size = 10;
  const MatrixSize output_size = 7;
  const MatrixSize batch_size = 5;
  const double learning_rate = 1.0;
  const size_t epochs = 50000;

  // create layer
  auto layer = Seq2SeqLayer::create_lstm(input_size, output_size, make_unique<SigmoidFunction>());
  BOOST_CHECK(layer);

  test_layer_training_from_random(
      *layer,
      batch_size,
      1, // tests num
      make_unique<CrossEntropyCost>(),
      learning_rate,
      epochs
  );
}

BOOST_AUTO_TEST_SUITE_END()

