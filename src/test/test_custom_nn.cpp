//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "Custom NeuralNetwork Tests"
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "types.h"
#include "functions.h"
#include "utils.h"
#include "random.h"
#include "nntraining.h"
#include "nn.h"
#include "layers/contlayer.h"
#include "layers/convlayer.h"
#include "layers/fclayer.h"
#include "layers/polllayer.h"
#include "layers/smaxlayer.h"

#include "test_utils.h"
#include "timer.h"
#include "mnist-test.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

#define MNIST_TEST_FOLDER   "../mnist/"
#define TMP_FOLDER  "/tmp/"

struct CustomNNTestFixture
{
  MnistTest _mnist_test;

  CustomNNTestFixture()
  {
    BOOST_TEST_MESSAGE("*** Reading MNIST dataset ...");
    _mnist_test.read(MNIST_TEST_FOLDER);
    BOOST_TEST_MESSAGE(_mnist_test);
  }
  ~CustomNNTestFixture()
  {
  }

  static void progress_callback(const MatrixSize & cur_pos, const MatrixSize & step, const MatrixSize & total)
  {
    // we want to print out progress at 1/10 increment
    auto progress_delta = total / 10;
    if(progress_delta == 0 || cur_pos % progress_delta < step) {
      BOOST_TEST_MESSAGE("  ... at " << cur_pos << " out of " << total);
    }
  }

  unique_ptr<MappingLayer> create_conv_mapping_layer(
      const vector<MappingLayer::InputsMapping> & mappings,
      const size_t & input_frames_num,
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const MatrixSize & filter_size,
      const unique_ptr<ActivationFunction> & activation_function)
  {
    BOOST_VERIFY(input_frames_num > 0);
    BOOST_VERIFY(mappings.size() > 0);

    auto mapping_layer = make_unique<MappingLayer>(input_frames_num);
    BOOST_VERIFY(mapping_layer);
    for(size_t ii = 0; ii < mappings.size(); ++ii) {
      auto conv_layer = make_unique<ConvolutionalLayer>(
          input_rows,
          input_cols,
          filter_size);
      BOOST_VERIFY(conv_layer);

      if(activation_function) {
        conv_layer->set_activation_function(activation_function);
      }
      mapping_layer->append_layer(std::move(conv_layer), mappings[ii]);
    }

    return mapping_layer;
  }

  unique_ptr<Network> create_lenet(
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    const MatrixSize & conv1_frames_num,
    const MatrixSize & conv1_filter_size,
    const MatrixSize & poll1_filter_size,
    const vector<MappingLayer::InputsMapping> & mappings,
    const MatrixSize & conv2_filter_size,
    const MatrixSize & poll2_filter_size,
    const MatrixSize & fc1_size,
    const MatrixSize & output_size
   ) {
    // ConvolutionalLayer #1
    MatrixSize conv1_input_rows = input_rows;
    MatrixSize conv1_input_cols = input_cols;
    auto conv1_layer = ConvolutionalLayer::create_conv_bcast_layer(
       conv1_frames_num,  // output frames num
       conv1_input_rows,  // input_rows,
       conv1_input_cols,  // input_cols,
       conv1_filter_size, // filter_size,
       make_unique<SigmoidFunction>()
    );
    BOOST_VERIFY(conv1_layer);

    // PollingLayer #1
    MatrixSize poll1_input_rows = ConvolutionalLayer::get_conv_output_rows(conv1_input_rows, conv1_filter_size);
    MatrixSize poll1_input_cols = ConvolutionalLayer::get_conv_output_cols(conv1_input_cols, conv1_filter_size);
    auto poll1_layer = PollingLayer::create_poll_parallel_layer(
       conv1_layer->get_layers_num(),
       poll1_input_rows,
       poll1_input_cols,
       poll1_filter_size,
       PollingLayer::PollMode_Avg);
    BOOST_VERIFY(poll1_layer);

    // ConvolutionalLayer #2
    MatrixSize conv2_input_rows = PollingLayer::get_output_rows(poll1_input_rows, poll1_filter_size);
    MatrixSize conv2_input_cols = PollingLayer::get_output_cols(poll1_input_cols, poll1_filter_size);
    auto conv2_layer = create_conv_mapping_layer(
       mappings,
       poll1_layer->get_layers_num(), // input frames num
       conv2_input_rows,  // input_rows,
       conv2_input_cols,  // input_cols,
       conv2_filter_size, // filter_size,
       make_unique<SigmoidFunction>()
    );
    BOOST_VERIFY(conv2_layer);

    // PollingLayer #2
    MatrixSize poll2_input_rows = ConvolutionalLayer::get_conv_output_rows(conv2_input_rows, conv2_filter_size);
    MatrixSize poll2_input_cols = ConvolutionalLayer::get_conv_output_cols(conv2_input_cols, conv2_filter_size);
    auto poll2_layer = PollingLayer::create_poll_parallel_layer(
       conv2_layer->get_layers_num(),
       poll2_input_rows,
       poll2_input_cols,
       poll2_filter_size,
       PollingLayer::PollMode_Avg);
    BOOST_VERIFY(poll2_layer);

    // FullyConnectedLayer #1
    auto fc1_layer = make_unique<FullyConnectedLayer>(
        poll2_layer->get_output_size(),
        fc1_size);
    BOOST_VERIFY(fc1_layer);

    // FullyConnectedLayer #2
    auto fc2_layer = make_unique<FullyConnectedLayer>(
        fc1_layer->get_output_size(),
        output_size);
    fc2_layer->set_activation_function(make_unique<IdentityFunction>());
    BOOST_VERIFY(fc2_layer);

    // SoftmaxLayer
    auto smax_layer = make_unique<SoftmaxLayer>(
        fc2_layer->get_output_size()
    );
    BOOST_VERIFY(smax_layer);

    // add to the network
    auto nn = make_unique<Network>();
    BOOST_VERIFY(nn);
    nn->set_cost_function(make_unique<QuadraticCost>());
    nn->append_layer(std::move(conv1_layer));
    nn->append_layer(std::move(poll1_layer));
    nn->append_layer(std::move(conv2_layer));
    nn->append_layer(std::move(poll2_layer));
    nn->append_layer(std::move(fc1_layer));
    nn->append_layer(std::move(fc2_layer));
    nn->append_layer(std::move(smax_layer));

    // done
    return nn;
  }

  unique_ptr<Network> create_mapping_layer(
      const size_t & input_frames_num,
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const vector<MappingLayer::InputsMapping> & mappings,
      const MatrixSize & conv1_filter_size,
      const MatrixSize & poll1_filter_size,
      const MatrixSize & output_size
   ) {
      // ConvolutionalLayer #1
      MatrixSize conv1_input_rows = input_rows;
      MatrixSize conv1_input_cols = input_cols;
      auto conv1_layer = create_conv_mapping_layer(
         mappings,
         input_frames_num,  // input frames num
         conv1_input_rows,  // input_rows,
         conv1_input_cols,  // input_cols,
         conv1_filter_size, // filter_size,
         make_unique<SigmoidFunction>()
      );
      BOOST_VERIFY(conv1_layer);

      // PollingLayer #1
      MatrixSize poll1_input_rows = ConvolutionalLayer::get_conv_output_rows(conv1_input_rows, conv1_filter_size);
      MatrixSize poll1_input_cols = ConvolutionalLayer::get_conv_output_cols(conv1_input_cols, conv1_filter_size);
      auto poll1_layer = PollingLayer::create_poll_parallel_layer(
         conv1_layer->get_layers_num(),
         poll1_input_rows,
         poll1_input_cols,
         poll1_filter_size,
         PollingLayer::PollMode_Avg);
      BOOST_VERIFY(poll1_layer);

      // FullyConnectedLayer #1
      auto fc1_layer = make_unique<FullyConnectedLayer>(
          poll1_layer->get_output_size(),
          output_size);
      BOOST_VERIFY(fc1_layer);
      fc1_layer->set_activation_function(make_unique<IdentityFunction>());

      // SoftmaxLayer
      auto smax_layer = make_unique<SoftmaxLayer>(
          fc1_layer->get_output_size()
      );
      BOOST_VERIFY(smax_layer);

      // add to the network
      auto nn = make_unique<Network>();
      BOOST_VERIFY(nn);
      nn->set_cost_function(make_unique<QuadraticCost>());
      nn->append_layer(std::move(conv1_layer));
      nn->append_layer(std::move(poll1_layer));
      nn->append_layer(std::move(fc1_layer));
      nn->append_layer(std::move(smax_layer));

      // done
      return nn;
    }
}; // struct CustomNNTestFixture

BOOST_FIXTURE_TEST_SUITE(CustomNNTest, CustomNNTestFixture);

BOOST_AUTO_TEST_CASE(Mapping_Test, * disabled())
{
  // reduce the test size to two labels to make it faster
  {
   BOOST_TEST_MESSAGE("*** Filtering test set...");
   MnistTest filtered;
   filtered.filter(_mnist_test, 1);
   _mnist_test = filtered;
   BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);
  }

  const double learning_rate = 2.0;
  const double regularization = 0.00001;
  const MatrixSize training_batch_size = 10;
  const MatrixSize testing_batch_size = 10;
  const size_t epochs = 100;

  vector<MappingLayer::InputsMapping> mappings;
  mappings.push_back({ 0 });
  mappings.push_back({ 0 });
  mappings.push_back({ 1 });

  // look at the image as 2 separate parts: top and bottom
  const size_t input_frames_num = 2;
  const auto input_rows = _mnist_test.get_image_rows() / input_frames_num;
  const auto input_cols = _mnist_test.get_image_cols();
  auto nn = create_mapping_layer(
      input_frames_num,
      input_rows,
      input_cols,
      mappings,
      7,  // conv1_filter_size
      2,  // poll1_filter_size
      _mnist_test.get_label_size()  // output_size
  );
  BOOST_VERIFY(nn);

  // Trainer
  auto trainer = make_unique<Trainer_MiniBatch_GD>(
      learning_rate,      // learning rate
      regularization,     // regularization
      Trainer::Random,
      training_batch_size  // batch_size
  );
  BOOST_CHECK(trainer);
  trainer->set_progress_callback(progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Testing against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // train and test
  nn->init(InitMode_Random_01);
  pair<double, Value> res = _mnist_test.train_and_test(*nn, *trainer,epochs, testing_batch_size);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // save
  // string filename = TMP_FOLDER + Timer::get_time() + ".nn";
  // nn->save(filename);
  // BOOST_TEST_MESSAGE("*** Saved to file: " << filename);
}

BOOST_AUTO_TEST_CASE(LeNet4_Test)
{
  // reduce the test size to make it faster
  /*
  {
   BOOST_TEST_MESSAGE("*** Filtering test set...");
   MnistTest filtered;
   filtered.filter(_mnist_test, 1, 1000, 1000); // 1000 training and 1000 testing
   _mnist_test = filtered;
   BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);
  }
  */

  const double learning_rate = 0.75;
  const double regularization = 0.0001;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 100;

  // see http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf
  const size_t conv2_frames_num = 12;
  vector<MappingLayer::InputsMapping> mappings(conv2_frames_num);
  mappings[0]  = { 0 };
  mappings[1]  = { 0, 1 };
  mappings[2]  = { 0, 1 };
  mappings[3]  = { 1 };
  mappings[4]  = { 0, 1 };
  mappings[5]  = { 0, 1 };
  mappings[6]  = { 2 };
  mappings[7]  = { 2, 3 };
  mappings[8]  = { 2, 3 };
  mappings[9]  = { 3 };
  mappings[10] = { 2, 3 };
  mappings[11] = { 2, 3 };

  auto nn = create_lenet(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      4,  // conv1_frames_num
      5,  // conv1_filter_size
      2,  // poll1_filter_size
      mappings,
      5,  // conv2_filter_size
      2,  // poll2_filter_size
      30, // 100, // fc1_size
      _mnist_test.get_label_size()  // output_size
  );
  BOOST_VERIFY(nn);

  // Trainer
  auto trainer = make_unique<Trainer_MiniBatch_GD>(
      learning_rate,      // learning rate
      regularization,     // regularization
      Trainer::Random,
      training_batch_size  // batch_size
  );
  BOOST_CHECK(trainer);
  trainer->set_progress_callback(progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Testing against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // train and test
  nn->init(InitMode_Random_01);
  pair<double, Value> res = _mnist_test.train_and_test(*nn, *trainer, epochs, 100);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // save
  string filename = TMP_FOLDER + Timer::get_time() + ".nn";
  nn->save(filename);
  BOOST_TEST_MESSAGE("*** Saved to file: " << filename);
}

BOOST_AUTO_TEST_CASE(Large_FC_Test, * disabled())
{
  const auto input_rows  = _mnist_test.get_image_rows();
  const auto input_cols  = _mnist_test.get_image_cols();
  const auto input_size  = input_rows * input_cols;
  const auto output_size = _mnist_test.get_label_size();
  const auto fc1_size    = 300;
  const double learning_rate = 0.75;
  const double regularization = 0.0;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 100;

  // FullyConnectedLayer #1
  auto fc1_layer = make_unique<FullyConnectedLayer>(
      input_size,
      fc1_size);
  BOOST_VERIFY(fc1_layer);
  fc1_layer->set_activation_function(make_unique<ReluFunction>());

  // FullyConnectedLayer #2
  auto fc2_layer = make_unique<FullyConnectedLayer>(
      fc1_layer->get_output_size(),
      output_size);
  BOOST_VERIFY(fc2_layer);
  fc2_layer->set_activation_function(make_unique<SigmoidFunction>());

  // SoftmaxLayer
  auto smax_layer = make_unique<SoftmaxLayer>(
      fc2_layer->get_output_size()
  );
  BOOST_VERIFY(smax_layer);

  // add to the network
  auto nn = make_unique<Network>();
  BOOST_VERIFY(nn);
  nn->set_cost_function(make_unique<QuadraticCost>());
  // nn->set_cost_function(make_unique<CrossEntropyCost>());
  nn->append_layer(std::move(fc1_layer));
  nn->append_layer(std::move(fc2_layer));
  nn->append_layer(std::move(smax_layer));

  // Trainer
  auto trainer = make_unique<Trainer_MiniBatch_GD>(
      learning_rate,      // learning rate
      regularization,     // regularization
      Trainer::Random,
      training_batch_size  // batch_size
  );
  BOOST_CHECK(trainer);
  trainer->set_progress_callback(progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Testing against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // train and test
  nn->init(InitMode_Random_01);
  pair<double, Value> res = _mnist_test.train_and_test(*nn, *trainer,epochs, 100);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);
}

BOOST_AUTO_TEST_SUITE_END()
