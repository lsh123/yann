//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "Custom NeuralNetwork Tests"
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "core/types.h"
#include "core/functions.h"
#include "core/utils.h"
#include "core/random.h"
#include "core/training.h"
#include "core/nn.h"
#include "layers/contlayer.h"
#include "layers/convlayer.h"
#include "layers/fclayer.h"
#include "layers/polllayer.h"
#include "layers/smaxlayer.h"

#include "test_utils.h"
#include "timer.h"
#include "mnist_test.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

#define MNIST_TEST_FOLDER   "../data/mnist/"
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

}; // struct CustomNNTestFixture

BOOST_FIXTURE_TEST_SUITE(CustomNNTest, CustomNNTestFixture);

BOOST_AUTO_TEST_CASE(Test, * disabled())
{
  const auto input_rows  = _mnist_test.get_image_rows();
  const auto input_cols  = _mnist_test.get_image_cols();
  const auto output_size = _mnist_test.get_label_size();
  const size_t training_batch_size = 10;
  const size_t testing_batch_size = 100;
  const size_t epochs = 100;
  const double learning_rate = 0.75;
  const double regularization = 0.001;

  auto create_one_path = [&]() {
    auto poll_layer = make_unique<PollingLayer>(input_rows, input_cols, 2, PollingLayer::PollMode_Avg);
    YANN_CHECK(poll_layer);
    poll_layer->set_activation_function(make_unique<SigmoidFunction>());

    auto fc1_layer = make_unique<FullyConnectedLayer>(
        poll_layer->get_output_size(),
        300);
    YANN_CHECK(fc1_layer);
    fc1_layer->set_activation_function(make_unique<SigmoidFunction>());

    auto fc2_layer = make_unique<FullyConnectedLayer>(
        fc1_layer->get_output_size(),
        output_size);
    YANN_CHECK(fc2_layer);
    fc2_layer->set_activation_function(make_unique<SigmoidFunction>());

    // add to container
    auto container = make_unique<SequentialLayer>();
    YANN_CHECK(container);
    container->append_layer(std::move(poll_layer));
    container->append_layer(std::move(fc1_layer));
    container->append_layer(std::move(fc2_layer));

    return container;
  };
  auto container1 = create_one_path();
  auto container2 = create_one_path();

  auto conv_layer = make_unique<ConvolutionalLayer>(
      1,
      container1->get_output_size(),
      1);
  YANN_CHECK(conv_layer);
  conv_layer->set_activation_function(make_unique<SigmoidFunction>());

  auto bcast_layer = make_unique<BroadcastLayer>();
  bcast_layer->append_layer(std::move(container1));
  bcast_layer->append_layer(std::move(container2));

  auto merge_layer = make_unique<MergeLayer>(2);
  merge_layer->append_layer(std::move(conv_layer));

  auto container = make_unique<SequentialLayer>();
  YANN_CHECK(container);
  container->append_layer(std::move(bcast_layer));
  container->append_layer(std::move(merge_layer));

  // create the network
  auto nn = make_unique<Network>(std::move(container));
  YANN_CHECK(nn);
  nn->set_cost_function(make_unique<QuadraticCost>());
  nn->init(Layer::InitMode_Random);

  // Trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_GradientDescent>(learning_rate, regularization));
  BOOST_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Testing against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" Network: " << nn->get_info());
  BOOST_TEST_MESSAGE(" Trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" Epochs: " << epochs);

  // train and test
  pair<double, Value> res = _mnist_test.train_and_test(
      *nn,
      *trainer,
      DataSource_Stochastic::Random,
      training_batch_size,
      epochs,
      testing_batch_size);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" Network: " << nn->get_info());
  BOOST_TEST_MESSAGE(" Trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" Epochs: " << epochs);
}

BOOST_AUTO_TEST_SUITE_END()
