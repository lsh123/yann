//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "FullyConnectedNetwork Tests"

#include <boost/test/unit_test.hpp>

#include <sstream>

#include "core/training.h"
#include "core/utils.h"
#include "core/functions.h"
#include "layers/fclayer.h"
#include "layers/smaxlayer.h"
#include "networks/fcnn.h"

#include "timer.h"
#include "test_utils.h"
#include "mnist_test.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

#define MNIST_TEST_FOLDER   "../data/mnist/"

// TODO: remove this
// Enums to simplify tests
enum ActivationMode
{
 Sigmoid,
 Softmax,
};

struct FcnnTestFixture
{

  MnistTest _mnist_test;

  FcnnTestFixture()
  {
    BOOST_TEST_MESSAGE("*** Reading MNIST dataset ...");
    _mnist_test.read(MNIST_TEST_FOLDER);
    BOOST_TEST_MESSAGE(_mnist_test);
  }
  ~FcnnTestFixture()
  {

  }

  unique_ptr<Network> create_fcnn_for_mnist(
      const vector<MatrixSize> & hidden_layer_sizes,
      const unique_ptr<CostFunction> & cost,
      enum ActivationMode activ_mode)
  {
    YANN_CHECK_GT(hidden_layer_sizes.size(), 0); // at least one hidden layer required

    // add input / output layers based on mnist data
    std::vector<MatrixSize> layer_sizes = _mnist_test.get_layer_sizes(hidden_layer_sizes);
    YANN_CHECK_GE(layer_sizes.size(), 2); // at least input + output

    // create params
    vector<FullyConnectedNetwork::Params> params(layer_sizes.size() - 1);
    for (size_t ii = 0; ii < params.size(); ++ii) {
      params[ii]._input_size  = layer_sizes[ii];
      params[ii]._output_size = layer_sizes[ii + 1];
      switch(activ_mode) {
        case Sigmoid:
          params[ii]._activation_function = make_unique<SigmoidFunction>();
          break;
        case Softmax:
          // Softmax only on the last layer
          if (ii < hidden_layer_sizes.size()) {
            params[ii]._activation_function = make_unique<SigmoidFunction>();
          } else {
            params[ii]._activation_function = make_unique<IdentityFunction>();
          }
          break;
      }
    }

    // create network
    auto fcnn = FullyConnectedNetwork::create(params);
    fcnn->set_cost_function(cost);

    if(activ_mode == Softmax) {
      auto smax_layer = make_unique<SoftmaxLayer>(fcnn->get_output_size());
      fcnn->append_layer(std::move(smax_layer));
    }

    // done
    return fcnn;
  }

  pair<double, Value> test_fcnn_mnist(
      const std::unique_ptr<Layer::Updater> & updater,
      DataSource_Stochastic::Mode mode,
      const MatrixSize & training_batch_size,
      const vector<MatrixSize> & hidden_layer_sizes,
      const unique_ptr<CostFunction> & cost,
      enum ActivationMode activ_mode,
      size_t epochs,
      const MatrixSize & test_batch_size = 100)
  {
    // setup
    auto fcnn = create_fcnn_for_mnist(hidden_layer_sizes, cost, activ_mode);
    YANN_CHECK(fcnn);
    fcnn->init(Layer::InitMode_Random, Layer::InitContext(12345));

    // create trainer
    Trainer trainer(updater->copy());
    trainer.set_batch_progress_callback(batch_progress_callback);

    BOOST_TEST_MESSAGE("*** Testing against MNIST dataset with ");
    BOOST_TEST_MESSAGE(" fcnn: " << fcnn->get_info());
    BOOST_TEST_MESSAGE(" trainer: " << trainer.get_info());
    BOOST_TEST_MESSAGE(" epochs: " << epochs);

    auto res = _mnist_test.train_and_test(
        *fcnn, trainer, mode, training_batch_size, epochs, test_batch_size);
    BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100)
                       << "% Loss: " << res.second
                       << " after " << epochs << " epochs");
    return res;
  }

  double test_fcnn_mnist_find_best(const std::unique_ptr<Layer::Updater> & updater,
                                   DataSource_Stochastic::Mode mode,
                                   const MatrixSize & training_batch_size,
                                   const Range& hidden_layers_num_range,
                                   const Range& hidden_layer_size_range,
                                   const unique_ptr<CostFunction> & cost,
                                   enum ActivationMode activ_mode,
                                   size_t epochs)
  {
    // run various networks and record results
    size_t best_hidden_layers_num = 0, best_hidden_layer_size = 0;
    double best_success_rate = 0;
    for (size_t hidden_layers_num = hidden_layers_num_range.min();
        hidden_layers_num <= hidden_layers_num_range.max(); hidden_layers_num +=
            hidden_layers_num_range.step()) {
      for (size_t hidden_layer_size = hidden_layer_size_range.min();
          hidden_layer_size <= hidden_layer_size_range.max();
          hidden_layer_size += hidden_layer_size_range.step()) {

        BOOST_TEST_MESSAGE("*** MNIST find best test with");
        BOOST_TEST_MESSAGE(" hidden_layers_num = " << hidden_layers_num);
        BOOST_TEST_MESSAGE(" hidden_layer_size = " << hidden_layer_size);

        vector<MatrixSize> hidden_layer_sizes;
        hidden_layer_sizes.insert(hidden_layer_sizes.end(), hidden_layers_num,
                                  hidden_layer_size);

        auto res = test_fcnn_mnist(
            updater, mode, training_batch_size, hidden_layer_sizes, cost, activ_mode, epochs);

        // update best
        if (res.first > best_success_rate) {
          best_success_rate = res.first;
          best_hidden_layers_num = hidden_layers_num;
          best_hidden_layer_size = hidden_layer_size;
        }
      }
    }

    BOOST_TEST_MESSAGE(
        "*** MNIST best success rate: " << best_success_rate << " with "
            << best_hidden_layers_num << " hidden layers num" << " and "
            << best_hidden_layer_size << " hidden layer size");
    return best_success_rate;
  }
};
// struct FcnnTestFixture

BOOST_FIXTURE_TEST_SUITE(FcnnTest, FcnnTestFixture);

BOOST_AUTO_TEST_CASE(Fcnn_IO_Test)
{
  BOOST_TEST_MESSAGE("*** FullyConnectedNetwork IO test ...");
  vector<MatrixSize> layers( { 3, 2, 1 });

  auto one = FullyConnectedNetwork::create(layers);
  YANN_CHECK(one);
  one->init(Layer::InitMode_Random);

  BOOST_TEST_MESSAGE("fcnn before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << *one;
  BOOST_CHECK(!oss.fail());

  auto two = FullyConnectedNetwork::create(layers);
  YANN_CHECK(two);
  std::istringstream iss(oss.str());
  iss >> *two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("fcnn after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Training_StochasticGD_Test)
{

  BOOST_TEST_MESSAGE("*** FullyConnectedNetwork training test ...");

  const MatrixSize input_size = 4;
  const MatrixSize output_size = 2;
  const MatrixSize training_batch_size = 4;
  const MatrixSize testing_batch_size = 2;
  const size_t epochs = 100;
  VectorBatch training_inputs, training_outputs;
  VectorBatch testing_inputs, testing_outputs;

  resize_batch(training_inputs,  training_batch_size, input_size);
  resize_batch(training_outputs, training_batch_size, output_size);
  resize_batch(testing_inputs,   testing_batch_size,  input_size);
  resize_batch(testing_outputs,  testing_batch_size,  output_size);

  training_inputs  << 0, 0, 0, 1,
                      0, 0, 1, 0,
                      0, 1, 0, 0,
                      1, 0, 0, 0;
  training_outputs << 0, 1,
                      0, 1,
                      1, 0,
                      1, 0;

  testing_inputs   << 0, 0, 1, 1,
                      1, 1, 0, 0;
  testing_outputs  << 0, 1,
                      1, 0;

  // training
  auto fcnn = FullyConnectedNetwork::create( { 4, 2 });
  YANN_CHECK(fcnn);

  Trainer trainer(make_unique<Updater_GradientDescent>(3.0, 0.0));
  DataSource_Stochastic data_source(
      training_inputs,
      training_outputs,
      DataSource_Stochastic::Sequential,
      4);
  BOOST_TEST_MESSAGE("trainer: " << trainer.get_info());
  {
    Timer timer("Initializing FullyConnectedNetwork");
    fcnn->init(Layer::InitMode_Random, Layer::InitContext(12345));
    BOOST_TEST_MESSAGE(timer);
  }

  {
    Timer timer("Training");
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
      // BOOST_TEST_MESSAGE("Training epoch " << epoch << " out of " << epochs);
      trainer.train(*fcnn, data_source);
    }
    BOOST_TEST_MESSAGE(timer);
  }
  BOOST_TEST_MESSAGE("After training:\n" << *fcnn);

  // testing
  VectorBatch res;
  for (MatrixSize ii = 0; ii < testing_batch_size; ii++) {
    VectorBatch in = get_batch(testing_inputs, ii); // copy
    VectorBatch out = get_batch(testing_outputs, ii); // copy
    res.resizeLike(out);

    fcnn->calculate(in, res);

    Value cost = fcnn->cost(res, out);
    BOOST_TEST_MESSAGE("test " << ii << " input: " << in);
    BOOST_TEST_MESSAGE("test " << ii << " expected ouput: " << out);
    BOOST_TEST_MESSAGE("test " << ii << " actual output: " << res);
    BOOST_TEST_MESSAGE("test " << ii << " cost/loss: " << cost);

    BOOST_CHECK_LE(cost, 0.01);
  }
}

BOOST_AUTO_TEST_CASE(Mnist_Sigmoid_CrossEntropy_Test)
{
  // reduce the test size to two labels to make it faster
  BOOST_TEST_MESSAGE("*** Filtering test set...");
  _mnist_test.filter(1, 2000, 1000); // only allow 0,1 images; 1000 count
  BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  auto res = test_fcnn_mnist(
      make_unique<Updater_GradientDescent>(2.0, 0.0),
      DataSource_Stochastic::Sequential, // reduce test variability
      10,   // batch size
      {20},
      make_unique<CrossEntropyCost>(1.0e-100),
      Sigmoid,
      3);
  BOOST_CHECK_GE(res.first, 0.995);  // > 99.5%
  BOOST_CHECK_LE(res.second, 0.05); // < 0.4 per test
}

BOOST_AUTO_TEST_CASE(Mnist_StochasticGD_Quadratic_Sigmoid_Test)
{
  auto res = test_fcnn_mnist(
      make_unique<Updater_GradientDescent>(3.0, 0.0),
      DataSource_Stochastic::Sequential,
      10, // batch size
      {20},
      make_unique<QuadraticCost>(),
      Sigmoid,
      3 // epochs
  );
  BOOST_CHECK_GE(res.first, 0.90); // > 90%
  BOOST_CHECK_LE(res.second, 0.20);// < 0.2 per test
}

BOOST_AUTO_TEST_CASE(Mnist_HellingerDistance_Softmax_Test)
{
  // reduce the test size to two labels to make it faster
  BOOST_TEST_MESSAGE("*** Filtering test set...");
  _mnist_test.filter(1, 2000, 1000); // only allow 0,1 images; 1000 count
  BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  auto res = test_fcnn_mnist(
      make_unique<Updater_GradientDescent>(5.0, 0.0),
      DataSource_Stochastic::Sequential, // reduce tests variability
      10, // batch size
      {20},
      make_unique<HellingerDistanceCost>(0.0001),
      Softmax,
      5 // epochs
  );
  BOOST_CHECK_GE(res.first, 0.98); // > 98%
  BOOST_CHECK_LE(res.second, 0.40);// < 0.4 per test
}

BOOST_AUTO_TEST_CASE(Mnist_StochasticGD_Quadratic_Sigmoid_Regularization_Test)
{
  auto res = test_fcnn_mnist(
      make_unique<Updater_GradientDescent>(3.0, 0.0005),
      DataSource_Stochastic::Sequential, // reduce tests variability
      10, // batch size
      {20},
      make_unique<QuadraticCost>(),
      Sigmoid,
      3);
  BOOST_CHECK_GE(res.first, 0.85); // > 85%
  BOOST_CHECK_LE(res.second, 0.25);// < 0.25 per test
}

BOOST_AUTO_TEST_CASE(Mnist_StochasticGD_CrossEntropy_Sigmoid_Test)
{
  auto res = test_fcnn_mnist(
      make_unique<Updater_GradientDescentWithMomentum>(0.75, 0.5),
      DataSource_Stochastic::Sequential, // reduce tests variability
      10, // batch size
      {20},
      make_unique<CrossEntropyCost>(),
      Sigmoid,
      3);
  BOOST_CHECK_GE(res.first, 0.90);// > 90%
  BOOST_CHECK_LE(res.second, 0.50);// < 0.5 per test
}

BOOST_AUTO_TEST_CASE(Mnist_FindBest_StochasticGD_Quadratic_Sigmoid_Test, * disabled())
{
  double res = test_fcnn_mnist_find_best(
      make_unique<Updater_GradientDescent>(3.0, 0.0),
      DataSource_Stochastic::Sequential, // reduce tests variability
      10, // batch size
      {0, 2, 1}, {10, 20, 5},
      make_unique<QuadraticCost>(),
      Sigmoid,
      2);
  BOOST_CHECK_GE(res, 0.90); // > 90%
}

BOOST_AUTO_TEST_CASE(Mnist_FindBest_StochasticGD_CrossEntropy_Sigmoid_Test, * disabled())
{
  double res = test_fcnn_mnist_find_best(
      make_unique<Updater_GradientDescent>(3.0, 0.0),
      DataSource_Stochastic::Sequential, // reduce tests variability
      10, // batch size
      {0, 2, 1}, {10, 20, 5},
      make_unique<CrossEntropyCost>(),
      Sigmoid,
      2);
  BOOST_CHECK_GE(res, 0.90); // > 90%
}

// The best result so far:
//
// Success rate for epoch 54: against training dataset: 100% against test dataset: 97.85%
// Cost/loss per test for epoch 54: against training dataset: 0.000619965 against test dataset: 0.236747
//
// Training time 14386.1 milliseconds
// Testing against training dataset time 3705.5 milliseconds
// Testing against test dataset time 617.968 milliseconds
//
// With:
// const vector<MatrixSize> hidden_layers = { 300, 100 };
// enum ActivationMode activ_mode = Sigmoid;
// unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>(1.0e-10);
// const double learning_rate = 0.90;
// const double regularization = 0.05;
// const size_t training_batch_size = 10;
// const size_t testing_batch_size = 100;
// Trainer trainer(make_unique<Updater_GradientDescentWithMomentum>(....)...)
//
BOOST_AUTO_TEST_CASE(Mnist_LargeFCNN_Test, * disabled())
{
  const vector<MatrixSize> hidden_layers = { 1000 };
  enum ActivationMode activ_mode = Sigmoid;
  unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>(1.0e-10);
  const double learning_rate = 0.90;
  const double regularization = 0.05;
  const size_t training_batch_size = 10;
  const size_t testing_batch_size = 100;
  const size_t epochs = 10;

  // Network
  auto fcnn = create_fcnn_for_mnist(
      hidden_layers,
      cost_func,
      activ_mode);
  YANN_CHECK(fcnn);

  // auto layer = dynamic_cast<FullyConnectedLayer*>(fcnn->get_layer(0));
  // YANN_CHECK(layer);
  // layer->set_sampling_rate(0.01); // 1%
  fcnn->init(Layer::InitMode_Random, Layer::InitContext(12345));

  // Trainer
  Trainer trainer(
      make_unique<Updater_GradientDescent>(
          learning_rate,
          regularization
      )
    );
  trainer.set_batch_progress_callback(batch_progress_callback);

  // Run test
  BOOST_TEST_MESSAGE("*** Testing against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" fcnn: " << fcnn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer.get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);
  auto res = _mnist_test.train_and_test(
      *fcnn, trainer, DataSource_Stochastic::Random,
      training_batch_size, epochs, testing_batch_size);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100)
                     << "% Loss: " << res.second
                     << " after " << epochs << " epochs");
}


BOOST_AUTO_TEST_SUITE_END()

