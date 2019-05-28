//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "FullyConnectedNetwork Tests"

#include <boost/test/unit_test.hpp>

#include <sstream>

#include "fcnn.h"
#include "nntraining.h"
#include "utils.h"
#include "functions.h"
#include "layers/smaxlayer.h"

#include "timer.h"
#include "test_utils.h"
#include "mnist-test.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

#define MNIST_TEST_FOLDER   "../mnist/"

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

  static void progress_callback(const MatrixSize & cur_pos, const MatrixSize & step, const MatrixSize & total)
  {
    // we want to print out progress at 1/10 increment
    auto progress_delta = total / 10;
    if(progress_delta == 0 || cur_pos % progress_delta < step) {
      BOOST_TEST_MESSAGE("  ... at " << cur_pos << " out of " << total);
    }
  }

  unique_ptr<Network> create_fcnn_for_mnist(
      const vector<MatrixSize> & hidden_layer_sizes,
      enum CostMode cost_mode,
      enum ActivationMode activ_mode)
  {
    BOOST_VERIFY(hidden_layer_sizes.size() > 0); // at least one hidden layer required

    // add input / output layers based on mnist data
    std::vector<MatrixSize> layer_sizes = _mnist_test.get_layer_sizes(hidden_layer_sizes);
    BOOST_VERIFY(layer_sizes.size() >= 2); // at least input + output

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
    switch(cost_mode) {
      case Quadratic:
        fcnn->set_cost_function(make_unique<QuadraticCost>());
        break;
      case CrossEntropy:
        fcnn->set_cost_function(make_unique<CrossEntropyCost>());
        break;
    }
    if(activ_mode == Softmax) {
      auto smax_layer = make_unique<SoftmaxLayer>(fcnn->get_output_size());
      fcnn->append_layer(std::move(smax_layer));
    }

    // done
    return fcnn;
  }

  pair<double, Value> test_fcnn_mnist(const Trainer & trainer,
                                      const vector<MatrixSize> & hidden_layer_sizes,
                                      enum CostMode cost_mode,
                                      enum ActivationMode activ_mode,
                                      InitMode init_mode,
                                      size_t epochs)
  {
    // setup
    auto fcnn = create_fcnn_for_mnist(hidden_layer_sizes, cost_mode, activ_mode);
    BOOST_VERIFY(fcnn);
    fcnn->init(init_mode);

    BOOST_TEST_MESSAGE("*** Testing against MNIST dataset with ");
    BOOST_TEST_MESSAGE(" fcnn: " << fcnn->get_info());
    BOOST_TEST_MESSAGE(" trainer: " << trainer.get_info());
    BOOST_TEST_MESSAGE(" epochs: " << epochs);

    pair<double, Value> res = _mnist_test.train_and_test(*fcnn, trainer, epochs, 100);
    BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
    return res;
  }

  double test_fcnn_mnist_find_best(const Trainer & trainer,
                                   const Range& hidden_layers_num_range,
                                   const Range& hidden_layer_size_range,
                                   enum CostMode cost_mode,
                                   enum ActivationMode activ_mode,
                                   InitMode init_mode,
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

        pair<double, Value> res = test_fcnn_mnist(
            trainer,
            hidden_layer_sizes,
            cost_mode, activ_mode,
            init_mode, epochs);

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
  BOOST_VERIFY(one);
  one->init(InitMode_Random_01);

  BOOST_TEST_MESSAGE("fcnn before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << *one;
  BOOST_CHECK(!oss.fail());

  auto two = FullyConnectedNetwork::create(layers);
  BOOST_VERIFY(two);
  std::istringstream iss(oss.str());
  iss >> *two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("fcnn after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Training_MiniBatch_GD_Test)
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
  BOOST_VERIFY(fcnn);

  Trainer_MiniBatch_GD trainer(3, 0.0, Trainer::Random, 4);
  BOOST_TEST_MESSAGE("trainer: " << trainer.get_info());
  {
    Timer timer("Initializing FullyConnectedNetwork");
    fcnn->init(InitMode_Random_01);
    BOOST_TEST_MESSAGE(timer);
  }

  {
    Timer timer("Training");
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
      // BOOST_TEST_MESSAGE("Training epoch " << epoch << " out of " << epochs);
      trainer.train(*fcnn, training_inputs, training_outputs);
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

BOOST_AUTO_TEST_CASE(Mnist_Incremental_GD_Quadratic_Sigmoid_Test)
{
  Trainer_Incremental_GD trainer(3.0, 0, Trainer::Random, 10);
  trainer.set_progress_callback(progress_callback);

  pair<double, Value> res = test_fcnn_mnist(
      trainer,
      {20},
      Quadratic,
      Sigmoid,
      InitMode_Random_01,
      3);
  BOOST_CHECK_GE(res.first, 0.90); // > 90%
  BOOST_CHECK_LE(res.second, 0.20);// < 0.2 per test
}

BOOST_AUTO_TEST_CASE(Mnist_MiniBatch_GD_Quadratic_Sigmoid_Test)
{
  Trainer_MiniBatch_GD trainer(3.0, 0, Trainer::Random, 10);
  trainer.set_progress_callback(progress_callback);

  pair<double, Value> res = test_fcnn_mnist(
      trainer,
      {20},
      Quadratic,
      Sigmoid,
      InitMode_Random_01,
      3);
  BOOST_CHECK_GE(res.first, 0.90); // > 90%
  BOOST_CHECK_LE(res.second, 0.20);// < 0.2 per test
}

BOOST_AUTO_TEST_CASE(Mnist_Incremental_GD_Quadratic_Softmax_Test)
{
  Trainer_Incremental_GD trainer(5.0, 0, Trainer::Random, 10);
  trainer.set_progress_callback(progress_callback);

  pair<double, Value> res = test_fcnn_mnist(
      trainer,
      {20},
      Quadratic,
      Softmax,
      InitMode_Random_01,
      3);
  BOOST_CHECK_GE(res.first, 0.85); // > 85% -- Softmax + Quadratic is a bad choice
  BOOST_CHECK_LE(res.second, 0.20);// < 0.2 per test
}

BOOST_AUTO_TEST_CASE(Mnist_MiniBatch_GD_Quadratic_Sigmoid_Regularization_Test)
{
  Trainer_MiniBatch_GD trainer(3.0, 0.0005, Trainer::Random, 10);
  trainer.set_progress_callback(progress_callback);

  pair<double, Value> res = test_fcnn_mnist(
      trainer,
      {20},
      Quadratic,
      Sigmoid,
      InitMode_Random_01,
      3);
  BOOST_CHECK_GE(res.first, 0.85); // > 85%
  BOOST_CHECK_LE(res.second, 0.25);// < 0.25 per test
}

BOOST_AUTO_TEST_CASE(Mnist_MiniBatch_GD_CrossEntropy_Sigmoid_Test)
{
  Trainer_MiniBatch_GD trainer(1.0, 0.0, Trainer::Random, 10);
  trainer.set_progress_callback(progress_callback);

  pair<double, Value> res = test_fcnn_mnist(
      trainer,
      {20},
      CrossEntropy,
      Sigmoid,
      InitMode_Random_01,
      3);
  BOOST_CHECK_GE(res.first, 0.80);// > 80%
  BOOST_CHECK_LE(res.second, 0.80);// < 0.8 per test
}

BOOST_AUTO_TEST_CASE(Mnist_FindBest_MiniBatch_GD_Quadratic_Sigmoid_Test, * disabled())
{
  Trainer_Incremental_GD trainer(3.0, 0.0, Trainer::Random, 10);
  trainer.set_progress_callback(progress_callback);

  double res = test_fcnn_mnist_find_best(
      trainer,
      {0, 2, 1}, {10, 20, 5},
      Quadratic,
      Sigmoid,
      InitMode_Random_01,
      2);
  BOOST_CHECK_GE(res, 0.90); // > 90%
}

BOOST_AUTO_TEST_CASE(Mnist_FindBest_MiniBatch_GD_CrossEntropy_Sigmoid_Test, * disabled())
{
  Trainer_Incremental_GD trainer(3.0, 0.0, Trainer::Random, 10);
  trainer.set_progress_callback(progress_callback);

  double res = test_fcnn_mnist_find_best(
      trainer,
      {0, 2, 1}, {10, 20, 5},
      CrossEntropy,
      Sigmoid,
      InitMode_Random_01,
      2);
  BOOST_CHECK_GE(res, 0.90); // > 90%
}

BOOST_AUTO_TEST_CASE(Mnist_MiniBatch_Sigmoid_400_epochs_Test, * disabled())
{
  Trainer_MiniBatch_GD trainer(
      0.5,     // learning rate
      0.001,     // regularization
      Trainer::Random,
      10       // batch size
    );
  trainer.set_progress_callback(progress_callback);

  pair<double, Value> res = test_fcnn_mnist(
      trainer,
      {100},
      CrossEntropy,
      Sigmoid,
      InitMode_Random_01,
      400 // epochs
  );
  BOOST_TEST_MESSAGE("Result success rate: " << res.first);
}


BOOST_AUTO_TEST_SUITE_END()

