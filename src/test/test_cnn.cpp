//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "ConvolutionalNetwork Tests"
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "core/types.h"
#include "core/functions.h"
#include "core/utils.h"
#include "core/random.h"
#include "core/updaters.h"
#include "core/training.h"
#include "layers/smaxlayer.h"

#include "networks/cnn.h"

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

struct CnnTestFixture
{
  MnistTest _mnist_test;

  CnnTestFixture()
  {
    BOOST_TEST_MESSAGE("*** Reading MNIST dataset ...");
    _mnist_test.read(MNIST_TEST_FOLDER);
    BOOST_TEST_MESSAGE(_mnist_test);
  }
  ~CnnTestFixture()
  {
  }

  void save_to_file(const unique_ptr<Network> & nn)
  {
    // save
    string filename = TMP_FOLDER + Timer::get_time() + ".nn";
    nn->save(filename);
    BOOST_TEST_MESSAGE("*** Saved to file: " << filename);
  }

  // Creates CNN with one Conv+Poll layer, FC layer and Softmax layer
  unique_ptr<Network> create_cnn_for_mnist(
      const MatrixSize & conv_filter_size,
      const MatrixSize & polling_size,
      const MatrixSize & frames_num,
      const unique_ptr<CostFunction> & cost)
  {
    // create params
    ConvolutionalNetwork::ConvPollParams params;
    params._output_frames_num = frames_num;
    params._conv_filter_size = conv_filter_size;
    params._conv_activation_funtion = make_unique<SigmoidFunction>();
    params._polling_mode = PollingLayer::PollMode_Max;
    params._polling_filter_size = polling_size;


    // create conv layers
    auto layers = ConvolutionalNetwork::create(
        _mnist_test.get_image_rows(),
        _mnist_test.get_image_cols(),
        params,
        make_unique<IdentityFunction>(),
        _mnist_test.get_label_size());
    YANN_CHECK(layers);

    // add softmaxLayer
    auto smax_layer = make_unique<SoftmaxLayer>(
        layers->get_output_size()
    );
    YANN_CHECK(smax_layer);
    layers->append_layer(std::move(smax_layer));

    // networks
    auto cnn = make_unique<Network>(std::move(layers));
    YANN_CHECK(cnn);
    cnn->set_cost_function(cost);

    // done
    return cnn;
  }
}; // struct CnnTestFixture

BOOST_FIXTURE_TEST_SUITE(CnnTest, CnnTestFixture);

BOOST_AUTO_TEST_CASE(IO_Test)
{
  BOOST_TEST_MESSAGE("*** ConvolutionalNetwork IO test ...");

  const MatrixSize image_size = 32;
  const MatrixSize output_size = 10;

  // create params
  ConvolutionalNetwork::ConvPollParams params;
  params._output_frames_num = 3;
  params._conv_filter_size = 2;
  params._conv_activation_funtion = make_unique<SigmoidFunction>();
  params._polling_mode = PollingLayer::PollMode_Max;
  params._polling_filter_size = 2;

  // create, init and write the network
  auto one_layers = ConvolutionalNetwork::create(
      image_size, // input rows,
      image_size, // input cols
      params,
      make_unique<SigmoidFunction>(), // FC layer activation func
      output_size      // output size
  );
  auto one = make_unique<Network>(std::move(one_layers));
  YANN_CHECK(one);
  one->init(Layer::InitMode_Random);

  // BOOST_TEST_MESSAGE("ConvolutionalNetwork before writing to file: " << (*one));
  ostringstream oss;
  oss << (*one);
  BOOST_CHECK(!oss.fail());

  // read network
  auto two_layers = ConvolutionalNetwork::create(
      image_size, // input rows,
      image_size, // input cols
      params,
      make_unique<SigmoidFunction>(), // FC layer activation func
      output_size      // output size
  );
  auto two = make_unique<Network>(std::move(two_layers));
  YANN_CHECK(two);
  std::istringstream iss(oss.str());
  iss >> (*two);
  BOOST_CHECK(!iss.fail());
  // BOOST_TEST_MESSAGE("ConvolutionalNetwork after loading from file: " << (*two));

  // compare
  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Training_GradientDescent_Test)
{
  BOOST_TEST_MESSAGE("*** ConvolutionalNetwork training test...");

  const size_t epochs = 300;
  const MatrixSize image_size = 2;
  const MatrixSize filter_size = 2;
  const MatrixSize polling_size = 1;
  const MatrixSize input_size = image_size * image_size;
  const MatrixSize output_size = 2;
  const MatrixSize training_batch_size = 4;
  const MatrixSize testing_batch_size = 2;
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

  ConvolutionalNetwork::ConvPollParams params;
  params._output_frames_num = 1;
  params._conv_filter_size = filter_size;
  params._conv_activation_funtion = make_unique<SigmoidFunction>();
  params._polling_mode = PollingLayer::PollMode_Max;
  params._polling_filter_size = polling_size;
  auto layers = ConvolutionalNetwork::create(
      image_size, // input rows
      image_size, // input cols
      params,
      make_unique<SigmoidFunction>(), // FC layer activation
      output_size
  );
  YANN_CHECK(layers);

  auto cnn = make_unique<Network>(std::move(layers));
  YANN_CHECK(cnn);
  cnn->set_cost_function(make_unique<QuadraticCost>());
  {
    Timer timer("Initializing ConvolutionalNetwork");
    cnn->init(Layer::InitMode_Random, Layer::InitContext(12345));
    BOOST_TEST_MESSAGE(timer);
  }

  // training
  Trainer trainer(make_unique<Updater_GradientDescent>(1.0, 0.0));
  DataSource_Stochastic data_source(
      training_inputs,
      training_outputs,
      DataSource_Stochastic::Sequential,
      1);
  BOOST_TEST_MESSAGE("trainer: " << trainer.get_info());
  {
    Timer timer("Training");
    trainer.train(*cnn, data_source, epochs);

    BOOST_TEST_MESSAGE(timer);
  }
  BOOST_TEST_MESSAGE("After training: " << (*cnn));

  // testing
  VectorBatch res;
  for (MatrixSize ii = 0; ii < testing_batch_size; ii++) {
    VectorBatch in = get_batch(testing_inputs, ii); // copy
    VectorBatch out = get_batch(testing_outputs, ii); // copy
    res.resizeLike(out);

    cnn->calculate(in, res);

    Value cost = cnn->cost(res, out);
    BOOST_TEST_MESSAGE("test " << ii << " input: " << in);
    BOOST_TEST_MESSAGE("test " << ii << " expected: " << out);
    BOOST_TEST_MESSAGE("test " << ii << " actual: " << res);
    BOOST_TEST_MESSAGE("test " << ii << " cost/loss: " << cost);

    BOOST_CHECK_LE(cost, 0.06);
  }
}

BOOST_AUTO_TEST_CASE(Mnist_OneLayer_Two_Labels_Test)
{
  const size_t epochs = 3;

  // reduce the test size to two labels to make it faster
  BOOST_TEST_MESSAGE("*** Filtering test set...");
  _mnist_test.filter(1, 1000, 1000); // only allow 0,1 images; 1000 count
  BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  // setup
  auto cnn = create_cnn_for_mnist(
        10, // filter_size
        4, // polling_size,
        2, // frames_num,
        make_unique<QuadraticCost>());
  BOOST_CHECK(cnn);
  cnn->init(Layer::InitMode_Random, Layer::InitContext(12345));

  // create trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_GradientDescent>(1.0, 0.0));
  BOOST_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Training against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" ConvolutionalNetwork: " << cnn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // test
  pair<double, Value> res = _mnist_test.train_and_test(
      *cnn,
      *trainer,
      DataSource_Stochastic::Sequential, // want consistency for this test
      10,                  // training_batch_size
      epochs,
      100                  // testing_batch_size
  );
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" ConvolutionalNetwork: " << cnn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // check
  BOOST_CHECK_GE(res.first, 0.995); // > 995%%
  BOOST_CHECK_LE(res.second, 0.006); // < 0.006 per test
}

BOOST_AUTO_TEST_CASE(Mnist_OneLayer_Full_Test, * disabled())
{
  const size_t epochs = 30;

  // setup
  auto cnn = create_cnn_for_mnist(
        5, // filter_size
        2, // polling_size,
        3, // frames_num,
        make_unique<QuadraticCost>());
  BOOST_CHECK(cnn);
  cnn->init(Layer::InitMode_Random);

  // trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_GradientDescent>(5.0, 0.0));
  BOOST_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Training against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" ConvolutionalNetwork: " << cnn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // test
  pair<double, Value> res = _mnist_test.train_and_test(
      *cnn,
      *trainer,
      DataSource_Stochastic::Random,
      20, // training_batch_size
      epochs,
      100 // testing_batch_size
  );
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" ConvolutionalNetwork: " << cnn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);
}

BOOST_AUTO_TEST_CASE(LeNet1_Two_Labels_Test)
{
  // reduce the test size to two labels to make it faster
  BOOST_TEST_MESSAGE("*** Filtering test set...");
  _mnist_test.filter(1, 1000, 1000); // only allow 0,1 images; 1000 count
  BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  const double alpha = 0.5;
  const double beta = 0.9;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 3;

  auto layers = ConvolutionalNetwork::create_lenet1(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      PollingLayer::PollMode_Avg,
      20, // fc_size
      _mnist_test.get_label_size(),   // output_size
      make_unique<SigmoidFunction>(), // conv activation
      make_unique<SigmoidFunction>(), // poll activation
      make_unique<SigmoidFunction>()  // fc activation
  );
  YANN_CHECK(layers);
  auto nn = make_unique<Network>(std::move(layers));
  nn->init(Layer::InitMode_Random, Layer::InitContext(12345));
  nn->set_cost_function(make_unique<QuadraticCost>());

  // Trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_GradientDescentWithMomentum>(alpha, beta));
  BOOST_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Training against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // train and test
  pair<double, Value> res = _mnist_test.train_and_test(
      *nn,
      *trainer,
      DataSource_Stochastic::Sequential, // want consistency for this test
      training_batch_size,
      epochs,
      100 // testing_batch_size
  );
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100)
                     << "% Loss: " << res.second
                     << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // save_to_file(nn);

  // check
  BOOST_CHECK_GE(res.first, 0.99);  // > 99%%
  BOOST_CHECK_LE(res.second, 0.05); // < 0.05 per test
}

// The best result so far:
//
// Success rate for epoch 100: against training dataset: 98.3633% against test dataset: 98.38%
// Cost/loss per test for epoch 100: against training dataset: 0.122933 against test dataset: 0.122639
// Training time 17385 milliseconds
// Testing against training dataset time 7825.93 milliseconds
// Testing against test dataset time 1340.66 milliseconds
//
//  auto polling_mode = PollingLayer::PollMode_Avg;
//  unique_ptr<ActivationFunction> activation_func = make_unique<SigmoidFunction>();
//  unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>();
//  const MatrixSize fc_size = 100;
//  auto init_mode = Layer::InitMode_Random;
//  const double learning_rate = 0.01;
//  const double regularization = 0.0001;
//  const MatrixSize training_batch_size = 10;
//  const size_t epochs = 100;
//
BOOST_AUTO_TEST_CASE(LeNet1_Full_Test, * disabled())
{
  // reduce the test size to two labels to make it faster
  // BOOST_TEST_MESSAGE("*** Filtering test set...");
  // _mnist_test.filter(9, 10000, 1000); // only allow 0,1 images; 1000 count
  // BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  auto polling_mode = PollingLayer::PollMode_Max;
  unique_ptr<ActivationFunction> conv_activation_func = make_unique<SigmoidFunction>();
  unique_ptr<ActivationFunction> poll_activation_func = make_unique<SigmoidFunction>();
  unique_ptr<ActivationFunction> fc_activation_func = make_unique<SigmoidFunction>();
  unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>();
  const MatrixSize fc_size = 120;
  auto init_mode = Layer::InitMode_Random;
  const double learning_rate = 0.05;
  const double regularization = 0.0001;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 100;

  auto layers = ConvolutionalNetwork::create_lenet1(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      polling_mode,
      fc_size,
      _mnist_test.get_label_size(), // output_size
      conv_activation_func,
      poll_activation_func,
      fc_activation_func
  );
  YANN_CHECK(layers);

  auto nn = make_unique<Network>(std::move(layers));
  YANN_CHECK(nn);
  nn->set_cost_function(cost_func);
  nn->init(init_mode);

  // Trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_GradientDescent>(learning_rate, regularization));
  BOOST_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Training against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // train and test
  pair<double, Value> res = _mnist_test.train_and_test(
      *nn,
      *trainer,
      DataSource_Stochastic::Random,
      training_batch_size,
      epochs,
      100 // testing_batch_size
  );
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100)
                     << "% Loss: " << res.second
                     << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);
}


// The best result so far:
//
// Success rate for epoch 93: against training dataset: 96.4867% against test dataset: 96.14%
// Cost/loss per test for epoch 93: against training dataset: 0.229392 against test dataset: 0.256736
//
// Training time 35861.4 milliseconds
// Testing against training dataset time 17260.3 milliseconds
// Testing against test dataset time 2885.71 milliseconds
//
BOOST_AUTO_TEST_CASE(BoostedLeNet1_Full_Test, * disabled())
{
  // reduce the test size to two labels to make it faster
  // BOOST_TEST_MESSAGE("*** Filtering test set...");
  // _mnist_test.filter(9, 10000, 1000); // only allow 0,1 images; 1000 count
  // BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  auto polling_mode = PollingLayer::PollMode_Max;
  unique_ptr<ActivationFunction> conv_activation_func = make_unique<SigmoidFunction>();
  unique_ptr<ActivationFunction> poll_activation_func = make_unique<SigmoidFunction>();
  unique_ptr<ActivationFunction> fc_activation_func = make_unique<SigmoidFunction>();
  unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>();
  const MatrixSize paths_num = 2;
  const MatrixSize fc_size = 50;
  const auto init_mode = Layer::InitMode_Random;
  const double alpha = 0.5;
  const double beta = 0.9;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 100;

  auto layers = ConvolutionalNetwork::create_boosted_lenet1(
      paths_num,
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      polling_mode,
      fc_size,
      _mnist_test.get_label_size(), // output_size
      conv_activation_func,
      poll_activation_func,
      fc_activation_func
  );
  YANN_CHECK(layers);

  auto nn = make_unique<Network>(std::move(layers));
  YANN_CHECK(nn);
  nn->set_cost_function(cost_func);
  nn->init(init_mode);

  // Trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_GradientDescentWithMomentum>(alpha, beta));
  BOOST_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Training against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // train and test
  pair<double, Value> res = _mnist_test.train_and_test(
      *nn,
      *trainer,
      DataSource_Stochastic::Random,
      training_batch_size,
      epochs,
      100 // testing_batch_size
  );
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);
}

BOOST_AUTO_TEST_CASE(LeNet5_Two_Labels_Test)
{
  // reduce the test size to two labels to make it faster
  BOOST_TEST_MESSAGE("*** Filtering test set...");
  _mnist_test.filter(1, 2000, 1000); // only allow 0,1 images; 1000 count
  BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  const double learning_rate = 0.1;
  const double regularization = 0.0;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 3;

  auto layers = ConvolutionalNetwork::create_lenet5(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      PollingLayer::PollMode_Avg,
      100, // fc1 size
      30, // fc2 size
      _mnist_test.get_label_size(), // output_size
      make_unique<SigmoidFunction>(), // conv activation
      make_unique<SigmoidFunction>(), // poll activation
      make_unique<SigmoidFunction>()  // fc activation
  );
  YANN_CHECK(layers);

  auto nn = make_unique<Network>(std::move(layers));
  YANN_CHECK(nn);
  nn->init(Layer::InitMode_Random, Layer::InitContext(12345));
  nn->set_cost_function(make_unique<CrossEntropyCost>());
  // nn->set_cost_function(make_unique<ExponentialCost>(2.0));

  // Trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_GradientDescent>(learning_rate, regularization));
  BOOST_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Training against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // train and test
  pair<double, Value> res = _mnist_test.train_and_test(
      *nn,
      *trainer,
      DataSource_Stochastic::Sequential,
      training_batch_size,
      epochs,
      100  // testing_batch_size
  );
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100)
                     << "% Loss: " << res.second
                     << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // check
  BOOST_CHECK_GE(res.first, 0.95); // > 95%%
  BOOST_CHECK_LE(res.second, 0.5); // < 0.5 per test
}

// The best result so far:
//
// Success rate for epoch 50: against training dataset: 97.87% against test dataset: 97.61%
// Cost/loss per test for epoch 50: against training dataset: 0.144058 against test dataset: 0.154121
//
// Training time 39157.3 milliseconds
// Testing against training dataset time 17102.2 milliseconds
// Testing against test dataset time 2835.35 milliseconds
//
// unique_ptr<ActivationFunction> activation_func = make_unique<SigmoidFunction>();
// unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>();
// auto polling_mode = PollingLayer::PollMode_Max;
// const MatrixSize fc1_size = 120;
// const MatrixSize fc2_size = 84;
// auto init_mode = Layer::InitMode_Random;
// const double learning_rate = 0.01;
// const double regularization = 0.0001;
// const MatrixSize training_batch_size = 10;
//
BOOST_AUTO_TEST_CASE(LeNet5_Full_Test, * disabled())
{
  unique_ptr<ActivationFunction> conv_activation_func = make_unique<SigmoidFunction>();
  unique_ptr<ActivationFunction> poll_activation_func = make_unique<SigmoidFunction>();
  unique_ptr<ActivationFunction> fc_activation_func = make_unique<SigmoidFunction>();
  unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>();
  auto polling_mode = PollingLayer::PollMode_Max;
  const MatrixSize fc1_size = 120;
  const MatrixSize fc2_size = 84;
  auto init_mode = Layer::InitMode_Random;
  const double alpha = 0.01;
  const double beta = 0.9;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 50;

  auto layers = ConvolutionalNetwork::create_lenet5(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      polling_mode,
      fc1_size,
      fc2_size,
      _mnist_test.get_label_size(), // output_size
      conv_activation_func,
      poll_activation_func,
      fc_activation_func
  );
  YANN_CHECK(layers);

  auto nn = make_unique<Network>(std::move(layers));
  YANN_CHECK(nn);
  nn->init(init_mode);
  nn->set_cost_function(cost_func);

  // Trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_GradientDescentWithMomentum>(alpha, beta));
  BOOST_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Training against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // train and test
  pair<double, Value> res = _mnist_test.train_and_test(
      *nn,
      *trainer,
      DataSource_Stochastic::Random,
      training_batch_size,
      epochs,
      100 // testing_batch_size
  );
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100)
                     << "% Loss: " << res.second
                     << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);
}

BOOST_AUTO_TEST_SUITE_END()
