//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "ConvolutionalNetwork Tests"
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "types.h"
#include "functions.h"
#include "utils.h"
#include "random.h"
#include "training.h"
#include "cnn.h"
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

struct CnnTestFixture
{
  static void progress_callback(const MatrixSize & cur_pos, const MatrixSize & step, const MatrixSize & total)
  {
    // we want to print out progress at 1/10 increment
    auto progress_delta = total / 10;
    if(progress_delta == 0 || cur_pos % progress_delta < step) {
      BOOST_TEST_MESSAGE("  ... at " << cur_pos << " out of " << total);
    }
  }

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


    // create network
    auto cnn = ConvolutionalNetwork::create(
        _mnist_test.get_image_rows(),
        _mnist_test.get_image_cols(),
        params,
        make_unique<IdentityFunction>(),
        _mnist_test.get_label_size());
    cnn->set_cost_function(cost);

    // add softmaxLayer
    auto smax_layer = make_unique<SoftmaxLayer>(
        cnn->get_output_size()
    );
    YANN_CHECK(smax_layer);
    cnn->append_layer(std::move(smax_layer));

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
  auto one = ConvolutionalNetwork::create(
      image_size, // input rows,
      image_size, // input cols
      params,
      make_unique<SigmoidFunction>(), // FC layer activation func
      output_size      // output size
  );
  YANN_CHECK(one);
  one->init(InitMode_Random_01);

  // BOOST_TEST_MESSAGE("ConvolutionalNetwork before writing to file: " << (*one));
  ostringstream oss;
  oss << (*one);
  BOOST_CHECK(!oss.fail());

  // read network
  auto two = ConvolutionalNetwork::create(
      image_size, // input rows,
      image_size, // input cols
      params,
      make_unique<SigmoidFunction>(), // FC layer activation func
      output_size      // output size
  );
  YANN_CHECK(two);
  std::istringstream iss(oss.str());
  iss >> (*two);
  BOOST_CHECK(!iss.fail());
  // BOOST_TEST_MESSAGE("ConvolutionalNetwork after loading from file: " << (*two));

  // compare
  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Training_BatchGradientDescent_Test)
{
  BOOST_TEST_MESSAGE("*** ConvolutionalNetwork training test...");

  const size_t epochs = 100;
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
  auto cnn = ConvolutionalNetwork::create(
      image_size, // input rows
      image_size, // input cols
      params,
      make_unique<SigmoidFunction>(), // FC layer activation
      output_size
  );
  YANN_CHECK(cnn);
  {
    Timer timer("Initializing ConvolutionalNetwork");
    cnn->init(InitMode_Zeros); // want consistency for this test
    BOOST_TEST_MESSAGE(timer);
  }

  // training
  Trainer_Batch trainer(
      make_unique<Updater_GradientDescent>(3.0, 0.0),
      Trainer::Random,
      1);
  BOOST_TEST_MESSAGE("trainer: " << trainer.get_info());
  {
    Timer timer("Training");
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
      // BOOST_TEST_MESSAGE("Training epoch " << epoch << " out of " << epochs);
      trainer.train(*cnn, training_inputs, training_outputs);
    }
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

    BOOST_CHECK_LE(cost, 0.7);
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
  cnn->init(InitMode_Random_01);

  // create trainer
  auto trainer = make_unique<Trainer_Stochastic>(
      make_unique<Updater_GradientDescent>(1.0, 0.0),
      Trainer::Sequential, // want consistency for this test
      10     // batch_size
  );
  BOOST_CHECK(trainer);
  trainer->set_progress_callback(progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Testing against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" ConvolutionalNetwork: " << cnn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // test
  pair<double, Value> res = _mnist_test.train_and_test(*cnn, *trainer, epochs, 100);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" ConvolutionalNetwork: " << cnn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // check
  BOOST_CHECK_GE(res.first, 0.95); // > 95%%
  BOOST_CHECK_LE(res.second, 0.05); // < 0.05 per test
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
  cnn->init(InitMode_Random_SqrtInputs);

  // trainer
  auto trainer = make_unique<Trainer_Stochastic>(
      make_unique<Updater_GradientDescent>(5.0, 0.0),
      Trainer::Random,
      20    // batch_size
  );
  BOOST_CHECK(trainer);
  trainer->set_progress_callback(progress_callback);

  // print info
  BOOST_TEST_MESSAGE("*** Testing against MNIST dataset with ");
  BOOST_TEST_MESSAGE(" ConvolutionalNetwork: " << cnn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // test
  pair<double, Value> res = _mnist_test.train_and_test(*cnn, *trainer, epochs, 100);
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

  const double learning_rate = 0.5;
  const double regularization = 0.0;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 5;

  auto nn = ConvolutionalNetwork::create_lenet1(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      PollingLayer::PollMode_Avg,
      20, // fc_size
      _mnist_test.get_label_size(), // output_size
      make_unique<SigmoidFunction>()
  );
  YANN_CHECK(nn);
  nn->init(InitMode_Random_01);
  nn->set_cost_function(make_unique<QuadraticCost>());

  // Trainer
  auto trainer = make_unique<Trainer_Stochastic>(
      make_unique<Updater_GradientDescentWithMomentum>(learning_rate, regularization),
      Trainer::Sequential, // want consistency for this test
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
  pair<double, Value> res = _mnist_test.train_and_test(*nn, *trainer, epochs, 100);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
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
//  unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>(1.0e-10);
//  const MatrixSize fc_size = 100;
//  auto init_mode = InitMode_Random_01;
//  const double learning_rate = 0.01;
//  const double regularization = 0.0001;
//  const MatrixSize training_batch_size = 10;
//  const size_t epochs = 100;
//
//  auto trainer = make_unique<Trainer_Stochastic>(
//      make_unique<Updater_GradientDescent>(learning_rate, regularization),
//      Trainer::Random,
//      training_batch_size  // batch_size
//  );
BOOST_AUTO_TEST_CASE(LeNet1_Full_Test, * disabled())
{
  auto polling_mode = PollingLayer::PollMode_Max;
  unique_ptr<ActivationFunction> activation_func = make_unique<SigmoidFunction>();
  unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>(1.0e-10);
  const MatrixSize fc_size = 120;
  auto init_mode = InitMode_Random_01;
  const double learning_rate = 0.01;
  const double regularization = 0.0001;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 100;

  auto nn = ConvolutionalNetwork::create_lenet1(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      polling_mode,
      fc_size,
      _mnist_test.get_label_size(), // output_size
      activation_func
  );
  YANN_CHECK(nn);
  nn->init(init_mode);
  nn->set_cost_function(cost_func);

  // Trainer
  auto trainer = make_unique<Trainer_Stochastic>(
      make_unique<Updater_GradientDescent>(learning_rate, regularization),
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
  pair<double, Value> res = _mnist_test.train_and_test(*nn, *trainer, epochs, 100);
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
  const size_t epochs = 10;

  auto nn = ConvolutionalNetwork::create_lenet5(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      PollingLayer::PollMode_Avg,
      100, // fc1 size
      30, // fc2 size
      _mnist_test.get_label_size(), // output_size
      make_unique<SigmoidFunction>()
  );
  YANN_CHECK(nn);
  nn->init(InitMode_Random_SqrtInputs);
  nn->set_cost_function(make_unique<CrossEntropyCost>(1.0e-300));
  // nn->set_cost_function(make_unique<ExponentialCost>(2.0));

  // Trainer
  auto trainer = make_unique<Trainer_Stochastic>(
      make_unique<Updater_GradientDescent>(learning_rate, regularization),
      Trainer::Sequential,
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
  pair<double, Value> res = _mnist_test.train_and_test(*nn, *trainer, epochs, 100);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  // check
  BOOST_CHECK_GE(res.first, 0.95); // > 95%%
  BOOST_CHECK_LE(res.second, 0.5); // < 0.5 per test
}

BOOST_AUTO_TEST_CASE(LeNet5_Full_Test, * disabled())
{
  // reduce the test size
  BOOST_TEST_MESSAGE("*** Filtering test set...");
  //_mnist_test.filter(9, 10000, 1000);
  BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  // unique_ptr<ActivationFunction> activation_func = make_unique<ReluFunction>(0);
  // unique_ptr<ActivationFunction> activation_func = make_unique<TanhFunction>(1.7159, 0.6666);
  unique_ptr<ActivationFunction> activation_func = make_unique<SigmoidFunction>();
  unique_ptr<CostFunction> cost_func = make_unique<CrossEntropyCost>(1.0e-10);
  auto polling_mode = PollingLayer::PollMode_Max;
  const MatrixSize fc1_size = 120;
  const MatrixSize fc2_size = 84;
  auto init_mode = InitMode_Random_01;
  const double learning_rate = 0.01;
  const double regularization = 0.0001;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 300;

  auto nn = ConvolutionalNetwork::create_lenet5(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      polling_mode,
      fc1_size,
      fc2_size,
      _mnist_test.get_label_size(), // output_size
      activation_func
  );
  YANN_CHECK(nn);

  // add softmaxLayer
  /*
  auto smax_layer = make_unique<SoftmaxLayer>(
      nn->get_output_size(),
      1000.0 // beta to make max more prominent
  );
  YANN_CHECK(smax_layer);
  nn->append_layer(std::move(smax_layer));
  */

  nn->init(init_mode);
  nn->set_cost_function(cost_func);

  // Trainer
  auto trainer = make_unique<Trainer_Stochastic>(
      make_unique<Updater_GradientDescent>(learning_rate, regularization),
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
  pair<double, Value> res = _mnist_test.train_and_test(*nn, *trainer, epochs, 100);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);
}

BOOST_AUTO_TEST_SUITE_END()
