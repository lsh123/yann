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
#include "nntraining.h"
#include "convnn.h"
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
    BOOST_VERIFY(smax_layer);
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
  BOOST_VERIFY(one);
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
  BOOST_VERIFY(two);
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
  BOOST_VERIFY(cnn);
  {
    Timer timer("Initializing ConvolutionalNetwork");
    cnn->init(InitMode_Zeros); // want consistency for this test
    BOOST_TEST_MESSAGE(timer);
  }

  // training
  Trainer_BatchGradientDescent trainer(3, 0.0, Trainer::Random, 1);
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

    BOOST_CHECK_LE(cost, 0.01);
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
        8, // filter_size
        4, // polling_size,
        2, // frames_num,
        make_unique<QuadraticCost>());
  BOOST_CHECK(cnn);
  cnn->init(InitMode_Zeros); // want consistency for this test

  // create trainer
  auto trainer = make_unique<Trainer_StochasticGradientDescent>(
      5.0,    // learning rate
      0.001,  // regularization
      Trainer::Sequential, // want consistency for this test
      100     // batch_size
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
  BOOST_CHECK_GE(res.first, 0.995); // > 99.5%%
  BOOST_CHECK_LE(res.second, 0.02); // < 0.02 per test
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
  auto trainer = make_unique<Trainer_StochasticGradientDescent>(
      5.0,  // learning rate
      0,    // regularization
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

BOOST_AUTO_TEST_CASE(LeNet4_Two_Labels_Test)
{
  // reduce the test size to two labels to make it faster
  BOOST_TEST_MESSAGE("*** Filtering test set...");
  _mnist_test.filter(1, 1000, 1000); // only allow 0,1 images; 1000 count
  BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  const double learning_rate = 1.0;
  const double regularization = 0.0;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 10;

  auto nn = ConvolutionalNetwork::create_lenet4(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      20, // fc_size
      _mnist_test.get_label_size(), // output_size
      make_unique<SigmoidFunction>(),
      make_unique<QuadraticCost>()
  );
  BOOST_VERIFY(nn);
  nn->init(InitMode_Zeros); // want consistency for this test

  // Trainer
  auto trainer = make_unique<Trainer_StochasticGradientDescent>(
      learning_rate,       // learning rate
      regularization,      // regularization
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
  BOOST_CHECK_GE(res.first, 0.995); // > 995%%
  BOOST_CHECK_LE(res.second, 0.05); // < 0.05 per test
}

BOOST_AUTO_TEST_CASE(LeNet4_Full_Test, * disabled())
{
  const double learning_rate = 0.01;
  const double regularization = 0.0001;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 30;

  auto nn = ConvolutionalNetwork::create_lenet4(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      100, // fc_size
      _mnist_test.get_label_size(), // output_size
      make_unique<SigmoidFunction>(),
      make_unique<HellingerDistanceCost>()
  );
  BOOST_VERIFY(nn);
  nn->init(InitMode_Random_SqrtInputs);

  // Trainer
  auto trainer = make_unique<Trainer_StochasticGradientDescent>(
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
  pair<double, Value> res = _mnist_test.train_and_test(*nn, *trainer, epochs, 100);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);
}

BOOST_AUTO_TEST_CASE(LeNet5_Two_Labels_Test, * disabled())
{
  // reduce the test size to two labels to make it faster
  BOOST_TEST_MESSAGE("*** Filtering test set...");
  _mnist_test.filter(1, 2000, 1000); // only allow 0,1 images; 1000 count
  _mnist_test.shift_values(-1.0, 1.0);
  BOOST_TEST_MESSAGE("*** Filtered test set: " << "\n" << _mnist_test);

  const double learning_rate = 0.01;
  const double regularization = 0.0;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 100;

  auto nn = ConvolutionalNetwork::create_lenet5(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      20, // fc1 size
      15, // fc2 size
      _mnist_test.get_label_size(), // output_size
      //make_unique<TanhFunction>(1.7159, 0.6666),
      make_unique<SigmoidFunction>(),
      //make_unique<ReluFunction>(0.1),
      // make_unique<HellingerDistanceCost>(0.00000000000000001)
      make_unique<CrossEntropyCost>()
  );
  BOOST_VERIFY(nn);
  // nn->init(InitMode_Random_SqrtInputs);
  nn->init(InitMode_Random_SqrtInputs);

  // Trainer
  auto trainer = make_unique<Trainer_StochasticGradientDescent>(
      learning_rate,      // learning rate
      regularization,     // regularization
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
  BOOST_CHECK_GE(res.first, 0.995); // > 995%%
  BOOST_CHECK_LE(res.second, 0.05); // < 0.05 per test
}

BOOST_AUTO_TEST_CASE(LeNet5_Full_Test, * disabled())
{
  const double learning_rate = 0.01;
  const double regularization = 0.0001;
  const MatrixSize training_batch_size = 10;
  const size_t epochs = 300;

  auto nn = ConvolutionalNetwork::create_lenet5(
      _mnist_test.get_image_rows(), // input_rows
      _mnist_test.get_image_cols(), // input_cols
      120, // fc1 size
      84,  // fc2 size
      _mnist_test.get_label_size(), // output_size
      make_unique<TanhFunction>(),
      make_unique<HellingerDistanceCost>()
  );
  BOOST_VERIFY(nn);
  nn->init(InitMode_Random_SqrtInputs);

  // Trainer
  auto trainer = make_unique<Trainer_StochasticGradientDescent>(
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
  pair<double, Value> res = _mnist_test.train_and_test(*nn, *trainer, epochs, 100);
  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100) << "% Loss: " << res.second << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" CustomlNetwork: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);
}

BOOST_AUTO_TEST_SUITE_END()
