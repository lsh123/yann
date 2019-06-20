//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "RecurrentNeuralNetwork Tests"

#include <boost/test/unit_test.hpp>

#include <sstream>

#include "core/training.h"
#include "core/utils.h"
#include "core/functions.h"
#include "layers/reclayer.h"
#include "layers/smaxlayer.h"

#include "timer.h"
#include "test_utils.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;


struct RnnTestFixture
{
  RnnTestFixture()
  {
  }
  ~RnnTestFixture()
  {
  }

  const std::vector<std::string> SMALL_FRUITS = {
      "apple", "fig", "grape", "lemon", "lime",
      "melon", "peach", "pear", "plum",
  };

  const std::vector<std::string> FRUITS = {
      "apple", "apricot",
      "banana", "bilberry", "blackberry", "blackcurrant", "blueberry",
      "cantaloupe", "coconut", "currant", "cherry", "cherimoya", "clementine", "cloudberry",
      "damson", "durian",
      "elderberry",
      "fig", "feijoa",
      "gooseberry", "grape", "grapefruit",
      "honeydew", "huckleberry",
      "jackfruit", "jambul", "jujube",
      "kiwifruit", "kumquat",
      "lemon", "lime", "loquat", "lychee",
      "mango", "melon",
      "nectarine",
      "orange",
      "passionfruit", "peach", "pear", "plum", "plumcot", "prune", "pineapple", "pomegranate", "pomelo",
      "raisin", "raspberry", "rambutan", "redcurrant",
      "satsuma", "strawberry",
      "tangerine", "tomato",
      "watermelon",
  };

  class DataSource_Words : public Trainer::DataSource
  {
  public:
    // helpers
    static const int SEPARATOR_CHAR = 0;
    static const int MAX_CHAR = 27; // 26 + SEPARATOR_CHAR;

    inline static MatrixSize from_char(const char & ch)
    {
      return ch - 'a' + 1; // account for SEPARATOR_CHAR
    }
    inline static char to_char(const MatrixSize & ch)
    {
      return ch - 1 + 'a'; // account for SEPARATOR_CHAR
    }
    template<typename T>
    inline static MatrixSize get_char(const T & vv) {
      MatrixSize pos = 0;
      vv.maxCoeff(&pos);
      YANN_CHECK_LT(pos, MAX_CHAR);
      return pos;
    }
    inline static MatrixSize max_word_length(const vector<string> & words)
    {
      size_t res = 0;
      for(const auto & word : words) {
        if(word.length() > res) {
          res = word.length();
        }
      }
      return res;
    }

  public:
    DataSource_Words(const vector<string> & words) :
      _words(words),
      _cur_word(0)
    {
      auto max_batch_size = max_word_length(words) + 1; // SEPARATOR_CHAR at the end
      YANN_CHECK_GT(max_batch_size, 0);

      resize_batch(_inputs_batch, max_batch_size, MAX_CHAR);
      resize_batch(_outputs_batch, max_batch_size, MAX_CHAR);
    }
    virtual ~DataSource_Words()
    {
    }

    // Trainer::DataSource overwrites
    virtual std::string get_info() const
    {
      ostringstream oss;
      oss << "yann::DataSource_Words("
          << "words num=" << _words.size()
          << "batch size=" << get_batch_size()
          << ")";
      return oss.str();
    }
    virtual MatrixSize get_batch_size() const
    {
      return yann::get_batch_size(_inputs_batch);
    }
    virtual MatrixSize get_num_batches() const
    {
      return _words.size();
    }
    virtual MatrixSize get_tests_num() const
    {
      return get_num_batches();
    }
    virtual void start_epoch()
    {
      _cur_word = 0;
    }
    virtual boost::optional<Batch> get_next_batch()
    {
      YANN_CHECK_GE(_cur_word, 0);
      if(_cur_word >= _words.size()) {
        return boost::none;
      }
      auto res = create_batch(_words[_cur_word], _inputs_batch, _outputs_batch);
      ++_cur_word;
      return Batch(_inputs_batch.topRows(res), _outputs_batch.topRows(res));  // RowMajor
    }
    virtual void end_epoch()
    {
      // do nothing
    }

  private:
    static MatrixSize create_batch(const string & word, SparseVectorBatch & inputs, VectorBatch & outputs)
    {
      inputs.setZero();
      outputs.setZero();

      MatrixSize ii = 0, jj = 0;
      inputs.insert(ii++, SEPARATOR_CHAR) = 1.0; // RowMajor
      for(const auto & ch : word) {
        inputs.insert(ii++, from_char(ch)) = 1.0; // RowMajor
        outputs(jj++, from_char(ch)) = 1.0; // RowMajor
      }
      outputs(jj++, SEPARATOR_CHAR) = 1.0; // RowMajor

      YANN_SLOW_CHECK_EQ(ii, jj);
      return ii;
    }

  private:
    const vector<string> _words;
    size_t _cur_word;
    SparseVectorBatch _inputs_batch;
    VectorBatch _outputs_batch;
  }; // class DataSource_Words

  unique_ptr<Network> create_one_layer_rnn(
      const MatrixSize & input_size,
      const MatrixSize & state_size,
      const MatrixSize & output_size,
      const std::unique_ptr<ActivationFunction> & state_activation_function,
      const std::unique_ptr<ActivationFunction> & output_activation_function)
  {
    // create the network
    auto nn = make_unique<Network>();
    YANN_CHECK(nn);

    auto layer = make_unique<RecurrentLayer>(input_size, state_size, output_size);
    BOOST_CHECK(layer);
    layer->set_activation_functions(state_activation_function, output_activation_function);
    nn->append_layer(std::move(layer));

    return nn;
  }

  string get_word(Network & nn, const string & prefix = "")
  {
    ostringstream oss;

    const MatrixSize batch_size = 32;

    SparseVectorBatch input;
    resize_batch(input, 1, DataSource_Words::MAX_CHAR); // one char at a time

    auto ctx = nn.create_context(batch_size);
    YANN_CHECK(ctx);
    ctx->reset_state();

    MatrixSize cur_ch = DataSource_Words::SEPARATOR_CHAR;
    for(int ii = 0; ii < batch_size; ++ii) {
      input.setZero();
      input.insert(0, cur_ch) = 1.0; // RowMajor
      nn.calculate(input, ctx.get());

      if(ii >= (MatrixSize)prefix.size()) {
        cur_ch = DataSource_Words::get_char(get_batch(ctx->get_output(), 0));
        if(cur_ch == DataSource_Words::SEPARATOR_CHAR) {
          break;
        }
      } else {
        cur_ch = DataSource_Words::from_char(prefix[ii]);
      }
      oss << DataSource_Words::to_char(cur_ch);
    }

    // done
    return oss.str();
  }
};
// struct RnnTestFixture

BOOST_FIXTURE_TEST_SUITE(RnnTest, RnnTestFixture);


BOOST_AUTO_TEST_CASE(Training_TwoFruits_Test)
{
  BOOST_TEST_MESSAGE("*** RecurrentNeuralNetwork training test ...");
  const MatrixSize state_size = 50;
  const double learning_rate = 0.9;
  const double regularization = 0.0;
  const size_t epochs = 50;

  DataSource_Words data_source({"lemon", "apple"});

  auto nn = create_one_layer_rnn(
      DataSource_Words::MAX_CHAR, // input_size
      state_size,
      DataSource_Words::MAX_CHAR, // output_size
      make_unique<SigmoidFunction>(),
      make_unique<SigmoidFunction>()
  );
  YANN_CHECK(nn);

  nn->set_cost_function(make_unique<CrossEntropyCost>());
  nn->init(Layer::InitMode_Random, Layer::InitContext(123456));

  // train the network
  Trainer trainer(make_unique<Updater_GradientDescent>(
      learning_rate, regularization));
  // trainer.set_batch_progress_callback(batch_progress_callback);
  // trainer.set_epochs_progress_callback(ecpoch_progress_callback);

  Value cost = trainer.train(*nn, data_source, epochs);
  BOOST_CHECK_LE(cost, 10.0);
  BOOST_CHECK_EQUAL(get_word(*nn), "apple");
  BOOST_CHECK_EQUAL(get_word(*nn, "a"), "apple");
  BOOST_CHECK_EQUAL(get_word(*nn, "l"), "lemon");
}


BOOST_AUTO_TEST_CASE(Training_AllFruits_Test)
{
  BOOST_TEST_MESSAGE("*** RecurrentNeuralNetwork training test ...");
  const MatrixSize state_size = 50;
  const double learning_rate = 0.5;
  const double regularization = 0.0;
  const size_t epochs = 200;

  DataSource_Words data_source(FRUITS);

  auto nn = create_one_layer_rnn(
      DataSource_Words::MAX_CHAR, // input_size
      state_size,
      DataSource_Words::MAX_CHAR, // output_size
      make_unique<SigmoidFunction>(),
      make_unique<SigmoidFunction>()
  );
  YANN_CHECK(nn);

  // auto smax_layer = make_unique<SoftmaxLayer>(nn->get_output_size());
  // nn->append_layer(std::move(smax_layer));

  nn->set_cost_function(make_unique<CrossEntropyCost>());
  nn->init(Layer::InitMode_Random, Layer::InitContext(123456));

  // train the network
  Trainer trainer(make_unique<Updater_GradientDescent>(
      learning_rate, regularization));
  // trainer.set_batch_progress_callback(batch_progress_callback);
  // trainer.set_epochs_progress_callback(ecpoch_progress_callback);

  Value cost = trainer.train(*nn, data_source, epochs);
  BOOST_CHECK_LE(cost, 10.0);
  BOOST_CHECK_EQUAL(get_word(*nn),        "raspberry");
  BOOST_CHECK_EQUAL(get_word(*nn, "a"),   "apricot");
  BOOST_CHECK_EQUAL(get_word(*nn, "b"),   "blueberry");
  BOOST_CHECK_EQUAL(get_word(*nn, "ba"),  "banana");
  BOOST_CHECK_EQUAL(get_word(*nn, "l"),   "loquat");
  BOOST_CHECK_EQUAL(get_word(*nn, "le"),  "lemon");
  BOOST_CHECK_EQUAL(get_word(*nn, "o"),   "orange");
  BOOST_CHECK_EQUAL(get_word(*nn, "to"),  "tomato");
}

BOOST_AUTO_TEST_CASE(Training_AllFruits_Tanh_Test, * disabled())
{
  BOOST_TEST_MESSAGE("*** RecurrentNeuralNetwork training test ...");
  const MatrixSize state_size = 100;
  const double learning_rate = 0.0001;
  const double regularization = 0.0;
  const size_t epochs = 5000;

  DataSource_Words data_source({"apple", "lemon"});

  auto nn = create_one_layer_rnn(
      DataSource_Words::MAX_CHAR, // input_size
      state_size,
      DataSource_Words::MAX_CHAR, // output_size
      make_unique<TanhFunction>(),
      make_unique<IdentityFunction>() // make_unique<SigmoidFunction>()
  );
  YANN_CHECK(nn);

  auto smax_layer = make_unique<SoftmaxLayer>(nn->get_output_size());
  nn->append_layer(std::move(smax_layer));

  nn->set_cost_function(make_unique<CrossEntropyCost>());
  nn->init(Layer::InitMode_Random, Layer::InitContext(123456));

  // train the network
  Trainer trainer(make_unique<Updater_GradientDescent>(
      learning_rate, regularization));
  // trainer.set_batch_progress_callback(batch_progress_callback);
  // trainer.set_epochs_progress_callback(ecpoch_progress_callback);

  for(int ii = 0; ii < 10; ++ ii) {
    DBG(ii);
    Value cost = trainer.train(*nn, data_source, epochs);
    DBG(cost);
    DBG(get_word(*nn, ""));
    DBG(get_word(*nn, "a"));
    DBG(get_word(*nn, "l"));
    DBG(get_word(*nn, "b"));
    DBG(get_word(*nn, "o"));
    DBG(get_word(*nn, "w"));
  }
}

BOOST_AUTO_TEST_SUITE_END()

