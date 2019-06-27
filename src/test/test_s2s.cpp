//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "Seq2Seq Tests"

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <random>
#include <sstream>
#include <vector>
#include <codecvt>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "core/training.h"
#include "core/updaters.h"
#include "core/utils.h"
#include "core/dict.h"
#include "core/functions.h"
#include "layers/s2slayer.h"
#include "layers/fclayer.h"
#include "layers/smaxlayer.h"

#include "timer.h"
#include "test_utils.h"

using namespace std;
using namespace boost;
using namespace boost::iostreams;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

#define RUS_ENG_TEST_PATH    "../data/rus-eng.txt.gz"

// Collection of sentences in a given language
typedef struct _Sentences {
  size_t get_max_sentence_length() const
  {
    size_t res = 0;
    for(const auto & sentence : _sentences) {
      if(res < sentence.size()) {
        res = sentence.size();
      }
    }
    return res;
  }

  size_t get_sentences_size() const
  {
    return _sentences.size();
  }

  size_t get_dict_size() const
  {
    return _dict.get_size();
  }

  const Sentence & get(const size_t & pos) const
  {
    YANN_SLOW_CHECK_LT(pos, _sentences.size());
    return _sentences[pos];
  }

  std::string get_sentence(const Sentence & sentence) const
  {
    ostringstream oss;
    bool write_end_of_word_space = false;
    for(const auto & word_num : sentence) {
      if(write_end_of_word_space) {
        oss << " ";
      }
      oss << _dict.get_word(word_num);
      write_end_of_word_space = true;
    }
    oss << ".";
    return oss.str();
  }

  Sentence get_sentence(const vector<string> & sentence) const
  {
    Sentence res;
    res.reserve(sentence.size());
    for(const auto & word : sentence) {
      res.push_back(_dict.get_word(word));
    }
    return res;
  }

  void shuffle(const vector<size_t> & shuffled_pos)
  {
    YANN_CHECK_EQ(_sentences.size(), shuffled_pos.size());

    vector<Sentence> tmp(_sentences.size());
    for(size_t ii = 0; ii < tmp.size(); ++ii) {
      tmp[ii] = _sentences[shuffled_pos[ii]];
    }
    swap(tmp, _sentences);
  }

  Dictionary _dict;
  vector<Sentence> _sentences;
} Sentences;

class Translator {

public:
  Translator() { }
  virtual ~Translator() { }

  std::string get_info() const
  {
    ostringstream oss;
    oss << "Translator"
        << " dictionay1 size: " << _sentences1.get_dict_size()
        << ", dictionay2 size: " << _sentences2.get_dict_size()
        << ", sentences1 size: " << _sentences1.get_sentences_size()
        << ", sentences2 size: " << _sentences2.get_sentences_size()
    ;
    return oss.str();
  }

  void read(std::istream & is, const size_t & max_count = 0, const char & separator1 = '\t', const char & separator2 = '\n')
  {
    // just in case...
    is >> std::noskipws;

    // read sentence
    while(!is.eof() && !is.fail()) {
      auto sentence1 = read_sentence(is, _sentences1._dict, separator1);
      if(sentence1.empty()) {
        // TODO: check for errors?
        break;
      }
      auto sentence2 = read_sentence(is, _sentences2._dict, separator2);
      if(sentence2.empty()) {
        throw runtime_error("can't read the second sentence");
      }
      _sentences1._sentences.push_back(sentence1);
      _sentences2._sentences.push_back(sentence2);
      if(max_count > 0 && _sentences1._sentences.size() >= max_count) {
        break;
      }
    }
    // done
    YANN_CHECK_EQ(_sentences1._sentences.size(), _sentences2._sentences.size());
  }

  void shuffle()
  {
    vector<size_t> shuffled_pos(get_size());
    for(size_t ii = 0; ii < shuffled_pos.size(); ++ii) {
      shuffled_pos[ii] = ii;
    }
    auto rng = default_random_engine { };
    std::shuffle(shuffled_pos.begin(), shuffled_pos.end(), rng);

    _sentences1.shuffle(shuffled_pos);
    _sentences2.shuffle(shuffled_pos);
  }

  size_t get_dict1_size() const
  {
    return _sentences1.get_dict_size();
  }

  size_t get_dict2_size() const
  {
    return _sentences2.get_dict_size();
  }

  size_t get_size() const
  {
    YANN_CHECK_EQ(_sentences1._sentences.size(), _sentences2._sentences.size());

    return _sentences1._sentences.size();
  }

  size_t get_max_sentence_length() const
  {
    auto len1 = _sentences1.get_max_sentence_length();
    auto len2 = _sentences2.get_max_sentence_length();
    return max(len1, len2);
  }

  const Sentence & get1(const size_t & pos) const
  {
    return _sentences1.get(pos);
  }

  const Sentence & get2(const size_t & pos) const
  {
    return _sentences2.get(pos);
  }

  std::string get_sentence1(const Sentence & sentence) const
  {
    return _sentences1.get_sentence(sentence);
  }

  std::string get_sentence2(const Sentence & sentence) const
  {
    return _sentences2.get_sentence(sentence);
  }

  Sentence get_sentence1(const vector<string> & sentence) const
  {
    return _sentences1.get_sentence(sentence);
  }

  Sentence get_sentence2(const vector<string> & sentence) const
  {
    return _sentences2.get_sentence(sentence);
  }

  static unique_ptr<Translator> parse_gzip_file(const string & filename, const size_t & max_count = 0)
  {
    ifstream file(filename, ios_base::in | ios_base::binary);
    filtering_streambuf<input> inbuf;
    inbuf.push(gzip_decompressor());
    inbuf.push(file);
    istream is(&inbuf);

    // parse text
    auto translator = make_unique<Translator>();
    BOOST_CHECK(translator);
    {
      Timer timer("Parsing file");
      translator->read(is, max_count);
    }
    BOOST_TEST_MESSAGE("Parsed file: " << translator->get_info());

    return translator;
  }

private:
  static Sentence read_sentence(std::istream & is, Dictionary & dict, const char & separator)
  {
    YANN_CHECK(!isalnum(separator)); // separator can't be isalnum()

    char ch;
    string word;
    Sentence res;
    while(!is.eof() && !is.fail() && is >> ch) {
      // we can have utf8 chars
      if(isalnum(ch) || ch < 0) {
        word += tolower(ch);
      } else if(!word.empty()) {
        res.push_back(dict.add_word(word));
        word.clear();
      }
      if(ch == separator) {
        break;
      }
    }

    return res;
  }

private:
  Sentences _sentences1;
  Sentences _sentences2;
}; // Translator

class DataSource_Translator : public Trainer::DataSource
{
public:
  DataSource_Translator(const Translator & translator, const size_t & start_pos, const size_t & size) :
    _translator(translator),
    _start_pos(start_pos),
    _size(size),
    _cur_pos(start_pos)
  {
    YANN_CHECK_LT(_start_pos, _translator.get_size());
    YANN_CHECK_LE(_start_pos + _size, _translator.get_size());
    YANN_CHECK_GT(_size, 0);
    YANN_CHECK_GT(_translator.get_size(), 0);
    YANN_CHECK_GT(_translator.get_dict1_size(), 0);
    YANN_CHECK_GT(_translator.get_dict2_size(), 0);

    auto max_batch_size = _translator.get_max_sentence_length() + 1; // SEPARATOR at the end
    YANN_CHECK_GT(max_batch_size, 0);

    resize_batch(_inputs_batch, max_batch_size, _translator.get_dict1_size());
    resize_batch(_outputs_batch, max_batch_size, _translator.get_dict2_size());
  }
  virtual ~DataSource_Translator()
  {
  }

  // Trainer::DataSource overwrites
  virtual std::string get_info() const
  {
    ostringstream oss;
    oss << "yann::DataSource_Translator("
        << "translator=" << _translator.get_info()
        << ", start_pos=" << _start_pos
        << ", size=" << _size
        << ", batch size=" << get_batch_size()
        << ")";
    return oss.str();
  }
  virtual MatrixSize get_batch_size() const
  {
    YANN_SLOW_CHECK_EQ(yann::get_batch_size(_inputs_batch), yann::get_batch_size(_outputs_batch));
    return yann::get_batch_size(_inputs_batch);
  }
  virtual MatrixSize get_num_batches() const
  {
    return _size;
  }
  virtual void start_epoch()
  {
    _cur_pos = 0;
  }
  virtual boost::optional<Batch> get_next_batch()
  {
    YANN_CHECK_GE(_cur_pos, 0);
    if(_cur_pos >= _start_pos + _size) {
      return boost::none;
    }

    auto res1 = convert(_translator.get1(_cur_pos), _inputs_batch);
    auto res2 = convert(_translator.get2(_cur_pos), _outputs_batch);
    auto res = max(res1, res2) + 1;
    YANN_CHECK(res > 0);
    ++_cur_pos;

//    return Batch(_inputs_batch.topRows(res), _outputs_batch.topRows(res), 1);  // RowMajor, the whole batch is one test
    return Batch(_inputs_batch, _outputs_batch, 1);  // RowMajor, the whole batch is one test
  }
  virtual void end_epoch()
  {
    // do nothing
  }

  static MatrixSize convert(const Sentence & sentence, SparseVectorBatch & buffer)
  {
    YANN_SLOW_CHECK_LE((MatrixSize)sentence.size() + 1, yann::get_batch_size(buffer));

    buffer.setZero();
    MatrixSize ii = 0;
    for(const auto & word_num : sentence) {
      YANN_SLOW_CHECK_LT(word_num,  yann::get_batch_item_size(buffer));
      buffer.insert(ii++, word_num) = 1.0; // RowMajor
    }
    return ii;
  }

  static MatrixSize convert(const Sentence & sentence, VectorBatch & buffer)
  {
    YANN_SLOW_CHECK_LE((MatrixSize)sentence.size() + 1, yann::get_batch_size(buffer));

    buffer.setZero();
    MatrixSize ii = 0;
    for(const auto & word_num : sentence) {
      YANN_SLOW_CHECK_LT(word_num,  yann::get_batch_item_size(buffer));
      buffer(ii++, word_num) = 1.0; // RowMajor
    }
    return ii;
  }

  static Sentence get_sentence(const RefConstVectorBatch & vectors, const Value & epsilon)
  {
    Sentence res;
    res.reserve(yann::get_batch_size(vectors));
    for(MatrixSize ii = 0; ii < yann::get_batch_size(vectors); ++ii) {
      MatrixSize pos = 0;
      if(get_batch(vectors, ii).maxCoeff(&pos) < epsilon) {
        // stop if we have zero's vector
        break;
      }
      res.push_back(pos);
    }

    return res;
  }

  static Sentence get_sentence(const RefConstSparseVectorBatch & vectors, const Value & epsilon)
  {
    Sentence res;
    res.reserve(yann::get_batch_size(vectors));
    for(MatrixSize ii = 0; ii < yann::get_batch_size(vectors); ++ii) {
      MatrixSize pos = -1;
      Value val = epsilon;
      for(MatrixSize jj = 0; jj < yann::get_batch_item_size(vectors); ++jj) {
        if(vectors.coeff(ii, jj) > val) {
          val = vectors.coeff(ii, jj);
          pos = jj;
        }
      }
      if(pos < 0) {
        // stop if we have zero's vector
        break;
      }
      res.push_back(pos);
    }

    return res;
  }

private:
  const Translator & _translator;
  const size_t    _start_pos;
  const size_t    _size;
  SparseVectorBatch _inputs_batch;
  VectorBatch _outputs_batch;

  size_t _cur_pos;
}; // class DataSource_Translator

struct S2STestFixture
{
  S2STestFixture()
  {
  }
  ~S2STestFixture()
  {
  }

  unique_ptr<Network> create_s2s_net(
      const Translator & translator,
      const MatrixSize & encoder_size,
      const MatrixSize & decoder_size)
  {
    // create the network
    auto nn = make_unique<Network>();
    YANN_CHECK(nn);

    // encoder embeddings layer
    auto fc1_layer = make_unique<FullyConnectedLayer>(
        translator.get_dict1_size(), encoder_size);
    YANN_CHECK(fc1_layer);
    fc1_layer->set_activation_function(make_unique<TanhFunction>());
    nn->append_layer(std::move(fc1_layer));

    // encoder/decoder
    auto lstm_layer = Seq2SeqLayer::create_lstm(
        encoder_size,
        decoder_size,
        make_unique<SigmoidFunction>(),
        make_unique<TanhFunction>());
    BOOST_CHECK(lstm_layer);
    nn->append_layer(std::move(lstm_layer));

    // decoder embeddings layer
    auto fc2_layer = make_unique<FullyConnectedLayer>(
        decoder_size, translator.get_dict2_size());
    YANN_CHECK(fc2_layer);
    fc2_layer->set_activation_function(make_unique<SigmoidFunction>());
    nn->append_layer(std::move(fc2_layer));

    // add softmax
    /*
    auto smax_layer = make_unique<SoftmaxLayer>(
        translator.get_dict2_size()
    );
    YANN_CHECK(smax_layer);
    nn->append_layer(std::move(smax_layer));
    */

    // done
    nn->set_cost_function(make_unique<CrossEntropyCost>());
    //nn->set_cost_function(make_unique<QuadraticCost>());


    return nn;
  }

  pair<double, Value> test(
      const Translator & translator,
      Network & nn,
      const size_t & tests_start,
      const size_t & tests_size,
      const double & epsilon,
      bool output_results = false)
  {
    DataSource_Translator data_source(translator, tests_start, tests_size);

    unique_ptr<Context> ctx = nn.create_context(data_source.get_batch_size());
    YANN_CHECK(ctx);

    Value total_cost = 0;
    size_t total_count = 0;
    size_t success_words_count = 0;
    size_t total_words_count = 0;
    // ensure we don't do allocations in eigen
    {
      BlockAllocations block;

      data_source.start_epoch();
      while(true) {
        auto batch = data_source.get_next_batch();
        if(!batch) {
          break;
        }

        ctx->reset_state();
        const auto outputs = batch->_outputs;
        string input_sentence;
        if(batch->_inputs) {
          const auto inputs = *(batch->_inputs);
          if(output_results) {
            input_sentence = translator.get_sentence1(
                 DataSource_Translator::get_sentence(inputs, epsilon));
          }
          nn.calculate(inputs, ctx.get());
        } else if(batch->_sparse_inputs) {
          const auto inputs = *(batch->_sparse_inputs);
          if(output_results) {
            input_sentence = translator.get_sentence1(
                 DataSource_Translator::get_sentence(inputs, epsilon));
          }
          nn.calculate(inputs, ctx.get());
        } else {
          YANN_CHECK("we can't be here" == nullptr);
        }

        auto cost = nn.cost(ctx->get_output(), outputs);
        total_cost += cost;
        ++total_count;

        auto sentence_actual = DataSource_Translator::get_sentence(
            ctx->get_output(),
            epsilon);
        auto sentence_expected = DataSource_Translator::get_sentence(
            outputs,
            epsilon);
        for(size_t ii = 0; ii < sentence_expected.size(); ++ii) {
          if(ii < sentence_actual.size() && sentence_actual[ii] == sentence_expected[ii]) {
            ++success_words_count;
          }
          ++total_words_count;
        }

        if(output_results) {
          cout << input_sentence << "\t"
               << translator.get_sentence2(sentence_expected) << "\t"
               << translator.get_sentence2(sentence_actual) << "\t"
               << cost << endl;
        }
      }
    }

    return total_words_count > 0 && total_count > 0 ?
        make_pair(success_words_count / (double)total_words_count, total_cost / (double)total_count) :
        make_pair(0.0, 0.0);
  }

  pair<double, Value> train_and_test(
      const Translator & translator,
      Network & nn,
      Trainer & trainer,
      const size_t & training_size,
      const size_t & test_size,
      const size_t & training_tests_num,
      const size_t & epochs,
      const double & epsilon)
  {
    YANN_CHECK_GT(epochs, 0);

    // training
    // double training_success_rate = 0;
    // Value training_cost = 0;
    double test_success_rate = 0;
    Value test_cost = 0;
    auto epochs_callback = [&](const MatrixSize & epoch, const MatrixSize & epochs, const std::string & message) {
      /*
      {
        Timer timer("Testing against training dataset");
        pair<double, Value> res = test(translator, nn, 0, training_size, epsilon);
        training_success_rate = res.first;
        training_cost = res.second;
        BOOST_TEST_MESSAGE(timer);
      }
      */
      {
        Timer timer("Testing against test dataset");
        pair<double, Value> res = test(translator, nn, training_size, test_size, epsilon);
        test_success_rate = res.first;
        test_cost = res.second;
        BOOST_TEST_MESSAGE(timer);
      }

      string extra = (epoch <= 0) ? " (before training)" : "";
      BOOST_TEST_MESSAGE(
          "Success rate for epoch " << epoch << extra << ":"
              // << " against training dataset: " << training_success_rate * 100 << "%"
              << " against test dataset: " << test_success_rate* 100 << "%"
      );
      BOOST_TEST_MESSAGE(
          "Cost per test for epoch " << epoch << extra << ":"
              // << " against training dataset: " << training_cost
              << " against test dataset: " << test_cost
      );

      // TODO: remove
      if(epoch % 2 == 0) {
        test_sentences(translator, nn);
      }

      save_to_file(nn, "nn");
      BOOST_TEST_MESSAGE("\n");

      if(epoch + 1 < epochs) {
        if(!message.empty()) {
          BOOST_TEST_MESSAGE("*** Training epoch " << (epoch + 1) << " out of " << epochs << " (" << message << ")");
        } else {
          BOOST_TEST_MESSAGE("*** Training epoch " << (epoch + 1) << " out of " << epochs);
        }
      }
    };

    // test ones before training
    BOOST_TEST_MESSAGE("\n");
    BOOST_TEST_MESSAGE("*** Testing before training");
    epochs_callback(0, epochs, "");

    // train
    {
      Timer timer("Total training/testing");
      DataSource_Translator data_source(translator, 0, training_size);
      trainer.set_epochs_progress_callback(epochs_callback);
      trainer.train(nn, data_source, training_tests_num, epochs);

      BOOST_TEST_MESSAGE(timer << "\n");
    }

    // print results
    test(translator, nn, training_size, test_size, epsilon, true);

    // done
    return make_pair(test_success_rate, test_cost);
  }

  string translate(
      const Translator & translator,
      const Network & nn,
      const vector<string> & sentence,
      const Value & epsilon)
  {
    try {
      // step 1: convert sentence into integers
      auto sentence1 = translator.get_sentence1(sentence);

      // step 2: prep inputs
      auto max_batch_size = translator.get_max_sentence_length();
      max_batch_size = max(max_batch_size, sentence1.size()) + 1;

      SparseVectorBatch inputs;
      resize_batch(inputs, max_batch_size, translator.get_dict1_size());
      DataSource_Translator::convert(sentence1, inputs);

      // step 3: calculate output
      unique_ptr<Context> ctx = nn.create_context(max_batch_size);
      YANN_CHECK(ctx);
      nn.calculate(inputs, ctx.get());
      // DBG(ctx->get_output());

      // step 4: convert output to sentence
      auto sentence2 = DataSource_Translator::get_sentence(ctx->get_output(), epsilon);
      return translator.get_sentence2(sentence2);
    } catch(std::exception & ex) {
      return "CANT TRANSLATE";
    }
  }


  void test_sentences(
      const Translator & translator,
      const Network & nn,
      const double & epsilon =  0.0001)
  {
    DBG(translate(translator, nn, { "go", "away" }, epsilon));
    DBG(translate(translator, nn, { "help", "me" }, epsilon));
    DBG(translate(translator, nn, { "i", "am", "going", "to", "play", "soccer", "after", "school" }, epsilon));
    DBG(translate(translator, nn, { "i", "am", "going", "to", "play", "basketball", "after", "school" }, epsilon));
  }

};
// struct S2STestFixture

BOOST_FIXTURE_TEST_SUITE(S2STest, S2STestFixture);

BOOST_AUTO_TEST_CASE(Training_500_Test, * disabled())
{
  const size_t encdec_size = 100;
  const size_t training_size = 500;
  const size_t test_size = 100;
  const size_t total_size = training_size + test_size;
  const size_t training_tests_num = 10;
  const size_t epochs = 100;
  const double epsilon = 0.001;

  auto translator = Translator::parse_gzip_file(RUS_ENG_TEST_PATH, total_size);
  YANN_CHECK(translator);
  translator->shuffle();

  auto nn = create_s2s_net(*translator, encdec_size, encdec_size);
  YANN_CHECK(nn);
  nn->init(Layer::InitMode_Random, Layer::InitContext(123456));

  // create trainer
  auto trainer = make_unique<Trainer>(make_unique<Updater_AdaDelta>());
  YANN_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  auto res = train_and_test(
      *translator,
      *nn,
      *trainer,
      training_size,
      test_size,
      training_tests_num,
      epochs,
      epsilon
  );

  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100)
                     << "% Loss: " << res.second
                     << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" network: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" translator: " << translator->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  test_sentences(*translator, *nn, epsilon);
}

BOOST_AUTO_TEST_CASE(Training_10000_Test, * disabled())
{
  const size_t encdec_size = 100;
  const size_t training_size = 10000;
  const size_t test_size = 1000;
  const size_t total_size = training_size + test_size;
  const size_t training_tests_num = 10;
  const size_t epochs = 100;
  const double epsilon = 0.001;

  auto translator = Translator::parse_gzip_file(RUS_ENG_TEST_PATH, total_size);
  YANN_CHECK(translator);
  translator->shuffle();

  auto nn = create_s2s_net(*translator, encdec_size, encdec_size);
  YANN_CHECK(nn);
  nn->init(Layer::InitMode_Random, Layer::InitContext(123456));

  // create trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_RMSprop>());
  YANN_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  auto res = train_and_test(
      *translator,
      *nn,
      *trainer,
      training_size,
      test_size,
      training_tests_num,
      epochs,
      epsilon
  );

  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100)
                     << "% Loss: " << res.second
                     << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" network: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" translator: " << translator->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  test_sentences(*translator, *nn, epsilon);
}

BOOST_AUTO_TEST_CASE(Training_100000_Test, * disabled())
{
  const size_t encdec_size = 100;
  const size_t training_size = 100000;
  const size_t test_size = 1000;
  const size_t total_size = training_size + test_size;
  const size_t training_tests_num = 10;
  const size_t epochs = 100;
  const double epsilon = 0.01;

  auto translator = Translator::parse_gzip_file(RUS_ENG_TEST_PATH, total_size);
  YANN_CHECK(translator);
  translator->shuffle();

  auto nn = create_s2s_net(*translator, encdec_size, encdec_size);
  YANN_CHECK(nn);
  nn->init(Layer::InitMode_Random, Layer::InitContext(123456));

  // create trainer
  auto trainer = make_unique<Trainer>(
      make_unique<Updater_RMSprop>());
  YANN_CHECK(trainer);
  trainer->set_batch_progress_callback(batch_progress_callback);

  auto res = train_and_test(
      *translator,
      *nn,
      *trainer,
      training_size,
      test_size,
      training_tests_num,
      epochs,
      epsilon
  );

  BOOST_TEST_MESSAGE("*** Success rate: " << (res.first * 100)
                     << "% Loss: " << res.second
                     << " after " << epochs << " epochs");
  BOOST_TEST_MESSAGE(" network: " << nn->get_info());
  BOOST_TEST_MESSAGE(" trainer: " << trainer->get_info());
  BOOST_TEST_MESSAGE(" translator: " << translator->get_info());
  BOOST_TEST_MESSAGE(" epochs: " << epochs);

  test_sentences(*translator, *nn, epsilon);
}


BOOST_AUTO_TEST_SUITE_END()

