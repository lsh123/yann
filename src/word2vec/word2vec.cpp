/*
 * word2vec.cpp
 *
 */
#include <random>
#include <set>
#include <string>
#include <utility>

#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/functions.h"
#include "layers/contlayer.h"
#include "layers/fclayer.h"
#include "layers/smaxlayer.h"

#include "word2vec.h"

using namespace std;
using namespace boost;
using namespace yann;
using namespace yann::word2vec;

namespace yann::word2vec {
class DataSource_Text : public Trainer::DataSource
{
  typedef vector<pair<MatrixSize, MatrixSize>> TestsVector;

public:
  DataSource_Text(
      const Text & text,
      const MatrixSize & window_size,
      const MatrixSize & batch_size
  ) :
    _text(text),
    _window_size(window_size),
    _batch_size(batch_size),
    _cur_pos(0)
  {
    YANN_CHECK_GT(_batch_size, 0);
    YANN_CHECK_GT(_text.get_dictionary_size(), 0);

    _tests = get_tests(_text, _window_size);
    resize_batch(_inputs_batch, _batch_size, _text.get_dictionary_size());
    resize_batch(_outputs_batch, _batch_size, _text.get_dictionary_size());
  }

  // Trainer::DataSource overwrites
  virtual std::string get_info() const
  {
    ostringstream oss;
    oss << "yann::DataSource_Text("
        << "dictionary_size=" << _text.get_dictionary_size()
        << ", batch_size=" << _batch_size
        << ", window_size=" << _window_size
        << ", tests_num=" << _tests.size()
        << ")";
    return oss.str();
  }
  virtual MatrixSize get_batch_size() const
  {
    return _batch_size;
  }

  virtual MatrixSize get_num_batches() const
  {
    YANN_CHECK(_batch_size > 0);
    return _tests.size() / _batch_size;
  }

  virtual void start_epoch()
  {
    _cur_pos = 0;

    // todo: support sequential mode w/o shuffling too
    auto rng = default_random_engine { };
    shuffle(begin(_tests), end(_tests), rng);
  }

  virtual optional<Batch> get_next_batch()
  {
    YANN_CHECK_GE(_cur_pos, 0);
    auto res = create_batch(_cur_pos);
    if(res <= 0) {
      return boost::none;
    }
    _cur_pos += res;
    return Batch(_inputs_batch, _outputs_batch);
  }
  virtual void end_epoch()
  {
    // do nothing
  }

public:
  double test_all(const Network & nn)
  {
    size_t success_count = 0, total_count = 0;

    VectorBatch outputs;
    outputs.resizeLike(_outputs_batch);
    for(MatrixSize start_pos = 0; ; ) {
      auto res = create_batch(start_pos);
      if(res <= 0) {
        break;
      }
      start_pos += res;

      nn.calculate(_inputs_batch, outputs);
      for(MatrixSize jj = 0; jj < yann::get_batch_size(outputs); ++jj) {
        const auto & actual = get_batch(outputs, jj);
        const auto & expected = get_batch(_outputs_batch, jj);
        MatrixSize aa = 0, ee = 0;
        actual.array().maxCoeff(&aa);
        expected.array().maxCoeff(&ee);
        if(aa == ee) {
          ++success_count;
        }
        ++total_count;
      }
    }

    return total_count > 0 ? success_count / (double)total_count : 0.0;
  }

private:
  MatrixSize create_batch(const MatrixSize & start_pos)
  {
    YANN_CHECK_GE(start_pos, 0);

    _inputs_batch.setZero();
    _outputs_batch.setZero();
    for(MatrixSize ii = 0, pos = start_pos; ii < _batch_size; ++ii, ++pos) {
      if(pos >= (MatrixSize)_tests.size()) {
        return 0;
      }
      const auto & cur =  _tests[pos];
      _inputs_batch.insert(ii, cur.first) = 1.0; // RowMajor
      _outputs_batch(ii, cur.second) = 1.0; // RowMajor
    }
    return _batch_size;
  }

  static TestsVector get_tests(const Text & text, const MatrixSize & window_size)
  {
    TestsVector tests;
    for(MatrixSize ii = 0, size = text.sentences().size(); ii < size; ++ii) {
      const auto & sentence = text.sentences()[ii];
      // sentence[jj] is the "input" word
      for(MatrixSize jj = 0, sentence_size = sentence.size(); jj < sentence_size; ++jj) {
        // sentence[kk] is the "output" word
        for(MatrixSize kk = jj - window_size; kk <= jj + window_size; ++kk) {
          // check if we are in-range and that we are not pointing to the word itself
          if(kk < 0 || kk >= sentence_size || sentence[jj] == sentence[kk]) {
            continue;
          }
          tests.push_back(make_pair(sentence[jj], sentence[kk]));
        }
      }
    }
    return tests;
  }


private:
  const Text & _text;
  const MatrixSize _window_size;
  const MatrixSize _batch_size;

  TestsVector _tests;
  size_t      _cur_pos;

  SparseVectorBatch _inputs_batch;
  VectorBatch _outputs_batch;
}; // class DataSource_Text

}; // namespace yann::word2vec

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Word2Vec::TrainingParams implementation
//
yann::word2vec::Word2Vec::TrainingParams::TrainingParams():
    _window_size(5),
    _dimensions(100),
    _learning_rate(0.9),
    _regularization(0),
    _training_batch_size(100),
    _epochs(10),
    _epochs_callback(nullptr),
    _batch_callback(nullptr)
{
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Word2Vec implementation
//

// the format is (d:<dictionary>,v:<vectors>)
ostream& std::operator<<(ostream & os, const Word2Vec & w2v)
{
  os << "(d:" << w2v._dictionary << ",v:" << w2v._vectors << ")";
  return os;
}

// the format is (d:<dictionary>,v:<vectors>)
istream& std::operator>>(istream & is, Word2Vec & w2v)
{
  char ch;
  if(is >> ch && ch != '(') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> ch && ch != 'd') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> ch && ch != ':') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  is >> w2v._dictionary;
  if(is.fail()) {
    return is;
  }
  if(is >> ch && ch != ',') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> ch && ch != 'v') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> ch && ch != ':') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  is >> w2v._vectors;
  if(is.fail()) {
    return is;
  }
  if(is >> ch && ch != ')') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }

  return is;
}


// cos alpha = A * B / (|A| * |B|)
Value yann::word2vec::Word2Vec::cosine(const RefConstVector & v1, const RefConstVector & v2)
{
  return (v1.array() * v2.array()).sum() / (v1.norm() * v2.norm());
}

yann::word2vec::Word2Vec::Word2Vec()
{
}

yann::word2vec::Word2Vec::~Word2Vec()
{
}

bool yann::word2vec::Word2Vec::is_equal(const Word2Vec& other, double tolerance) const
{
  return _dictionary == other._dictionary && _vectors.isApprox(other._vectors, tolerance);
}

optional<MatrixSize> yann::word2vec::Word2Vec::find_word(const std::string & word) const
{
  return _dictionary.find_word(word);
}
optional<const std::string &> yann::word2vec::Word2Vec::find_word(const MatrixSize & word_num) const
{
  return _dictionary.find_word(word_num);
}

optional<RefConstVector> yann::word2vec::Word2Vec::find_vector(const MatrixSize & word_num) const
{
  YANN_CHECK_LT(word_num, get_vectors_size());
  RefConstVector vv = _vectors.row(word_num);
  return vv;
}

optional<RefConstVector> yann::word2vec::Word2Vec::find_vector(const std::string & word) const
{
  auto word_num = find_word(word);
  if(!word_num) {
    return boost::none;
  }
  return find_vector(*word_num);
}

vector<pair<const string &, Value>> yann::word2vec::Word2Vec::find_closest(const string & word, const size_t & num) const
{
  YANN_CHECK_GT(num, 0);

  auto word_num = find_word(word);
  if(!word_num) {
    return vector<pair<const string &, Value>>(); // not found
  }
  auto vv = find_vector(*word_num);
  YANN_CHECK(vv);

  // compare by the vector distance
  auto cmp = [](const auto & aa, const auto & bb) { return aa.second < bb.second; };
  set<pair<MatrixSize, Value>, decltype(cmp)> ordered(cmp);
  for(MatrixSize ii = 0; ii < _vectors.rows(); ++ii) {
    if(ii == *word_num) continue; // ignore itself

    auto distance = cosine(*vv, _vectors.row(ii));
    ordered.insert(make_pair(ii, distance));
    if(ordered.size() > num) {
      ordered.erase(ordered.begin());
    }
  }

  vector<pair<const string &, Value>> res;
  res.reserve(ordered.size());
  for(const auto & pp : ordered) {
    // can't use make_pair() because of "const string &"
    res.push_back(pair<const string &, Value>(_dictionary.get_word(pp.first), pp.second));
  }
  return res;
}

unique_ptr<Word2Vec> yann::word2vec::Word2Vec::train_skip_gram(const Text & text, const TrainingParams & params)
{
  auto w2v = make_unique<Word2Vec>();
  YANN_CHECK(w2v);

  // create network
  auto nn = make_unique<Network>();
  YANN_CHECK(nn);
  // nn->set_cost_function(make_unique<CrossEntropyCost>(1.0e-10));
  nn->set_cost_function(make_unique<QuadraticCost>());

  auto fc_layer1 = make_unique<FullyConnectedLayer>(
      text.get_dictionary_size(), params._dimensions);
  fc_layer1->set_fixed_bias(0.0);
  fc_layer1->set_activation_function(make_unique<IdentityFunction>());
  fc_layer1->set_sampling_rate(0.0001); // 1%
  nn->append_layer(std::move(fc_layer1));

  auto fc_layer2 = make_unique<FullyConnectedLayer>(
      params._dimensions, text.get_dictionary_size());
  fc_layer2->set_fixed_bias(0.0);
  fc_layer2->set_activation_function(make_unique<IdentityFunction>());
  fc_layer2->set_sampling_rate(0.0001); // 1%
  nn->append_layer(std::move(fc_layer2));

  auto smax_layer = make_unique<SoftmaxLayer>(
      nn->get_output_size());
  YANN_CHECK(smax_layer);
  nn->append_layer(std::move(smax_layer));

  // train the network
  DataSource_Text data_source(
      text,
      params._window_size,
      params._training_batch_size);
  Trainer trainer(make_unique<Updater_GradientDescent>(
      params._learning_rate,
      params._regularization
  ));
  trainer.set_batch_progress_callback(params._batch_callback);

  // TODO: remove
  auto fc_layer = dynamic_cast<const FullyConnectedLayer*>(nn->get_layer(0));
  YANN_CHECK(fc_layer);
  w2v->_dictionary = text.dictionary();

  nn->init(Layer::InitMode_Random); // TODO: support static init context for testing

  // TODO: remove
  w2v->_vectors = fc_layer->get_weights();
  auto closest = w2v->find_closest("dirty", 5);
  DBG(closest);
  auto total_tests = (data_source.get_num_batches() * data_source.get_batch_size());

  for(size_t ii = 0; ii < params._epochs; ++ii) {
    if(params._epochs_callback != nullptr) {
      params._epochs_callback(ii + 1, params._epochs);
    }
    Value cost = trainer.train(*nn, data_source);
    DBG(cost);
    DBG(cost / total_tests);

    double success = data_source.test_all(*nn);
    DBG(success);

    // TODO: remove
    w2v->_vectors = fc_layer->get_weights();
    auto closest = w2v->find_closest("dirty", 5);
    DBG(closest);
  }

  double success = data_source.test_all(*nn);
  DBG(success);

  // extract the weights
  /* TODO: restore
  auto fc_layer = dynamic_cast<const FullyConnectedLayer*>(nn->get_layer(0));
  YANN_CHECK(fc_layer);
  w2v->_vectors = fc_layer->get_weights();
  w2v->_dictionary = text.dictionary();
  */

  // done
  return w2v;
}
