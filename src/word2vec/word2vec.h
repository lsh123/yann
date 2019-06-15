/*
 * word2vec.h
 *
 */

#ifndef WORD2VEC_H_
#define WORD2VEC_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/optional.hpp>

#include "core/types.h"
#include "core/training.h"
#include "text_parser.h"

// Forward declarations
namespace yann::word2vec {
class Word2Vec;
}; // namespace yann::word2vec

// Overwrites from std:: namespace
namespace std {
ostream& operator<<(ostream & os, const yann::word2vec::Word2Vec & w2v);
istream& operator>>(istream & is, yann::word2vec::Word2Vec & w2v);
}; // namespace std

namespace yann::word2vec {

// Word2Vec
class Word2Vec
{
  friend std::ostream& std::operator<<(std::ostream & os, const Word2Vec & w2v);
  friend std::istream& std::operator>>(std::istream & is, Word2Vec & w2v);

public:
  // TrainingParams
  class TrainingParams {
  public:
    TrainingParams();
  public:
   MatrixSize  _window_size;
   MatrixSize  _dimensions;
   boost::optional<Layer::InitContext> _training_init_context;
   double      _training_sampling_rate;
   double      _learning_rate;
   double      _regularization;
   MatrixSize  _training_batch_size;
   size_t      _epochs;
   Trainer::ProgressCallback _epochs_callback;
   Trainer::ProgressCallback _batch_callback;
  }; // class TrainingParams

public:
  Word2Vec();
  virtual ~Word2Vec();

  static std::unique_ptr<Word2Vec> train_skip_gram(const Text & text, const TrainingParams & params);
  static std::unique_ptr<Word2Vec> train_cbow(const Text & text, const TrainingParams & params);

  bool is_equal(const Word2Vec& other, double tolerance) const;

  inline MatrixSize get_dictionary_size() const { return (MatrixSize)_dictionary.get_size(); }
  inline MatrixSize get_vectors_size() const { return _vectors.rows(); }
  inline MatrixSize get_vectors_dimensions() const { return _vectors.cols(); }

  boost::optional<MatrixSize> find_word(const std::string & word) const;
  boost::optional<const std::string &> find_word(const MatrixSize & word_num) const;

  boost::optional<RefConstVector> find_vector(const std::string & word) const;
  boost::optional<RefConstVector> find_vector(const MatrixSize & word_num) const;

  void save(const std::string & filename) const;
  void load(const std::string & filename);

  Value distance(const MatrixSize & word_num1, const MatrixSize & word_num2) const;
  Value distance(const std::string word1, const std::string & word2) const;

  std::vector<std::pair<const std::string &, Value>> find_closest(const std::string & word, const size_t & num = 5) const;
  std::vector<std::pair<const std::string &, Value>> find_farthest(const std::string & word, const size_t & num = 5) const;


  // cos alpha = A * B / (|A| * |B|)
  static Value cosine(const RefConstVector & v1, const RefConstVector & v2);

private:
  template<typename DataSourceType>
  static std::unique_ptr<Word2Vec> train(const Text & text, const TrainingParams & params, DataSourceType & data_source);

private:
  Dictionary _dictionary;
  Matrix     _vectors;
}; // class Word2Vec

}; // namespace yann::word2vec


#endif /* WORD2VEC_H_ */
