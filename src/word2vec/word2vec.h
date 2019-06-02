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

  bool is_equal(const Word2Vec& other, double tolerance) const;

  inline MatrixSize get_dictionary_size() const { return (MatrixSize)_dictionary.get_size(); }
  inline MatrixSize get_vectors_size() const { return _vectors.rows(); }
  inline MatrixSize get_vectors_dimensions() const { return _vectors.cols(); }

  boost::optional<MatrixSize> find_word(const std::string & word) const;
  boost::optional<const std::string &> find_word(const MatrixSize & word_num) const;

  boost::optional<RefConstVector> find_vector(const std::string & word) const;
  boost::optional<RefConstVector> find_vector(const MatrixSize & word_num) const;


  std::vector<std::pair<const std::string &, Value>> find_closest(const std::string & word, const size_t & num = 1) const;


  // cos alpha = A * B / (|A| * |B|)
  static Value cosine(const RefConstVector & v1, const RefConstVector & v2);

private:
  Dictionary _dictionary;
  Matrix     _vectors;
}; // class Word2Vec

}; // namespace yann::word2vec


#endif /* WORD2VEC_H_ */
