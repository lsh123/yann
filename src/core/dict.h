/*
 * dict.h
 *
 */

#ifndef DICT_H
#define DICT_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/optional.hpp>

// Forward declarations
namespace yann {
class Dictionary;
}; // namespace yann

// Overwrites from std:: namespace
namespace std {
ostream& operator<<(ostream & os, const yann::Dictionary & dictionary);
istream& operator>>(istream & is, yann::Dictionary & dictionary);
}; // namespace std


namespace yann {

// Mapping word <-> number.
class Dictionary {
  typedef std::unordered_map<std::string, MatrixSize> Word2Num;
  typedef std::unordered_map<MatrixSize, std::string> Num2Word;

  friend std::ostream& std::operator<<(std::ostream & os, const Dictionary & dictionary);
  friend std::istream& std::operator>>(std::istream & is, Dictionary & dictionary);

public:
  Dictionary();
  virtual ~Dictionary();

  bool operator==(const Dictionary & other) const;

  bool empty() const;
  size_t get_size() const;
  MatrixSize add_word(const std::string & word, bool fail_if_duplicate = false);

  bool has_word(const MatrixSize & num) const;
  boost::optional<MatrixSize> find_word(const std::string & word) const;
  boost::optional<const std::string &> find_word(const MatrixSize & num) const;

  void erase(const MatrixSize & deleted);
  void erase(const std::unordered_set<MatrixSize> & deleted);
  std::unordered_map<MatrixSize, MatrixSize> compact();

  // throws is word is not found
  MatrixSize get_word(const std::string & word) const;
  const std::string & get_word(const MatrixSize & num) const;

private:
  MatrixSize add_word(const std::string & word, const MatrixSize & num, bool fail_if_duplicate = false);

private:
  Word2Num _w2n;
  Num2Word _n2w;
}; // Dictionary

// A single sentence represented by word's numbers.
typedef std::vector<MatrixSize> Sentence;

}; // namespace yann


#endif /* DICT_H */
