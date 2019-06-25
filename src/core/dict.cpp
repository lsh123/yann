/*
 * dict.cpp
 *
 */
#include <boost/assert.hpp>

#include "core/utils.h"

#include "dict.h"

using namespace std;
using namespace boost;
using namespace yann;


////////////////////////////////////////////////////////////////////////////////////////////////
//
// Dictionary implementation
//

// format: [<num>](<word0>:<num0>,<word1>:<num1>,...)
std::ostream& std::operator<<(std::ostream & os, const Dictionary & dictionary)
{
  os << "[" << dictionary._w2n.size() << "](";
  bool write_comma = false;
  for(const auto & word_num : dictionary._w2n) {
    if(write_comma) {
      os << ",";
    }
    os << word_num.first << ":" << word_num.second;
    write_comma = true;
  }
  os << ")";
  return os;
}

// format: [<num>](<word0>:<num0>,<word1>:<num1>,...)
std::istream& std::operator>>(std::istream & is, Dictionary & dictionary)
{
  char ch;
  size_t num;
  if(is >> ch && ch != '[') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  is >> num;
  if(is.fail()) {
    return is;
  }
  if(is >> ch && ch != ']') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }

  if(is >> ch && ch != '(') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }

  while(!is.fail()) {
    // TODO: optimize reading the string
    string word;
    MatrixSize num;
    while(is >> ch && ch != ':') {
      word += ch;
    }
    if(is.fail()) {
      return is;
    }
    is >> num;
    if(is.fail()) {
      return is;
    }

    dictionary.add_word(word, num, true); // fail if duplicate
    if(is >> ch && ch != ',' && ch != ')') {
      is.putback(ch);
      is.setstate(std::ios_base::failbit);
      return is;
    }
    if(ch == ')') {
      break;
    }
  }
  return is;
}

yann::Dictionary::Dictionary()
{
}

yann::Dictionary::~Dictionary()
{
}


bool yann::Dictionary::operator==(const Dictionary & other) const
{
  return _n2w == other._n2w && _w2n == other._w2n;
}

bool yann::Dictionary::empty() const
{
  YANN_CHECK_EQ(_n2w.empty(), _w2n.empty());
  return _n2w.empty();
}

size_t yann::Dictionary::get_size() const
{
  YANN_CHECK_EQ(_n2w.size(), _w2n.size());
  return _n2w.size();
}

MatrixSize yann::Dictionary::add_word(const std::string & word, const MatrixSize & num, bool fail_if_duplicate)
{
  // try to insert into dictionary
  auto it_w2n = _w2n.find(word);
  if(it_w2n != _w2n.end()) {
    if(fail_if_duplicate) {
      // word is already in the dic
      throw runtime_error("word '" + word + "' is already in the dictionary");
    }
    return it_w2n->second;
  }
  auto it_n2w = _n2w.find(num);
  if(it_n2w != _n2w.end()) {
    throw runtime_error("word '" + word + "' was not found in w2n but found in n2w: invalid dictionary");
  }

  // add the word
  _n2w.insert(it_n2w, make_pair(num, word));
  _w2n.insert(it_w2n, make_pair(word, num));
  return num;
}

MatrixSize yann::Dictionary::add_word(const std::string & word, bool fail_if_duplicate)
{
  return add_word(word, get_size(), fail_if_duplicate);
}

bool yann::Dictionary::has_word(const MatrixSize & num) const
{
  auto it = _n2w.find(num);
  return it != _n2w.end();
}

boost::optional<MatrixSize> yann::Dictionary::find_word(const std::string & word) const
{
  auto it = _w2n.find(word);
  if(it == _w2n.end()) {
    return boost::none;
  }
  return it->second;
}

boost::optional<const std::string &> yann::Dictionary::find_word(const MatrixSize & num) const
{
  auto it = _n2w.find(num);
  if(it == _n2w.end()) {
    return boost::none;
  }
  return it->second;
}

MatrixSize yann::Dictionary::get_word(const std::string & word) const
{
  auto res = find_word(word);
  if(!res) {
    throw runtime_error("word '" + word + "' is not found in the dictionary");
  }
  return *res;
}

const std::string & yann::Dictionary::get_word(const MatrixSize & num) const
{
  auto res = find_word(num);
  if(!res) {
    throw runtime_error("word num is not found in the dictionary");
  }
  return *res;
}

void yann::Dictionary::erase(const MatrixSize & deleted)
{
  auto it = _n2w.find(deleted);
  YANN_CHECK_NE(it, _n2w.end());
  _w2n.erase(it->second); // should be first
  _n2w.erase(it);
}

void yann::Dictionary::erase(const unordered_set<MatrixSize> & deleted)
{
  for(const auto & word_num: deleted) {
    erase(word_num);
  }
}

// re-maps word<->num and returns the new mapping old->new
std::unordered_map<MatrixSize, MatrixSize> yann::Dictionary::compact()
{
  // swap both maps into temps
  Num2Word n2w;
  Word2Num w2n;
  swap(_n2w, n2w);
  swap(_w2n, w2n);

  // re-insert everything back
  std::unordered_map<MatrixSize, MatrixSize> res;
  for(const auto & wn: w2n) {
    res[wn.second] = add_word(wn.first, true);
  }
  return res;
}
