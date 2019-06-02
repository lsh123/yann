/*
 * text_parser.cpp
 *
 */
#include <numeric>
#include <random>
#include <string>
#include <utility>

#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/functions.h"
#include "layers/contlayer.h"
#include "layers/fclayer.h"
#include "layers/smaxlayer.h"

#include "text_parser.h"

using namespace std;
using namespace boost;
using namespace yann;
using namespace yann::word2vec;


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

yann::word2vec::Dictionary::Dictionary()
{
}

yann::word2vec::Dictionary::~Dictionary()
{
}


bool yann::word2vec::Dictionary::operator==(const Dictionary & other) const
{
  return _n2w == other._n2w && _w2n == other._w2n;
}

size_t yann::word2vec::Dictionary::get_size() const
{
  YANN_CHECK_EQ(_n2w.size(), _w2n.size());
  return _n2w.size();
}

MatrixSize yann::word2vec::Dictionary::add_word(const std::string & word, const MatrixSize & num, bool fail_if_duplicate)
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

MatrixSize yann::word2vec::Dictionary::add_word(const std::string & word, bool fail_if_duplicate)
{
  return add_word(word, get_size(), fail_if_duplicate);
}

boost::optional<MatrixSize> yann::word2vec::Dictionary::find_word(const std::string & word) const
{
  auto it = _w2n.find(word);
  if(it == _w2n.end()) {
    return boost::none;
  }
  return it->second;
}

boost::optional<const std::string &> yann::word2vec::Dictionary::find_word(const MatrixSize & num) const
{
  auto it = _n2w.find(num);
  if(it == _n2w.end()) {
    return boost::none;
  }
  return it->second;
}

MatrixSize yann::word2vec::Dictionary::get_word(const std::string & word) const
{
  auto res = find_word(word);
  if(!res) {
    throw runtime_error("word '" + word + "' is not found in the dictionary");
  }
  return *res;
}

const std::string & yann::word2vec::Dictionary::get_word(const MatrixSize & num) const
{
  auto res = find_word(num);
  if(!res) {
    throw runtime_error("word num is not found in the dictionary");
  }
  return *res;
}

void yann::word2vec::Dictionary::erase(const unordered_set<MatrixSize> & deleted)
{
  for(const auto & word_num: deleted) {
    auto it = _n2w.find(word_num);
    YANN_CHECK_NE(it, _n2w.end());
    _w2n.erase(it->second); // should be first
    _n2w.erase(it);
  }
}

// re-maps word<->num and returns the new mapping old->new
std::unordered_map<MatrixSize, MatrixSize> yann::word2vec::Dictionary::compact()
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

////////////////////////////////////////////////////////////////////////////////////////////////
//
// EnglishTextParser implementation
//
bool yann::word2vec::EnglishTextParser::is_end_of_word_char(const char & ch)
{
  return !isalnum(ch) && ch != '-' && ch !='&';
}

bool yann::word2vec::EnglishTextParser::is_end_of_sentence_char(const char & ch)
{
  return ch == '.' || ch == '?' || ch == '!';
}
yann::word2vec::EnglishTextParser::EnglishTextParser(
    const size_t & min_word_size) :
  _min_word_size(min_word_size),
  _state(BeforeText)
{
  const static string ignored_words[] = {
      "a", "the",
  };
  for(const auto & word: ignored_words) {
    _ignored_words.insert(word);
  }
}

yann::word2vec::EnglishTextParser::~EnglishTextParser()
{

}

bool yann::word2vec::EnglishTextParser::normalize_and_filter_word(std::string & word)
{
  if(word.length() <= _min_word_size) {
    return false;
  }
  if(!isalpha(word[0])) {
    return false;
  }
  if(_ignored_words.find(word) != _ignored_words.end()) {
    return false;
  }
  // TODO: normalize the word (e.g. remove 's' at the end)
  return true;
}

// TextParser overwrites
enum yann::word2vec::TextParser::State yann::word2vec::EnglishTextParser::push_char(const char & ch)
{
  switch(_state) {
  case BeforeText:
    // just in case
    _cur_word.clear();
    _cur_sentence.clear();
    _state = InsideSentence;
    break;
  case InsideSentence:
    // do nothing
    break;
  case EndOfSentence:
    _cur_word.clear();
    _cur_sentence.clear();
    _state = InsideSentence;
    break;
  }

  if(is_end_of_sentence_char(ch)) {
    if(!_cur_word.empty()) {
      if(normalize_and_filter_word(_cur_word)) {
        _cur_sentence.push_back(_cur_word);
      }
      _cur_word.clear();
    }
    _state = EndOfSentence;
  } else if(is_end_of_word_char(ch)) {
    if(!_cur_word.empty()) {
      if(normalize_and_filter_word(_cur_word)) {
        _cur_sentence.push_back(_cur_word);
      }
      _cur_word.clear();
    }
  } else {
    // TODO: we shouldn't always lower case words
    _cur_word += tolower(ch);
  }

  // done
  return _state;
}

void yann::word2vec::EnglishTextParser::push_eof()
{
  switch(_state) {
  case BeforeText:
    // just in case
    _cur_word.clear();
    _cur_sentence.clear();
    break;
  case InsideSentence:
    // do nothing
    break;
  case EndOfSentence:
    // clear the current sentence, otherwise we double the last sentence in file
    _cur_word.clear();
    _cur_sentence.clear();
    break;
  }

  _state = BeforeText;
}


////////////////////////////////////////////////////////////////////////////////////////////////
//
// Text implementation
//
namespace std {

// format: (d:<dictionary>,s:<sentences>)
ostream& operator<<(ostream & os, const Text & text)
{
  os << "(d:" << text._dictionary << ",s:" << text._sentences << ")";
  return os;
}

// format: (d:<dictionary>,s:<sentences>)
istream& operator>>(istream & is, Text & text)
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
  is >> text._dictionary;
  if(is.fail()) {
    return is;
  }
  if(is >> ch && ch != ',') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> ch && ch != 's') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> ch && ch != ':') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  is >> text._sentences;
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
}; // namespace std

yann::word2vec::Text::Text()
{
}

yann::word2vec::Text::~Text()
{
}


std::string yann::word2vec::Text::get_info() const
{
  ostringstream oss;
  oss << "yann::word2vec::Text"
      << " dictionay size: " << get_dictionary_size()
      << ", sentences size: " << get_sentences_size()
  ;
  return oss.str();
}

bool yann::word2vec::Text::operator==(const Text & other) const
{
  return _dictionary == other._dictionary && _sentences == other._sentences;
}

optional<MatrixSize> yann::word2vec::Text::find_word(const string & word) const
{
  return _dictionary.find_word(word);
}

boost::optional<const string &> yann::word2vec::Text::find_word(const MatrixSize & num) const
{
  return _dictionary.find_word(num);
}

MatrixSize yann::word2vec::Text::get_word(const string & word) const
{
  return _dictionary.get_word(word);
}

const string & yann::word2vec::Text::get_word(const MatrixSize & num) const
{
  return _dictionary.get_word(num);
}

void yann::word2vec::Text::add_sentence(const std::vector<std::string> & sentence)
{
  Sentence ss;
  ss.reserve(sentence.size());
  for(auto it = sentence.begin(); it != sentence.end(); ++it) {
    size_t word = _dictionary.add_word(*it);
    ss.push_back(word);
  }
  _sentences.push_back(ss);
}

void yann::word2vec::Text::add_text(istream & is, TextParser & parser)
{
  // just in case...
  is >> std::noskipws;

  // read sentence
  char ch;
  while(!is.eof() && !is.fail() && is >> ch) {
    auto state = parser.push_char(ch);
    switch(state) {
    case TextParser::BeforeText:
    case TextParser::InsideSentence:
      // do nothing
      break;
    case TextParser::EndOfSentence:
      if(!parser.current_sentence().empty()) {
        add_sentence(parser.current_sentence());
      }
      break;
    }
  }

  // just in case
  parser.push_eof();
  if(!parser.current_sentence().empty()) {
    add_sentence(parser.current_sentence());
  }
}

// just helpers to "recreate" text back for debugging
std::string yann::word2vec::Text::get_text() const
{
  ostringstream oss;

  bool write_end_of_sentence_space = false;
  for(const auto & sentence: _sentences) {
    if(write_end_of_sentence_space) {
      oss << " ";
    }
    oss << get_sentence(sentence);

    write_end_of_sentence_space = true;
  }

  return oss.str();
}

std::string yann::word2vec::Text::get_sentence(const Sentence & sentence) const
{
  ostringstream oss;
  bool write_end_of_word_space = false;
  for(const auto & word_num : sentence) {
    if(write_end_of_word_space) {
      oss << " ";
    }
    oss << get_word(word_num);
    write_end_of_word_space = true;
  }
  oss << ".";
  return oss.str();
}

unordered_map<MatrixSize, size_t> yann::word2vec::Text::get_word_frequencies() const
{
  std::unordered_map<MatrixSize, size_t> res;
  for(const auto & sentence: _sentences) {
    for(const auto & word_num: sentence) {
      ++res[word_num];
    }
  }
  return res;
}

//
void yann::word2vec::Text::subsample(const double & sample)
{
  YANN_CHECK_GT(sample, 0);

  // get frequencies
  auto frequencies = get_word_frequencies();
  size_t total = std::accumulate(
      frequencies.begin(), frequencies.end(), 0,
      [](const size_t & a, const auto & b) {
        return a + b.second;
      }
  );
  if(total <= 0) {
    return;
  }

  // get list of words we want to delete
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  unordered_set<MatrixSize> deleted;
  for(const auto & word_freq : frequencies) {
    auto freq_by_sample = word_freq.second / ((double) total * sample);
    auto val = distribution(generator);

    if(val >= ((sqrt(freq_by_sample) + 1) / freq_by_sample)) {
      deleted.insert(word_freq.first);
    }
  }

  // now delete it!
  _dictionary.erase(deleted);
  for(auto & sentence: _sentences) {
    auto remove_it = std::remove_if(
        sentence.begin(), sentence.end(),
        [deleted](const auto & word_num) {
          return deleted.find(word_num) != deleted.end();
        }
    );
    sentence.erase(remove_it, sentence.end());
  }
  // remove any empty sentences
  auto remove_it = std::remove_if(
      _sentences.begin(), _sentences.end(),
     [](const auto & sentence) {
       return sentence.empty();
     }
 );
  _sentences.erase(remove_it, _sentences.end());

  // compact the dictionary and then re-map words in the sentences
  auto num_mapping = _dictionary.compact();
  for(auto & sentence: _sentences) {
    for(auto it = sentence.begin(); it != sentence.end(); ++it) {
      YANN_SLOW_CHECK_NE(num_mapping.find(*it), num_mapping.end());
      *it = num_mapping[*it];
    }
  }
}
