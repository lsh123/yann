/*
 * text_parser.cpp
 *
 */
#include <numeric>
#include <string>
#include <utility>

#include <boost/assert.hpp>

#include "core/random.h"
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
      << ", words size: " << get_words_size()
  ;
  return oss.str();
}

size_t yann::word2vec::Text::get_words_size() const
{
  return std::accumulate(_sentences.begin(), _sentences.end(), 0,
      [](const size_t & a, const auto & b) -> size_t {
        return a + b.size();
      }
  );
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

void yann::word2vec::Text::compact()
{
  // compact the dictionary and then re-map words in the sentences
  auto num_mapping = _dictionary.compact();
  for(auto & sentence: _sentences) {
    for(auto it = sentence.begin(); it != sentence.end(); ++it) {
      YANN_SLOW_CHECK_NE(num_mapping.find(*it), num_mapping.end());
      *it = num_mapping[*it];
    }
  }

  // remove any empty sentences
  auto remove_it = std::remove_if(_sentences.begin(), _sentences.end(),
     [](const auto & sentence) {
       return sentence.empty();
     }
  );
  _sentences.erase(remove_it, _sentences.end());
}

std::pair<size_t, size_t> yann::word2vec::Text::subsample(
    const size_t & min_count,
    const double & sample_rate,
    boost::optional<Value> rand_seed)
{
  YANN_CHECK_GE(min_count, 0);
  YANN_CHECK_GT(sample_rate, 0);

 // get frequencies and remove all rare words
  auto frequencies = get_word_frequencies();
  for(auto it = frequencies.begin(); it != frequencies.end();) {
    if(it->second < min_count) {
      _dictionary.erase(it->first);
      it = frequencies.erase(it);
    } else {
      ++it;
    }
  }

  // calculate total
  auto total = std::accumulate(frequencies.begin(), frequencies.end(), 0,
      [](const size_t & a, const auto & b) -> size_t {
        return a + b.second;
      }
  );
  if(total <= 0) {
    YANN_CHECK(_dictionary.empty());
    size_t total = get_words_size();
    _sentences.clear();
    return make_pair(total, 0);
  }

  // sample the frequent words
  size_t filtered_count = 0;
  size_t sampled_count = 0;
  auto rand = yann::RandomGenerator::uniform_distribution(0.0, 1.0, rand_seed);
  YANN_CHECK(rand);
  for(auto & sentence: _sentences) {
    auto remove_it = std::remove_if(
        sentence.begin(), sentence.end(),
        [&](const auto & word_num) -> bool {
          // first check if we already deleted this word
          if(!_dictionary.has_word(word_num)) {
            ++filtered_count;
            return true;
          }
          YANN_SLOW_CHECK_NE(frequencies.find(word_num), frequencies.end());

          // calculated ranking for this word
          const auto freq_by_sample = frequencies[word_num] / ((double) total * sample_rate);
          const auto ranking = (sqrt(freq_by_sample) + 1) / freq_by_sample;

          // and remove it
          if(ranking < rand->next()) {
            ++sampled_count;
            return true;
          }

          return false;
        }
    );
    sentence.erase(remove_it, sentence.end());
  }

  // finally cleanup and compact
  compact();

  // done
  return make_pair(filtered_count, sampled_count);
}
