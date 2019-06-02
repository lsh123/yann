/*
 * text_parser.h
 *
 */

#ifndef TEXT_PARSER_H
#define TEXT_PARSER_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/optional.hpp>

#include "core/nn.h"

// Forward declarations
namespace yann::word2vec {
class Dictionary;
class Text;
}; // namespace yann::word2vec


// Overwrites from std:: namespace
namespace std {
ostream& operator<<(ostream & os, const yann::word2vec::Dictionary & dictionary);
istream& operator>>(istream & is, yann::word2vec::Dictionary & dictionary);

ostream& operator<<(ostream & os, const yann::word2vec::Text & text);
istream& operator>>(istream & is, yann::word2vec::Text & text);
}; // namespace std


namespace yann::word2vec {

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

  size_t get_size() const;
  MatrixSize add_word(const std::string & word, bool fail_if_duplicate = false);

  boost::optional<MatrixSize> find_word(const std::string & word) const;
  boost::optional<const std::string &> find_word(const MatrixSize & num) const;

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

// Class representing a text parser
class TextParser {
public:
  enum State {
    BeforeText = 0,
    InsideSentence,
    EndOfSentence
  };
  typedef std::vector<std::string> Sentence;

public:
  virtual enum State push_char(const char & ch) = 0;
  virtual void push_eof() = 0;
  virtual const Sentence & current_sentence() const = 0;
}; // class TextParser

// Class representing an English parser
class EnglishTextParser : public TextParser {
public:
  EnglishTextParser(const size_t & min_word_size = 3);
  virtual ~EnglishTextParser();

  // TextParser overwrites
  virtual enum State push_char(const char & ch);
  virtual void push_eof();
  virtual const Sentence & current_sentence() const { return _cur_sentence; }

private:
  static bool is_end_of_word_char(const char & ch);
  static bool is_end_of_sentence_char(const char & ch);
  bool normalize_and_filter_word(std::string & word);

private:
  const size_t _min_word_size;
  enum State  _state;
  std::string _cur_word;
  Sentence    _cur_sentence;

  std::unordered_set<std::string> _ignored_words;
}; // class EnglishTextParser

// Class representing a text
class Text {
  friend std::ostream& std::operator<<(std::ostream & os, const yann::word2vec::Text & text);
  friend std::istream& std::operator>>(std::istream & is, yann::word2vec::Text & text);

public:
  // A single sentence represented by word's numbers.
  typedef std::vector<MatrixSize> Sentence;

public:
  Text();
  virtual ~Text();

  std::string get_info() const;
  bool operator==(const Text & other) const;

  inline size_t get_dictionary_size() const { return _dictionary.get_size(); }
  inline size_t get_sentences_size() const { return _sentences.size(); }

  boost::optional<MatrixSize> find_word(const std::string & word) const;
  boost::optional<const std::string &> find_word(const MatrixSize & num) const;

  // throws is word is not found
  MatrixSize get_word(const std::string & word) const;
  const std::string & get_word(const MatrixSize & num) const;

  void add_sentence(const std::vector<std::string> & sentence);
  void add_text(std::istream & is, TextParser & parser);

  std::unordered_map<MatrixSize, size_t> get_word_frequencies() const;
  void subsample(const double & sample = 0.0001);

  const std::vector<Sentence> & sentences() const { return _sentences; }
  const Dictionary & dictionary() const { return _dictionary; }

  // just helpers to "recreate" text back for debugging
  std::string get_text() const;
  std::string get_sentence(const Sentence & sentence) const;

private:
  Dictionary              _dictionary;
  std::vector<Sentence>   _sentences;
}; // class Text


}; // namespace yann::word2vec

#endif /* TEXT_PARSER_H */
