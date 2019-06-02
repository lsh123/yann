//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "Word2Vec Tests"
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <sstream>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "core/utils.h"
#include "word2vec/text_parser.h"
#include "word2vec/word2vec.h"

#include "test_utils.h"
#include "timer.h"

using namespace std;
using namespace boost;
using namespace boost::iostreams;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::word2vec;
using namespace yann::test;

#define HOTELS_100_TEST_PATH    "../data/hotels-100.txt.gz"
#define HOTELS_1000_TEST_PATH   "../data/hotels-1000.txt.gz"
#define HOTELS_10000_TEST_PATH  "../data/hotels-10000.txt.gz"
#define HOTELS_FULL_TEST_PATH   "../data/hotels-full.txt.gz"
#define TMP_FOLDER  "/tmp/"

struct Word2VecTestFixture
{

  Word2VecTestFixture()
  {
  }
  ~Word2VecTestFixture()
  {
  }

  const string some_text1 = "We shall defend our Island, whatever the cost may be."
                     "We shall fight  on the  beaches. We shall fight on the "
                     "landing grounds. We shall fight in the fields and in "
                     "the streets. We shall fight in the hills. We shall "
                     "never surrender.";
}; // struct Word2VecTestFixture

BOOST_FIXTURE_TEST_SUITE(Word2VecTest, Word2VecTestFixture);

BOOST_AUTO_TEST_CASE(Text_IO_Test)
{
  auto one = make_unique<Text>();
  BOOST_CHECK(one);
  one->add_sentence({"like", "a", "cat", "on", "a", "hot", "coals"});
  one->add_sentence({"like", "a", "cat", "on", "a", "hot", "tin", "roof"});

  BOOST_TEST_MESSAGE("Text before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << *one;
  BOOST_CHECK(!oss.fail());

  auto two = make_unique<Text>();
  BOOST_CHECK(two);
  std::istringstream iss(oss.str());
  iss >> *two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("Text after loading from file: " << "\n" << *two);

  BOOST_CHECK_EQUAL(*one, *two);
}

BOOST_AUTO_TEST_CASE(Text_Read_Test)
{
  auto text = make_unique<Text>();
  BOOST_CHECK(text);

  EnglishTextParser parser(2);
  std::istringstream iss(some_text1);
  text->add_text(iss, parser);

  BOOST_TEST_MESSAGE("Text: " << "\n" << *text);
  BOOST_TEST_MESSAGE("Text: " << "\n" << text->get_text());

  BOOST_CHECK_EQUAL(text->get_sentences_size(), 6);
  BOOST_CHECK_EQUAL(text->get_dictionary_size(), 17);
}

BOOST_AUTO_TEST_CASE(Text_Subsampling_Test)
{
  auto text = make_unique<Text>();
  BOOST_CHECK(text);

  EnglishTextParser parser(2);
  std::istringstream iss(some_text1);
  text->add_text(iss, parser);

  BOOST_TEST_MESSAGE("Text: " << "\n" << *text);
  BOOST_TEST_MESSAGE("Text: " << "\n" << text->get_text());

  auto sentences_size = text->get_sentences_size();
  auto dictionary_size = text->get_dictionary_size();

  text->subsample(0.0001);

  BOOST_TEST_MESSAGE("Text after subsampling: " << "\n" << *text);
  BOOST_TEST_MESSAGE("Text after subsampling: " << "\n" << text->get_text());

  BOOST_CHECK(text->get_sentences_size() <= sentences_size);
  BOOST_CHECK(text->get_dictionary_size() < dictionary_size);
}


BOOST_AUTO_TEST_CASE(Word2Vec_IO_Test)
{
  auto text = make_unique<Text>();
  BOOST_CHECK(text);

  EnglishTextParser parser;
  std::istringstream iss_text(some_text1);
  text->add_text(iss_text, parser);

  Word2Vec::TrainingParams params;
  params._dimensions = 3;
  params._epochs = 10;
  auto one = Word2Vec::train_skip_gram(*text, params);
  BOOST_CHECK(one);

  BOOST_TEST_MESSAGE("Word2Vec before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << *one;
  BOOST_CHECK(!oss.fail());

  auto two = make_unique<Word2Vec>();
  BOOST_CHECK(two);
  std::istringstream iss(oss.str());
  iss >> *two;
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("Word2Vec after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Word2Vec_Closest_Test)
{
  auto text = make_unique<Text>();
  BOOST_CHECK(text);

  EnglishTextParser parser;
  std::istringstream iss_text(some_text1);
  text->add_text(iss_text, parser);

  Word2Vec::TrainingParams params;
  params._dimensions = 10;
  params._epochs = 50;
  auto w2v = Word2Vec::train_skip_gram(*text, params);
  BOOST_CHECK(w2v);

  auto closest = w2v->find_closest("fight", 3);
  DBG(closest);
}

BOOST_AUTO_TEST_CASE(Word2Vec_Hotels_Test, * disabled())
{
  // setup file
  ifstream file(HOTELS_100_TEST_PATH, ios_base::in | ios_base::binary);
  filtering_streambuf<input> inbuf;
  inbuf.push(gzip_decompressor());
  inbuf.push(file);
  istream in(&inbuf);

  // parse text
  auto text = make_unique<Text>();
  BOOST_CHECK(text);
  {
    Timer timer("Parsing text");
    EnglishTextParser parser;
    text->add_text(in, parser);
  }
  BOOST_TEST_MESSAGE("Parsed text: " << text->get_info());
  {
    Timer timer("Subsampling text");
    text->subsample();
  }
  BOOST_TEST_MESSAGE("Subsampled text: " << text->get_info());

  unique_ptr<Word2Vec> w2v;
  {
    Timer timer("Word2Vec::train_skip_gram");
    Word2Vec::TrainingParams params;
    params._window_size = 3;
    params._dimensions = 100;
    params._learning_rate = 1.0;
    params._regularization = 0.00001;
    params._training_batch_size = 10;
    params._epochs = 10;
    params._epochs_callback = ecpoch_progress_callback;
    params._batch_callback = batch_progress_callback;
    w2v = Word2Vec::train_skip_gram(*text, params);
    BOOST_CHECK(w2v);
  }
  {
    Timer timer("Word2Vec::find_closest");
    auto closest = w2v->find_closest("dirty", 5);
    DBG(closest);
  }
}

BOOST_AUTO_TEST_SUITE_END()

