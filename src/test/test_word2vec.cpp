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
#include "core/updaters.h"
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

  unique_ptr<Text> parse_gzip_file(const string & filename)
  {
    // setup file
    ifstream file(filename, ios_base::in | ios_base::binary);
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

    return text;
  }

  void save_to_file(const unique_ptr<Word2Vec> & w2v)
  {
    // save
    string filename = TMP_FOLDER + Timer::get_time() + ".w2v";
    w2v->save(filename);
    BOOST_TEST_MESSAGE("*** Saved to file: " << filename);
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

  auto res = text->subsample();
  BOOST_TEST_MESSAGE("Filtered count: " << res.first << ", sampled count: " << res.second);

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

BOOST_AUTO_TEST_CASE(Word2Vec_SkipGram_Hotels100_Test)
{
  // parse text
  auto text = parse_gzip_file(HOTELS_100_TEST_PATH);
  BOOST_CHECK(text);

  auto res = text->subsample(5, 0.0001, make_optional<Value>(12345));
  BOOST_TEST_MESSAGE("Filtered count: " << res.first << ", sampled count: " << res.second);
  BOOST_TEST_MESSAGE("Filtered text: " << text->get_info());

  unique_ptr<Word2Vec> w2v;
  {
    Timer timer("Word2Vec::train_skip_gram");
    Word2Vec::TrainingParams params;
    params._window_size = 5;
    params._dimensions = 50;
    params._training_init_context = Layer::InitContext(12345);
    params._updater = make_unique<Updater_GradientDescent>(0.75);
    params._training_sampling_rate = 0.0;
    params._training_batch_size = 10;
    params._epochs = 5;
    params._epochs_callback = epoch_progress_callback;
    params._batch_callback = batch_progress_callback;
    w2v = Word2Vec::train_skip_gram(*text, params);
    BOOST_CHECK(w2v);
  }

  {
    Timer timer("Word2Vec::find_closest/find_farthest");
    DBG(w2v->find_closest("bedroom"));
    DBG(w2v->find_farthest("bedroom"));

    DBG(w2v->find_closest("arrival"));
    DBG(w2v->find_farthest("arrival"));
  }

  BOOST_CHECK_GE(w2v->distance("bedroom", "food"), 0.95);
  BOOST_CHECK_GE(w2v->distance("bedroom", "elevators"), 0.95);
  BOOST_CHECK_LE(w2v->distance("bedroom", "better"), 0.20);
  BOOST_CHECK_GE(w2v->distance("arrival", "corner"), 0.90);
  BOOST_CHECK_GE(w2v->distance("arrival", "room"), 0.80);
  BOOST_CHECK_LE(w2v->distance("arrival", "simply"), 0.20);
}

BOOST_AUTO_TEST_CASE(Word2Vec_CBOW_Hotels100_Test)
{
  // parse text
  auto text = parse_gzip_file(HOTELS_100_TEST_PATH);
  BOOST_CHECK(text);
  auto res = text->subsample(10, 0.0001, make_optional<Value>(12345));
  BOOST_TEST_MESSAGE("Filtered count: " << res.first << ", sampled count: " << res.second);
  BOOST_TEST_MESSAGE("Filtered text: " << text->get_info());

  unique_ptr<Word2Vec> w2v;
  {
    Timer timer("Word2Vec::train_cbow");
    Word2Vec::TrainingParams params;
    params._window_size = 3;
    params._dimensions = 50;
    params._training_init_context = Layer::InitContext(12345);
    params._updater = make_unique<Updater_GradientDescent>(0.75);
    params._training_sampling_rate = 0.0;
    params._training_batch_size = 10;
    params._epochs = 5;
    params._epochs_callback = epoch_progress_callback;
    params._batch_callback = batch_progress_callback;
    w2v = Word2Vec::train_cbow(*text, params);
    BOOST_CHECK(w2v);
  }

  {
    Timer timer("Word2Vec::find_closest/find_farthest");
    DBG(w2v->find_closest("bedroom"));
    DBG(w2v->find_farthest("bedroom"));

    DBG(w2v->find_closest("arrival"));
    DBG(w2v->find_farthest("arrival"));
  }

  BOOST_CHECK_GE(w2v->distance("bedroom", "lovely"), 0.85);
  BOOST_CHECK_GE(w2v->distance("bedroom", "husband"), 0.85);
  BOOST_CHECK_LE(w2v->distance("bedroom", "found"), 0.2);
  BOOST_CHECK_GE(w2v->distance("arrival", "excellent"), 0.75);
  BOOST_CHECK_GE(w2v->distance("arrival", "pool"), 0.90);
  BOOST_CHECK_LE(w2v->distance("arrival", "found"), 0.20);
}

BOOST_AUTO_TEST_CASE(Word2Vec_SkipGram_Hotels10000_Test, * disabled())
{
  // parse text
  auto text = parse_gzip_file(HOTELS_10000_TEST_PATH);
  BOOST_CHECK(text);
  auto res = text->subsample();
  BOOST_TEST_MESSAGE("Filtered count: " << res.first << ", sampled count: " << res.second);
  BOOST_TEST_MESSAGE("Filtered text: " << text->get_info());

  unique_ptr<Word2Vec> w2v;
  {
    Timer timer("Word2Vec::train_skip_gram");
    Word2Vec::TrainingParams params;
    params._window_size = 5;
    params._dimensions = 100;
    params._updater = make_unique<Updater_GradientDescent>(0.75);
    params._training_sampling_rate = 0.0;
    params._training_batch_size = 100;
    params._epochs = 5;
    params._epochs_callback = epoch_progress_callback;
    params._batch_callback = batch_progress_callback;
    w2v = Word2Vec::train_skip_gram(*text, params);
    BOOST_CHECK(w2v);
  }
  save_to_file(w2v);

  {
    Timer timer("Word2Vec::find_closest/find_farthest");
    DBG(w2v->find_closest("bedroom"));
    DBG(w2v->find_farthest("bedroom"));

    DBG(w2v->find_closest("arrival"));
    DBG(w2v->find_farthest("arrival"));

    DBG(w2v->find_closest("hotel"));
    DBG(w2v->find_farthest("hotel"));

    DBG(w2v->find_closest("breakfast"));
    DBG(w2v->find_farthest("breakfast"));

    DBG(w2v->find_closest("comfortable"));
    DBG(w2v->find_farthest("comfortable"));

    DBG(w2v->find_closest("dirty"));
    DBG(w2v->find_farthest("dirty"));
  }

  DBG(w2v->distance("comfortable", "excellent"));
  DBG(w2v->distance("comfortable", "dirty"));
}

BOOST_AUTO_TEST_CASE(Word2Vec_CBOW_Hotels10000_Test, * disabled())
{
  // parse text
  auto text = parse_gzip_file(HOTELS_10000_TEST_PATH);
  BOOST_CHECK(text);
  auto res = text->subsample(5, 0.00001);
  BOOST_TEST_MESSAGE("Filtered count: " << res.first << ", sampled count: " << res.second);
  BOOST_TEST_MESSAGE("Filtered text: " << text->get_info());

  unique_ptr<Word2Vec> w2v;
  {
    Timer timer("Word2Vec::train_cbow");
    Word2Vec::TrainingParams params;
    params._window_size = 8;
    params._dimensions = 100;
    params._updater = make_unique<Updater_GradientDescent>(0.75);
    params._training_sampling_rate = 0.0;
    params._training_batch_size = 10;
    params._epochs = 3;
    params._epochs_callback = epoch_progress_callback;
    params._batch_callback = batch_progress_callback;
    w2v = Word2Vec::train_cbow(*text, params);
    BOOST_CHECK(w2v);
  }
  save_to_file(w2v);

  {
    Timer timer("Word2Vec::find_closest/find_farthest");
    DBG(w2v->find_closest("bedroom"));
    DBG(w2v->find_farthest("bedroom"));

    DBG(w2v->find_closest("arrival"));
    DBG(w2v->find_farthest("arrival"));

    DBG(w2v->find_closest("hotel"));
    DBG(w2v->find_farthest("hotel"));

    DBG(w2v->find_closest("breakfast"));
    DBG(w2v->find_farthest("breakfast"));

    DBG(w2v->find_closest("comfortable"));
    DBG(w2v->find_farthest("comfortable"));

    DBG(w2v->find_closest("dirty"));
    DBG(w2v->find_farthest("dirty"));
  }

  DBG(w2v->distance("comfortable", "excellent"));
  DBG(w2v->distance("comfortable", "dirty"));

}

BOOST_AUTO_TEST_CASE(Word2Vec_SkipGram_HotelsFull_Test, * disabled())
{
  // parse text
  auto text = parse_gzip_file(HOTELS_FULL_TEST_PATH);
  BOOST_CHECK(text);
  auto res = text->subsample();
  BOOST_TEST_MESSAGE("Filtered count: " << res.first << ", sampled count: " << res.second);
  BOOST_TEST_MESSAGE("Filtered text: " << text->get_info());

  unique_ptr<Word2Vec> w2v;
  {
    Timer timer("Word2Vec::train_skip_gram");
    Word2Vec::TrainingParams params;
    params._window_size = 5;
    params._dimensions = 100;
    params._updater = make_unique<Updater_GradientDescent>(0.75);
    params._training_sampling_rate = 0.01;
    params._training_batch_size = 10;
    params._epochs = 5;
    params._epochs_callback = epoch_progress_callback;
    params._batch_callback = batch_progress_callback;
    w2v = Word2Vec::train_skip_gram(*text, params);
    BOOST_CHECK(w2v);
  }
  save_to_file(w2v);

  {
    Timer timer("Word2Vec::find_closest/find_farthest");
    DBG(w2v->find_closest("bedroom"));
    DBG(w2v->find_farthest("bedroom"));

    DBG(w2v->find_closest("arrival"));
    DBG(w2v->find_farthest("arrival"));

    DBG(w2v->find_closest("hotel"));
    DBG(w2v->find_farthest("hotel"));

    DBG(w2v->find_closest("breakfast"));
    DBG(w2v->find_farthest("breakfast"));

    DBG(w2v->find_closest("comfortable"));
    DBG(w2v->find_farthest("comfortable"));

    DBG(w2v->find_closest("dirty"));
    DBG(w2v->find_farthest("dirty"));
  }

  DBG(w2v->distance("comfortable", "excellent"));
  DBG(w2v->distance("comfortable", "dirty"));
}

BOOST_AUTO_TEST_CASE(Word2Vec_CBOW_HotelsFull_Test, * disabled())
{
  // parse text
  auto text = parse_gzip_file(HOTELS_FULL_TEST_PATH);
  BOOST_CHECK(text);
  auto res = text->subsample(5);
  BOOST_TEST_MESSAGE("Filtered count: " << res.first << ", sampled count: " << res.second);
  BOOST_TEST_MESSAGE("Filtered text: " << text->get_info());

  unique_ptr<Word2Vec> w2v;
  {
    Timer timer("Word2Vec::train_cbow");
    Word2Vec::TrainingParams params;
    params._window_size = 3;
    params._dimensions = 100;
    params._updater = make_unique<Updater_GradientDescent>(0.75);
    params._training_sampling_rate = 0.001;
    params._training_batch_size = 10;
    params._epochs = 3;
    params._epochs_callback = epoch_progress_callback;
    params._batch_callback = batch_progress_callback;
    w2v = Word2Vec::train_cbow(*text, params);
    BOOST_CHECK(w2v);
  }
  save_to_file(w2v);

  {
    Timer timer("Word2Vec::find_closest/find_farthest");
    DBG(w2v->find_closest("bedroom"));
    DBG(w2v->find_farthest("bedroom"));

    DBG(w2v->find_closest("arrival"));
    DBG(w2v->find_farthest("arrival"));

    DBG(w2v->find_closest("hotel"));
    DBG(w2v->find_farthest("hotel"));

    DBG(w2v->find_closest("breakfast"));
    DBG(w2v->find_farthest("breakfast"));

    DBG(w2v->find_closest("comfortable"));
    DBG(w2v->find_farthest("comfortable"));

    DBG(w2v->find_closest("dirty"));
    DBG(w2v->find_farthest("dirty"));
  }

  DBG(w2v->distance("comfortable", "excellent"));
  DBG(w2v->distance("comfortable", "dirty"));

}


BOOST_AUTO_TEST_SUITE_END()

