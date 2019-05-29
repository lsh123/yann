/*
 * MnisTest.cpp
 *
 */
#include <algorithm>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <boost/assert.hpp>
#include <boost/test/unit_test.hpp>

#include "nn.h"
#include "nntraining.h"
#include "utils.h"

#include "timer.h"
#include "test_utils.h"
#include "mnist-test.h"

using namespace std;
using namespace yann;
using namespace yann::test;

#define LABELS_COUNT      10    // digits 0...9
#define IMAGES_MAX_VALUE  256   // colors 0...256

#define LABELS_MAGIC_NUMBER 2049
#define IMAGES_MAGIC_NUMBER 2051

#define TRAINING_IMAGES_FILENAME "train-images-idx3-ubyte"
#define TRAINING_LABELS_FILENAME "train-labels-idx1-ubyte"
#define TESTING_IMAGES_FILENAME  "t10k-images-idx3-ubyte"
#define TESTING_LABELS_FILENAME  "t10k-labels-idx1-ubyte"

namespace yann::test {

  inline static unsigned int make_uint(Byte buf[4]) {
    return (((unsigned int) buf[0]) << 24 | ((unsigned int) buf[1]) << 16
        | ((unsigned int) buf[2]) << 8 | ((unsigned int) buf[3]) << 0);
  }

  template<typename T>
  inline static MatrixSize get_label(const T & vv) {
    MatrixSize pos = 0;
    vv.maxCoeff(&pos);
    BOOST_VERIFY(pos <= LABELS_COUNT);
    return pos;
  }
};// namespace yann::test

////////////////////////////////////////////////////////////////////////////////////////////////
//
// MnistDataset implementation
//
yann::test::MnistDataset::MnistDataset() :
    _image_rows(0),
    _image_cols(0)
{
}

yann::test::MnistDataset::~MnistDataset()
{
}

MatrixSize  yann::test::MnistDataset::get_size() const
{
  BOOST_VERIFY(get_batch_size(_images) == get_batch_size(_labels));
    return get_batch_size(_images);
}

MatrixSize yann::test::MnistDataset::get_image_size() const
{
  BOOST_VERIFY(get_batch_item_size(_images) == _image_rows * _image_cols);
  return get_batch_item_size(_images);
}

MatrixSize yann::test::MnistDataset::get_label_size() const
{
  return get_batch_item_size(_labels);
}


/*********************************************************
 *
 * TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000803(2051) magic number
 * 0004     32 bit integer  60000            number of images
 * 0008     32 bit integer  28               number of rows
 * 0012     32 bit integer  28               number of columns
 * 0016     unsigned byte   ??               pixel
 * 0017     unsigned byte   ??               pixel
 * ........
 * xxxx     unsigned byte   ??               pixel
 *
 * Pixels are organized row-wise. Pixel values are 0 to 255.
 * 0 means background (white), 255 means foreground (black).
 *
 *********************************************************/
void yann::test::MnistDataset::read_images(istream &in)
{
  unsigned int magic_number, images_count;
  Byte tmp4[4];
  vector<Byte> image;

  // magic number
  in.read((char*) tmp4, sizeof(tmp4));
  if (!in || in.eof()) {
    throw invalid_argument(
        "invalid file format: can't read images magic number");
  }
  magic_number = make_uint(tmp4);
  if (magic_number != IMAGES_MAGIC_NUMBER) {
    throw invalid_argument("invalid file format: invalid images magic number");
  }

  // images count
  in.read((char*) tmp4, sizeof(tmp4));
  if (!in || in.eof()) {
    throw invalid_argument("invalid file format: can't read images count");
  }
  images_count = make_uint(tmp4);

  // rows count
  in.read((char*) tmp4, sizeof(tmp4));
  if (!in || in.eof()) {
    throw invalid_argument("invalid file format: can't read rows count");
  }
  _image_rows = make_uint(tmp4);

  // columns count
  in.read((char*) tmp4, sizeof(tmp4));
  if (!in || in.eof()) {
    throw invalid_argument("invalid file format: can't read columns count");
  }
  _image_cols = make_uint(tmp4);

  // images
  image.resize(_image_rows * _image_cols);
  resize_batch(_images, images_count, image.size());
  _images.setZero();
  for (size_t ii = 0; ii < images_count; ++ii) {
    in.read((char*) &image.front(), image.size());
    if (!in || in.eof()) {
      throw invalid_argument("invalid file format: can't read image");
    }

    // transform colors 0...256 into -1.0 ... 1.0 signals
    for(size_t jj = 0; jj < image.size(); ++jj) {
      get_batch(_images, ii)(jj) = ((double)image[jj]) / IMAGES_MAX_VALUE;
    }
  }
}

/*********************************************************
 *
 * TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 * 0004     32 bit integer  60000            number of items
 * 0008     unsigned byte   ??               label
 * 0009     unsigned byte   ??               label
 * ........
 * xxxx     unsigned byte   ??               label
 *
 * The labels values are 0 to 9.
 *
 *********************************************************/
void yann::test::MnistDataset::read_labels(istream &in)
{
  unsigned int magic_number, labels_count;
  Byte ll, tmp4[4];

  // magic number
  in.read((char*) tmp4, sizeof(tmp4));
  if (!in || in.eof()) {
    throw invalid_argument(
        "invalid file format: can't read labels magic number");
  }
  magic_number = make_uint(tmp4);
  if (magic_number != LABELS_MAGIC_NUMBER) {
    throw invalid_argument("invalid file format: invalid labels magic number");
  }

  // labels count
  in.read((char*) tmp4, sizeof(tmp4));
  if (!in || in.eof()) {
    throw invalid_argument("invalid file format: can't read labels count");
  }
  labels_count = make_uint(tmp4);

  // labels
  resize_batch(_labels, labels_count, LABELS_COUNT);
  _labels.setZero();
  for (size_t ii = 0; ii < labels_count; ++ii) {
    in.read((char*) &ll, sizeof(ll));
    if (!in || in.eof()) {
      throw invalid_argument("invalid file format: can't read label");
    }
    if (ll >= LABELS_COUNT) {
      throw invalid_argument("invalid file format: label should be 0...9");
    }

    // transform label value into the vector
    get_batch(_labels, ii)(ll) = 1.0;
  }
}

void yann::test::MnistDataset::filter(const MatrixSize & max_label, const MatrixSize & max_num)
{
  BOOST_VERIFY(max_label <= LABELS_COUNT);

  // find how many labels <= max_label are there
  MatrixSize count = 0;
  for(MatrixSize ii = 0; ii < get_batch_size(_labels); ++ii) {
    if(get_label(get_batch(_labels, ii)) <= max_label) {
      ++count;
    }
  }
  if(max_num > 0 && count > max_num) {
    count = max_num;
  }

  // prep storage and copy
  VectorBatch images, labels;
  resize_batch(images, count, get_image_size());
  resize_batch(labels, count, max_label + 1);
  images.setZero();
  labels.setZero();
  for(MatrixSize ii = 0, jj = 0; ii < get_batch_size(_labels) && jj < count; ++ii) {
    MatrixSize label = get_label(get_batch(_labels, ii));
    if(label <= max_label) {
      get_batch(images, jj) = get_batch(_images, ii);
      get_batch(labels, jj)(label) = 1.0;
      ++jj;
    }
  }

  // ready to go
  swap(images, _images);
  swap(labels, _labels);
}

void yann::test::MnistDataset::shift_values(const Value & min_val, const Value & max_val)
{
  BOOST_VERIFY(min_val < max_val);
  _images.array() = _images.array() * (max_val - min_val) + min_val;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// MnistTest implementation
//

// Overwrites from std:: namespace
ostream& std::operator<<(ostream & os, const MnistTest & test)
{
  os << test.training().get_size() << " training labels/images" << endl;
  os << test.testing().get_size() << " test labels/images" << endl;
  return os;
}

yann::test::MnistTest::MnistTest()
{
}

yann::test::MnistTest::~MnistTest()
{
}

void yann::test::MnistTest::read(const string & base_folder)
{
  ifstream ifs;

  // training
  ifs.open(base_folder + "/" + TRAINING_LABELS_FILENAME, ios::in | ios::binary);
  _training.read_labels(ifs);
  ifs.close();

  ifs.open(base_folder + "/" + TRAINING_IMAGES_FILENAME, ios::in | ios::binary);
  _training.read_images(ifs);
  ifs.close();

  // testing
  ifs.open(base_folder + "/" + TESTING_LABELS_FILENAME, ios::in | ios::binary);
  _testing.read_labels(ifs);
  ifs.close();

  ifs.open(base_folder + "/" + TESTING_IMAGES_FILENAME, ios::in | ios::binary);
  _testing.read_images(ifs);
  ifs.close();

  // validation
  BOOST_VERIFY(get_batch_size(_training.labels()) == get_batch_size(_training.images()));
  BOOST_VERIFY(get_batch_size(_testing.labels()) == get_batch_size(_testing.images()));
  BOOST_VERIFY(_training.get_image_size() == _testing.get_image_size());
  BOOST_VERIFY(_training.get_label_size() == _testing.get_label_size());
}

pair<size_t, Value> yann::test::MnistTest::test(const Network & nn,
                                                const RefConstVectorBatch & images,
                                                const RefConstVectorBatch & labels,
                                                Context * ctx)
{
  BOOST_VERIFY(ctx);
  BOOST_VERIFY(get_batch_size(images) > 0);
  BOOST_VERIFY(get_batch_size(images) == get_batch_size(labels));

  nn.calculate(images, ctx);
  RefConstVectorBatch output = ctx->get_output();

  BOOST_VERIFY(is_same_size(output, labels));
  Value cost = nn.cost(output, labels);

  size_t success_count = 0;
  const MatrixSize batch_size = get_batch_size(images);
  for(MatrixSize ii = 0; ii < batch_size; ++ii) {
    MatrixSize actual = get_label(get_batch(output, ii));
    MatrixSize expected = get_label(get_batch(labels, ii));
    if (expected == actual) {
      ++success_count;
    }
  }

  // done
  return make_pair(success_count, cost);
}

pair<double, Value> yann::test::MnistTest::test(
    const Network & nn,
    const MnistDataset & dataset,
    const MatrixSize & test_batch_size)
{
  BOOST_VERIFY(test_batch_size > 0);
  BOOST_VERIFY(get_batch_size(dataset.images()) >= test_batch_size);
  BOOST_VERIFY(get_batch_size(dataset.images()) == get_batch_size(dataset.labels()));

  const MatrixSize size = get_batch_size(dataset.images());
  const MatrixSize image_size = get_batch_item_size(dataset.images());
  const MatrixSize label_size = get_batch_item_size(dataset.labels());

  unique_ptr<Context> ctx = nn.create_context(test_batch_size);
  BOOST_VERIFY(ctx);

  pair<double, Value> result = { 0, 0 };

  // ensure we don't do allocations in eigen
  {
    BlockAllocations block;

    MatrixSize pos = 0;
    for(; pos <= size - test_batch_size; pos += test_batch_size) {
      pair<size_t, Value> res = test(
          nn,
          dataset.images().block(pos, 0, test_batch_size, image_size), // assumes RowMajor layout
          dataset.labels().block(pos, 0, test_batch_size, label_size), // assumes RowMajor layout
          ctx.get()
      );
      result.first += res.first;
      result.second += res.second;
    }

    // normalize results
    result.first /= pos;
    result.second /= pos;
  }

  // done
  return result;
}

pair<double, Value> yann::test::MnistTest::train_and_test(
    Network & nn,
    const Trainer & trainer,
    size_t epochs,
    const MatrixSize & test_batch_size)
{
  BOOST_VERIFY(epochs > 0);
  BOOST_VERIFY(get_batch_size(_training.labels()) == get_batch_size(_training.images()));
  BOOST_VERIFY(get_batch_size(_testing.labels()) == get_batch_size(_testing.images()));

  // training
  double test_success_rate = 0, training_success_rate = 0;
  Value test_cost = 0, training_cost = 0;
  {
    Timer timer("Total training/testing");
    for (size_t epoch = 0; epoch <= epochs; ++epoch) {
      // epoch 0 is to run tests before training
      if(epoch > 0) {
        BOOST_TEST_MESSAGE("Training epoch " << epoch << " out of " << epochs);
        {
          Timer timer("Training");
          trainer.train(nn, _training.images(), _training.labels());
          BOOST_TEST_MESSAGE(timer);
        }
      }

      {
        Timer timer("Testing against training dataset");
        pair<double, Value> res = test(nn, _training, test_batch_size);
        training_success_rate = res.first;
        training_cost = res.second;
        BOOST_TEST_MESSAGE(timer);
      }
      {
        Timer timer("Testing against test dataset");
        pair<double, Value> res = test(nn, _testing, test_batch_size);
        test_success_rate = res.first;
        test_cost = res.second;
        BOOST_TEST_MESSAGE(timer);
      }

      string extra = (epoch == 0) ? " (before training)" : "";
      BOOST_TEST_MESSAGE(
          "Success rate for epoch " << epoch << extra << ":"
              << " against training dataset: " << training_success_rate * 100 << "%"
              << " against test dataset: " << test_success_rate* 100 << "%"
      );
      BOOST_TEST_MESSAGE(
          "Cost/loss per test for epoch " << epoch << extra << ":"
              << " against training dataset: " << training_cost
              << " against test dataset: " << test_cost
      );
    }
    BOOST_TEST_MESSAGE(timer << "\n");
  }

  // done
  return make_pair(test_success_rate, test_cost);
}


MatrixSize yann::test::MnistTest::get_training_size() const
{
  return _training.get_size();
}

MatrixSize yann::test::MnistTest::get_testing_size() const
{
  return _testing.get_size();
}

MatrixSize yann::test::MnistTest::get_image_rows() const
{
  BOOST_VERIFY(_training.get_image_rows() == _testing.get_image_rows());
  return _training.get_image_rows();
}
MatrixSize yann::test::MnistTest::get_image_cols() const
{
  BOOST_VERIFY(_training.get_image_cols() == _testing.get_image_cols());
  return _training.get_image_cols();
}
MatrixSize yann::test::MnistTest::get_image_size() const
{
  BOOST_VERIFY(_training.get_image_size() == _testing.get_image_size());
  return _training.get_image_size();
}

MatrixSize yann::test::MnistTest::get_label_size() const
{
  BOOST_VERIFY(_training.get_label_size() == _testing.get_label_size());
  return _training.get_label_size();
}

vector<MatrixSize> yann::test::MnistTest::get_layer_sizes(
    const vector<MatrixSize> & hidden_layer_sizes) const
{
  // add input / output layers based on mnist data
  vector<MatrixSize> layer_sizes;
  layer_sizes.reserve(hidden_layer_sizes.size() + 2);

  layer_sizes.push_back(get_image_size());
  layer_sizes.insert(layer_sizes.end(), hidden_layer_sizes.begin(),
                     hidden_layer_sizes.end());
  layer_sizes.push_back(get_label_size());
  return layer_sizes;
}

void yann::test::MnistTest::filter(
    const MatrixSize & max_label,
    const MatrixSize & training_max_num,
    const MatrixSize & testing_max_num)
{
  _training.filter(max_label, training_max_num);
  _testing.filter(max_label, testing_max_num);
}

void yann::test::MnistTest::shift_values(
    const Value & min_val,
    const Value & max_val)
{
  _training.shift_values(min_val, max_val);
  _testing.shift_values(min_val, max_val);
}
