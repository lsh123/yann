/*
 * MnistTest.h
 *
 * See http://yann.lecun.com/exdb/mnist/
 *
 */

#ifndef MNISTTEST_H_
#define MNISTTEST_H_

#include <iostream>
#include <string>
#include <utility>

#include "core/types.h"
#include "core/nn.h"
#include "core/training.h"

namespace yann::test {

class MnistDataset {
public:
  MnistDataset();
  virtual ~MnistDataset();

  void read_images(std::istream &in);
  void read_labels(std::istream &in);

  const VectorBatch & images() const {return _images;}
  const VectorBatch & labels() const {return _labels;}

  MatrixSize get_size() const;
  MatrixSize get_image_rows() const { return _image_rows; }
  MatrixSize get_image_cols() const { return _image_cols; }
  MatrixSize get_image_size() const;
  MatrixSize get_label_size() const;

  void filter(const MatrixSize & max_label, const MatrixSize & max_num = 0);
  void shift_values(const Value & min_val, const Value & max_val);

private:
  VectorBatch _images;
  VectorBatch _labels;
  MatrixSize _image_rows;
  MatrixSize _image_cols;
}; // class MnistDataset

class MnistTest {
public:
  MnistTest();
  virtual ~MnistTest();

  void read(const std::string & base_folder);
  std::pair<double, Value> train_and_test(
      yann::Network & nn,
      const yann::Trainer & trainer,
      DataSource_Stochastic::Mode mode,
      const MatrixSize & training_batch_size,
      const size_t & epochs,
      const MatrixSize & test_batch_size);

  const MnistDataset & training() const {return _training;}
  const MnistDataset & testing() const {return _testing;}

  std::vector<MatrixSize> get_layer_sizes(const std::vector<MatrixSize> & hidden_layer_sizes) const;
  MatrixSize get_training_size() const;
  MatrixSize get_testing_size() const;
  MatrixSize get_image_rows() const;
  MatrixSize get_image_cols() const;
  MatrixSize get_image_size() const;
  MatrixSize get_label_size() const;

  // Reduces the training set to labels {0, ..., max_label} no more
  // than training_max_num / testing_max_num for training / testing
  // respectively
  void filter(const MatrixSize & max_label, const MatrixSize & training_max_num = 0, const MatrixSize & testing_max_num = 0);

  // Shifts the range of image values from (0 ... 1) to (min_val ... max_val)
  void shift_values(const Value & min_val, const Value & max_val);

  static std::pair<double, Value> test(const yann::Network & nn,
                                       const MnistDataset & dataset,
                                       const MatrixSize & test_batch_size);

private:
  static std::pair<size_t, Value> test(const yann::Network & nn,
                                       const RefConstVectorBatch & images,
                                       const RefConstVectorBatch & labels,
                                       Context * ctx);

private:
  MnistDataset _training;
  MnistDataset _testing;
}; // class MnistTest

}; // namespace yann::test

// Overwrites from std:: namespace
namespace std {
ostream& operator<<(ostream & os, const yann::test::MnistTest & test);
};// namespace std

#endif /* MNISTTEST_H_ */

