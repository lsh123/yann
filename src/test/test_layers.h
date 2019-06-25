/*
 * test_layers.h
 *
 */

#ifndef TEST_LAYERS__H_
#define TEST_LAYERS__H_

#include <boost/assert.hpp>

#include "core/layer.h"
#include "core/nn.h"

namespace yann::test {

// Avgs N inputs into 1 output
class AvgLayer : public Layer
{
  typedef Layer Base;

  const MatrixSize _output_size = 1; // always 1 output

public:
  AvgLayer(MatrixSize input_size);
  virtual ~AvgLayer();

public:
  Value get_value() const { return _value; }

  // Layer overwrites
  virtual std::string get_name() const;
  virtual bool is_equal(const Layer& other, double tolerance) const;
  virtual MatrixSize get_input_size() const;
  virtual MatrixSize get_output_size() const;

  virtual std::unique_ptr<Layer::Context> create_context(const MatrixSize & batch_size) const;
  virtual std::unique_ptr<Layer::Context> create_context(const RefVectorBatch & output) const;
  virtual std::unique_ptr<Layer::Context> create_training_context(
      const MatrixSize & batch_size,
      const std::unique_ptr<Layer::Updater> & updater) const;
  virtual std::unique_ptr<Layer::Context> create_training_context(
      const RefVectorBatch & output,
      const std::unique_ptr<Layer::Updater> & updater) const;

  virtual void feedforward(
      const RefConstVectorBatch & input,
      Context * context,
      enum OperationMode mode) const;
  virtual void feedforward(
      const RefConstSparseVectorBatch & input,
      Context * context,
      enum OperationMode mode) const;
  virtual void backprop(
      const RefConstVectorBatch & gradient_output,
      const RefConstVectorBatch & input,
      boost::optional<RefVectorBatch> gradient_input,
      Context * context) const;
  virtual void backprop(
      const RefConstVectorBatch & gradient_output,
      const RefConstSparseVectorBatch & input,
      boost::optional<RefVectorBatch> gradient_input,
      Context * context) const;

  virtual void init(enum InitMode mode, boost::optional<InitContext> init_context);
  virtual void update(Layer::Context * context, const size_t & batch_size);

  virtual void read(std::istream & is);
  virtual void write(std::ostream & os) const;

private:
  MatrixSize _input_size;
  Value      _value; // just so we can test is_equal()

  static size_t g_counter;
}; // class AvgLayer

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Helpers for testing layers
//
void test_layer_feedforward(
    Layer & layer,
    const RefConstVectorBatch & input,
    const RefConstVectorBatch & expected_output);

// Adjust the input vector
void test_layer_backprop(
    Layer & layer,
    const RefConstVectorBatch & input,
    boost::optional<RefConstVectorBatch> expected_input,
    const RefConstVectorBatch & expected_output,
    const std::unique_ptr<CostFunction> & cost_func,
    const double learning_rate,
    const size_t & epochs);

// Adjust the input vector from random input
void test_layer_backprop_from_random(
    Layer & layer,
    const MatrixSize & batch_size,
    const std::unique_ptr<CostFunction> & cost_func,
    const double learning_rate,
    const size_t & epochs);

// Adjust layer params
void test_layer_training(
    Layer & layer,
    const RefConstVectorBatch & input,
    const RefConstVectorBatch & expected_output,
    const size_t & tests_num,
    const std::unique_ptr<CostFunction> & cost_func,
    const double learning_rate,
    const size_t & epochs);

// Adjust layer params from random input
void test_layer_training_from_random(
    Layer & layer,
    const MatrixSize & batch_size,
    const size_t & tests_num,
    const std::unique_ptr<CostFunction> & cost_func,
    const double learning_rate,
    const size_t & epochs);

}; // namespace yann::test

#endif /* TEST_LAYERS__H_ */
