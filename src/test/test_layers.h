/*
 * test_layers.h
 *
 */

#ifndef TEST_LAYERS__H_
#define TEST_LAYERS__H_

#include <boost/assert.hpp>

#include "nnlayer.h"

namespace yann{
namespace test {

// Avgs N inputs into 1 output
class AvgLayer : public Layer
{
  typedef Layer Base;

  static const MatrixSize _output_size = 1; // always 1 output

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
  virtual std::unique_ptr<Layer::Context> create_training_context(const MatrixSize & batch_size) const;
  virtual std::unique_ptr<Layer::Context> create_training_context(const RefVectorBatch & output) const;

  virtual void feedforward(const RefConstVectorBatch & input, Layer::Context * context, enum OperationMode mode) const;
  virtual void backprop(const RefConstVectorBatch & gradient_output, const RefConstVectorBatch & input,
                        boost::optional<RefVectorBatch> gradient_input, Layer::Context * context) const;

  virtual void init(enum InitMode mode);
  virtual void update(Layer::Context * context, double learning_factor, double decay_factor);

  virtual void read(std::istream & is);
  virtual void write(std::ostream & os) const;

private:
  MatrixSize _input_size;
  Value      _value; // just so we can test is_equal()

  static size_t g_counter;
}; // class AvgLayer

}  // namespace test
}; // namespace yann

#endif /* TEST_LAYERS__H_ */
