/*
 * polllayer.h
 *
 * Polls a submatrix into a single output value
 */

#ifndef POLLLAYER_H_
#define POLLLAYER_H_

#include "nnlayer.h"

namespace yann {

class ParallelLayer;

class PollingLayer : public Layer
{
  typedef Layer Base;

public:
  enum Mode {
    PollMode_Max,
    PollMode_Avg,
  }; // enum Mode

public:
  static MatrixSize get_output_rows(const MatrixSize & input_rows, const MatrixSize & filter_size);
  static MatrixSize get_output_cols(const MatrixSize & input_cols, const MatrixSize & filter_size);
  static MatrixSize get_output_size(const MatrixSize & input_rows, const MatrixSize & input_cols,
                                    const MatrixSize & filter_size);

  // Polling layer: each conv layer sends output to the corresponding poll layer
  static std::unique_ptr<ParallelLayer> create_poll_parallel_layer(
      const size_t & frames_num,
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const MatrixSize & filter_size,
      enum Mode mode);

public:
  PollingLayer(const MatrixSize & input_rows, const MatrixSize & input_cols,
               const MatrixSize & filter_size, enum Mode mode);
  virtual ~PollingLayer();

public:
  // Layer overwrites
  virtual std::string get_name() const;
  virtual void print_info(std::ostream & os) const;
  virtual bool is_equal(const Layer& other, double tolerance) const;
  virtual MatrixSize get_input_size() const;
  virtual MatrixSize get_output_size() const;

  virtual std::unique_ptr<Context> create_context(const MatrixSize & batch_size = 1) const;
  virtual std::unique_ptr<Context> create_context(const RefVectorBatch & output) const;
  virtual std::unique_ptr<Context> create_training_context(const MatrixSize & batch_size = 1) const;
  virtual std::unique_ptr<Context> create_training_context(const RefVectorBatch & output) const;

  virtual void feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode = Operation_Assign) const;
  virtual void backprop(const RefConstVectorBatch & gradient_output, const RefConstVectorBatch & input,
                        boost::optional<RefVectorBatch> gradient_input, Context * context) const;

  virtual void init(enum InitMode mode);
  virtual void update(Context * context, double learning_factor, double decay_factor);

  virtual void read(std::istream & is);
  virtual void write(std::ostream & os) const;

  //
  MatrixSize get_output_rows() const;
  MatrixSize get_output_cols() const;

private:
  static void poll_plus_equal(const RefConstMatrix & input, const MatrixSize & filter_size, enum Mode mode, RefMatrix output);
  static void backprop(const RefConstMatrix & gradient_output, const RefConstVectorBatch & input,
                       const MatrixSize & filter_size, enum Mode mode, RefMatrix gradient_input);

private:
  MatrixSize _input_rows;
  MatrixSize _input_cols;
  MatrixSize _filter_size;
  enum Mode  _mode;
}; // class PollingLayer

}; // namespace yann

#endif /* POLLLAYER_H_ */
