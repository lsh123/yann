/*
 * nnlayer.h
 *
 */
#ifndef NNLAYER_H_
#define NNLAYER_H_

#include <iostream>
#include <memory>
#include <string>

#include <boost/optional.hpp>

#include "core/types.h"

namespace yann {

class Network;


enum OperationMode {
  Operation_Assign = 0,      // zero the output buffer (equivalent to '=' operation)
  Operation_PlusEqual,        // keep the output buffer (equivalent to '+=' operation)
};

// Generic interface for activation functions
class ActivationFunction {
public:
  virtual std::string get_info() const = 0;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign) = 0;
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output) = 0;
  virtual std::unique_ptr<ActivationFunction> copy() const = 0;
}; // class ActivationFunction


class Layer
{
  friend class Network;

public:
  // Layer initialization mode
  enum InitMode
  {
    InitMode_Zeros = 0,
    InitMode_Random,
  };
  class InitContext {
  public:
    InitContext(double seed) : _seed(seed) { }
    inline const double & seed() const { return _seed; }
    InitContext next() const { return InitContext(_seed + 1); }

  private:
    double _seed;
  }; // InitContext

  // Context for processing inputs (i.e. for feedforward)
  class Context
  {
  public:
    Context(const MatrixSize & output_size, const MatrixSize & batch_size);
    Context(const RefVectorBatch & output);

    inline MatrixSize get_batch_size() const       { return yann::get_batch_size(*_output);  }
    inline MatrixSize get_output_size() const      { return yann::get_batch_item_size(*_output);  }

    inline RefConstVectorBatch get_output() const { return *_output; }
    inline RefVectorBatch get_output()            { return *_output; }
    inline RefConstVectorBatch get_output(const MatrixSize & pos) const { return _output->topRows(pos); } // RowMajor
    inline RefVectorBatch get_output(const MatrixSize & pos)            { return _output->topRows(pos); } // RowMajor

    virtual void start_epoch() { }
    virtual void reset_state() { }

  protected:
    boost::optional<RefVectorBatch> _output;
    VectorBatch                     _output_buffer;
  }; // class Context

  // Updater interface
  class Updater {
  public:
    virtual std::string get_info() const = 0;
    virtual std::unique_ptr<Layer::Updater> copy() const = 0;

    virtual void init(const MatrixSize & rows, const MatrixSize & cols) = 0;
    virtual void start_epoch() = 0;
    virtual void reset() = 0;
    virtual void update(const RefConstMatrix & delta, const size_t & tests_num, RefMatrix value) = 0;
    virtual void update(const Value & delta, const size_t & tests_num, Value & value) = 0;
  }; // class Updater

public:
  virtual std::string get_info() const;
  virtual std::string get_name() const = 0;

  virtual bool is_valid() const { return true; }
  virtual bool is_equal(const Layer& other, double tolerance) const { return true; }
  virtual MatrixSize get_input_size() const = 0;
  virtual MatrixSize get_output_size() const = 0;

  virtual std::unique_ptr<Context> create_context(const MatrixSize & batch_size) const = 0;
  virtual std::unique_ptr<Context> create_context(const RefVectorBatch & output) const = 0;
  virtual std::unique_ptr<Context> create_training_context(
      const MatrixSize & batch_size,
      const std::unique_ptr<Layer::Updater> & updater) const = 0;
  virtual std::unique_ptr<Context> create_training_context(
      const RefVectorBatch & output,
      const std::unique_ptr<Layer::Updater> & updater) const = 0;

  // Two versions for both feedforward and backprop for Dense and Sparse vectors
  virtual void feedforward(
      const RefConstVectorBatch & input,
      Context * context,
      enum OperationMode mode = Operation_Assign) const = 0;
  virtual void feedforward(
      const RefConstSparseVectorBatch & input,
      Context * context,
      enum OperationMode mode = Operation_Assign) const = 0;
  virtual void backprop(
      const RefConstVectorBatch & gradient_output,
      const RefConstVectorBatch & input,
      boost::optional<RefVectorBatch> gradient_input,
      Context * context) const = 0;
  virtual void backprop(
      const RefConstVectorBatch & gradient_output,
      const RefConstSparseVectorBatch & input,
      boost::optional<RefVectorBatch> gradient_input,
      Context * context) const = 0;

  virtual void init(enum InitMode mode, boost::optional<InitContext> init_context = boost::none) = 0;
  virtual void update(Context * context, const size_t & tests_num) = 0;

  virtual void read(std::istream & is);
  virtual void write(std::ostream & os) const;
}; // class Layer


}; // namespace yann

// Overwrites from std:: namespace
namespace std {
ostream& operator<<(ostream & os, const yann::Layer & layer);
istream& operator>>(istream & is, yann::Layer & layer);
}; // namespace std


#endif /* NNLAYER_H_ */
