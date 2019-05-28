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

#include "types.h"

namespace yann {

class Network;


enum OperationMode {
  Operation_Assign = 0,      // zero the output buffer (equivalent to '=' operation)
  Operation_PlusEqual,        // keep the output buffer (equivalent to '+=' operation)
};

// Generic interface for activation functions
class ActivationFunction {
public:
  virtual std::string get_name() const = 0;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign) = 0;
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output) = 0;
  virtual std::unique_ptr<ActivationFunction> copy() const = 0;
}; // class ActivationFunction


// Layer initialization mode
enum InitMode
{
  InitMode_Zeros = 0,
  InitMode_Random_01,
  InitMode_Random_SqrtInputs,
};

class Layer
{
  friend class Network;

public:
  // Context for processing inputs (i.e. for feedforward)
  class Context
  {
  public:
    Context(const MatrixSize & output_size, const MatrixSize & batch_size)
    {
      BOOST_VERIFY(output_size > 0);
      BOOST_VERIFY(batch_size > 0);
      resize_batch(_output_buffer, batch_size, output_size);
      _output = _output_buffer;
    }
    Context(const RefVectorBatch & output) :
      _output(output)
    {
      BOOST_VERIFY(yann::get_batch_item_size(output) > 0);
      BOOST_VERIFY(yann::get_batch_size(output) > 0);
    }

    inline MatrixSize get_batch_size() const      { return yann::get_batch_size(*_output);  }
    inline MatrixSize get_output_size() const     { return yann::get_batch_item_size(*_output);  }
    inline RefConstVectorBatch get_output() const { return *_output; }
    inline RefVectorBatch get_output()            { return *_output; }

    virtual void reset_state() { }

  private:
    boost::optional<RefVectorBatch> _output;
    VectorBatch                     _output_buffer;
  }; // class Context

public:
  std::string get_info() const;
  virtual void print_info(std::ostream & os) const;

  virtual std::string get_name() const = 0;

  virtual bool is_valid() const { return true; }
  virtual bool is_equal(const Layer& other, double tolerance) const { return true; }
  virtual MatrixSize get_input_size() const = 0;
  virtual MatrixSize get_output_size() const = 0;

  virtual std::unique_ptr<Context> create_context(const MatrixSize & batch_size = 1) const = 0;
  virtual std::unique_ptr<Context> create_context(const RefVectorBatch & output) const = 0;
  virtual std::unique_ptr<Context> create_training_context(const MatrixSize & batch_size = 1) const = 0;
  virtual std::unique_ptr<Context> create_training_context(const RefVectorBatch & output) const = 0;

  virtual void feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode = Operation_Assign) const = 0;
  virtual void backprop(const RefConstVectorBatch & gradient_output,
                        const RefConstVectorBatch & input,
                        boost::optional<RefVectorBatch> gradient_input,
                        Context * context) const = 0;

  virtual void init(enum InitMode mode) = 0;
  virtual void update(Context * context, double learning_factor, double decay_factor) = 0;

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
