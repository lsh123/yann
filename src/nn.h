/*
 * nn.h
 *
 */
#ifndef NN_H_
#define NN_H_

#include <iostream>
#include <memory>
#include <vector>

#include "layer.h"
#include "types.h"

// Forward declarations
namespace yann {
class Layer;
class Network;
class Context;
class TrainingContext;
class SequentialLayer;
}; // namespace yann

// Overwrites from std:: namespace
namespace std {
ostream& operator<<(ostream & os, const yann::Network & nn);
istream& operator>>(istream & is, yann::Network & nn);
}; // namespace std

namespace yann {

// Generic interface for cost functions
class CostFunction {
public:
  virtual std::string get_info() const = 0;
  virtual Value f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected) = 0;
  virtual void derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output) = 0;
  virtual std::unique_ptr<CostFunction> copy() const = 0;
}; // class CostFunction

class Network
{
  friend std::ostream& std::operator<<(std::ostream & os, const Network & nn);
  friend std::istream& std::operator>>(std::istream & is, Network & nn);

public:
  Network();
  Network(std::unique_ptr<SequentialLayer> container);
  virtual ~Network();

  size_t get_layers_num() const;
  MatrixSize get_input_size() const;
  MatrixSize get_output_size() const;

  void save(const std::string & filename) const;
  void load(const std::string & filename);

  void set_cost_function(const std::unique_ptr<CostFunction> & cost_function);

  virtual std::string get_info() const;

  virtual bool is_equal(const Network& other, double tolerance) const;
  virtual bool is_valid() const;

  // Execute
  void calculate(const RefConstVectorBatch & input, RefVectorBatch output) const;
  void calculate(const RefConstVectorBatch & input, Context * ctx) const;
  Value cost(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected) const;

  // Contexts
  std::unique_ptr<Context> create_context(const MatrixSize & batch_size) const;
  std::unique_ptr<Context> create_context(const RefVectorBatch & output) const;
  std::unique_ptr<TrainingContext> create_training_context(const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const;
  std::unique_ptr<TrainingContext> create_training_context(const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const;

  // Training
  virtual void init(enum Layer::InitMode mode, boost::optional<Layer::InitContext> init_context = boost::none);
  virtual void train(const VectorBatch & input, const VectorBatch & output, TrainingContext * ctx) const;
  virtual void update(const TrainingContext * ctx, const size_t & batch_size);

  // access to layers
  const Layer * get_layer(const size_t & pos) const;
  Layer * get_layer(const size_t & pos);
  void append_layer(std::unique_ptr<Layer> layer);

protected:
  void feedforward(const RefConstVectorBatch & input, Context * ctx) const;
  void backprop(const RefConstVectorBatch & input, const RefConstVectorBatch & output, TrainingContext * ctx) const;

  virtual void read(std::istream & is);
  virtual void write(std::ostream & os) const;

private:
  std::unique_ptr<SequentialLayer> _container;
  std::unique_ptr<CostFunction>    _cost_function;
}; // class Network

class Context {
  friend class Network;

public:
  Context();
  virtual ~Context();

  MatrixSize get_batch_size() const;
  RefConstVectorBatch get_output() const;
  void reset_state();
  inline bool is_valid() const {  return _container_ctx.get() != nullptr; }

private:
  std::unique_ptr<Layer::Context> _container_ctx;
}; // class Context

class TrainingContext :
    public Context
{
  friend class Network;

public:
  TrainingContext(const MatrixSize & batch_size, const MatrixSize & output_size);
  virtual ~TrainingContext();

private:
  VectorBatch _output_gradient;
}; // class TrainingContext

}; // namespace yann

#endif /* NN_H_ */
