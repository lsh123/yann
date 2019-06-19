/*
 * nn.cpp
 *
 */
#include <algorithm>
#include <sstream>
#include <fstream>
#include <string>

#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/functions.h"
#include "core/layer.h"
#include "core/training.h"
#include "layers/contlayer.h"
#include "core/nn.h"

using namespace std;
using namespace boost;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Network implementation
//
ostream& std::operator<<(ostream & os, const Network & nn)
{
  nn.write(os);
  return os;
}
istream& std::operator>>(istream & is, Network & nn)
{
  nn.read(is);
  return is;
}

yann::Network::Network() :
    _container(new SequentialLayer()),
    _cost_function(new QuadraticCost())
{
}

yann::Network::Network(std::unique_ptr<SequentialLayer> container) :
    _container(std::move(container)),
    _cost_function(new QuadraticCost())
{
  YANN_CHECK(_container);
}

yann::Network::~Network()
{
}

void yann::Network::save(const string & filename) const
{
  ofstream ofs(filename, ofstream::out | ofstream::trunc);
  if(!ofs || ofs.fail()) {
    throw runtime_error("can't open file " + filename);
  }
  ofs << (*this);
  if(!ofs || ofs.fail()) {
    throw runtime_error("can't write to file " + filename);
  }
  ofs.close();

}
void yann::Network::load(const std::string & filename)
{
  ifstream ifs(filename, ifstream::in);
  if(!ifs || ifs.fail()) {
    throw runtime_error("can't open file " + filename);
  }
  ifs >> (*this);
  if(!ifs || ifs.fail()) {
    throw runtime_error("can't read file " + filename);
  }
  ifs.close();
}

string yann::Network::get_info() const
{
  YANN_CHECK(_container);

  ostringstream oss;
  oss << "yann::Network"
      << "[" << get_input_size() << " -> " << get_output_size() << "]"
      << " cost: " << _cost_function->get_info()
      << ", layers: (" << _container->get_info() << ")"
  ;
  return oss.str();
}

bool yann::Network::is_valid() const
{
  if(!_container) {
    return false;
  }
  if(!_cost_function) {
    return false;
  }
  return true;
}

bool yann::Network::is_equal(const Network& other, double tolerance) const
{
  YANN_CHECK(_container);
  YANN_CHECK(other._container);

  // TODO: add deep copy comparison
  if(_cost_function->get_info() != other._cost_function->get_info()) {
    return false;
  }
  if(!_container->is_equal(*other._container, tolerance)) {
    return false;
  }
  return true;
}

MatrixSize yann::Network::get_input_size() const
{
  YANN_CHECK(_container);
  return _container->get_input_size();
}

MatrixSize yann::Network::get_output_size() const
{
  YANN_CHECK(_container);
  return _container->get_output_size();
}

size_t yann::Network::get_layers_num() const
{
  YANN_CHECK(_container);
  return _container->get_layers_num();
}

const Layer * yann::Network::get_layer(const size_t & pos) const
{
  YANN_CHECK(_container);
  return _container->get_layer(pos);
}

Layer * yann::Network::get_layer(const size_t & pos)
{
  YANN_CHECK(_container);
  return _container->get_layer(pos);
}

void yann::Network::append_layer(unique_ptr<Layer> layer)
{
  YANN_CHECK(_container);
  _container->append_layer(std::move(layer));
}

void yann::Network::set_cost_function(const unique_ptr<CostFunction> & cost_function)
{
  YANN_CHECK(cost_function);
  _cost_function = cost_function->copy();
}


Value yann::Network::cost(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(is_same_size(actual, expected));
  return _cost_function->f(actual, expected);
}

unique_ptr<yann::Context> yann::Network::create_context(const MatrixSize & batch_size) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(batch_size, 0);

  auto ctx = make_unique<Context>();
  YANN_CHECK(ctx);

  ctx->_container_ctx = _container->create_context(batch_size);
  YANN_CHECK(ctx->_container_ctx);
  return ctx;
}

unique_ptr<yann::Context> yann::Network::create_context(const RefVectorBatch & output) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(output), 0);
  YANN_CHECK_GT(get_batch_item_size(output), 0);

  auto ctx = make_unique<Context>();
  YANN_CHECK(ctx);

  ctx->_container_ctx = _container->create_context(output);
  YANN_CHECK(ctx->_container_ctx);
  return ctx;
}

unique_ptr<yann::TrainingContext> yann::Network::create_training_context(
    const MatrixSize & batch_size,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(batch_size, 0);

  auto ctx = make_unique<TrainingContext>(batch_size, get_output_size());
  YANN_CHECK(ctx);

  ctx->_container_ctx = _container->create_training_context(batch_size, updater);
  YANN_CHECK(ctx->_container_ctx);
  return ctx;
}


unique_ptr<yann::TrainingContext> yann::Network::create_training_context(
    const RefVectorBatch & output,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(output), 0);
  YANN_CHECK_EQ(get_batch_item_size(output), get_output_size());

  auto ctx = make_unique<TrainingContext>(get_batch_size(output), get_output_size());
  YANN_CHECK(ctx);

  ctx->_container_ctx = _container->create_training_context(output, updater);
  YANN_CHECK(ctx->_container_ctx);
  return ctx;
}


template<typename InputType>
void yann::Network::feedforward(
    const InputType & input,
    Context * ctx) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(ctx);
  YANN_CHECK(ctx->is_valid());

  _container->feedforward(
      input,
      ctx->_container_ctx.get(),
      Operation_Assign);
}

template<typename InputType>
void yann::Network::backprop(
    const InputType & input,
    const RefConstVectorBatch & output,
    TrainingContext * ctx) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(ctx);

  ////////////////////////////////////////////////////////////////////
  // Back propagation: push "gradient" from outputs to inputs
  // starting from the cost_derivative(actual, expected) output and moving
  // back through the network
  //

  // The initial delta is the cost function derivative for the output
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(output));
  auto batch_size = get_batch_size(input);

  YANN_CHECK_LE(batch_size, get_batch_size(ctx->get_output()));
  YANN_CHECK_EQ(get_batch_item_size(output), get_batch_item_size(ctx->get_output()));
  YANN_CHECK_LE(batch_size, get_batch_size(ctx->_output_gradient));
  YANN_CHECK_EQ(get_batch_item_size(output), get_batch_item_size(ctx->_output_gradient));

  RefVectorBatch output_gradient = ctx->_output_gradient.topRows(batch_size); // RowMajor
  _cost_function->derivative(ctx->get_output(batch_size), output, output_gradient);

  // we don't want to calculate the last delta thus no matrix for the input
  // gradient
  _container->backprop(
      output_gradient,
      input,
      optional<RefVectorBatch>(),
      ctx->_container_ctx.get());
}

// This is a slow but convinient method, don't use it in real life
void yann::Network::calculate(const RefConstVectorBatch & input, RefVectorBatch output) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(input), 0);
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());

  // write directly to the output
  std::unique_ptr<Context> ctx(create_context(output));
  feedforward(input, ctx.get());
}

void yann::Network::calculate(const RefConstSparseVectorBatch & input, RefVectorBatch output) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(input), 0);
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());

  // write directly to the output
  std::unique_ptr<Context> ctx(create_context(output));
  feedforward(input, ctx.get());
}

void yann::Network::calculate(const RefConstVectorBatch & input, Context * ctx) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(ctx);
  feedforward(input, ctx);
}

void yann::Network::calculate(const RefConstSparseVectorBatch & input, Context * ctx) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(ctx);
  feedforward(input, ctx);
}

void yann::Network::init(enum Layer::InitMode mode, boost::optional<Layer::InitContext> init_context)
{
  YANN_CHECK(is_valid());
  _container->init(mode, init_context);
}

template<typename InputType>
Value yann::Network::train_internal(
    const InputType & input,
    const RefConstVectorBatch & output,
    TrainingContext * ctx) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(ctx);
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(output));
  YANN_CHECK_LE(get_batch_size(input), ctx->get_batch_size());

  feedforward(input, ctx);
  Value result_cost = cost(ctx->get_output(get_batch_size(input)), output);
  backprop(input, output, ctx);
  return result_cost;
}

Value yann::Network::train(
    const RefConstVectorBatch & input,
    const RefConstVectorBatch & output,
    TrainingContext * ctx) const
{
  return train_internal(input, output, ctx);
}

Value yann::Network::train(
    const RefConstSparseVectorBatch & input,
    const RefConstVectorBatch & output,
    TrainingContext * ctx) const
{
  return train_internal(input, output, ctx);
}

void yann::Network::update(const TrainingContext * ctx, const size_t & batch_size)
{
  YANN_CHECK(is_valid());
  YANN_CHECK(ctx);
  _container->update(ctx->_container_ctx.get(), batch_size);
}

// format:
// (<layer0><layer1>...<layerN>)
void yann::Network::read(std::istream & is)
{
  YANN_CHECK(is_valid());
  _container->read(is);
}

// format:
// (<layer0><layer1>...<layerN>)
void yann::Network::write(std::ostream & os) const
{
  YANN_CHECK(is_valid());
  _container->write(os);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Context implementation
//
yann::Context::Context()
{
}

yann::Context::~Context()
{
}

MatrixSize yann::Context::get_batch_size() const
{
  YANN_CHECK(is_valid());
  auto ctx = dynamic_cast<const SequentialLayer::Context*>(_container_ctx.get());
  YANN_CHECK(ctx);
  return ctx->get_batch_size();
}

RefConstVectorBatch yann::Context::get_output() const
{
  YANN_CHECK(is_valid());
  auto ctx = dynamic_cast<const SequentialLayer::Context*>(_container_ctx.get());
  YANN_CHECK(ctx);
  return ctx->get_output();
}

RefConstVectorBatch yann::Context::get_output(const MatrixSize & pos) const
{
  YANN_CHECK(is_valid());
  auto ctx = dynamic_cast<const SequentialLayer::Context*>(_container_ctx.get());
  YANN_CHECK(ctx);
  return ctx->get_output(pos);
}

void yann::Context::reset_state()
{
  YANN_CHECK(is_valid());
  _container_ctx->reset_state();
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TrainingContext implementation
//
yann::TrainingContext::TrainingContext(const MatrixSize & batch_size, const MatrixSize & output_size)
{
  resize_batch(_output_gradient, batch_size, output_size);
}

yann::TrainingContext::~TrainingContext()
{
}
