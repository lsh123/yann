/*
 * contlayer.cpp
 *
 * Container layer contains over layers
 */
#include <algorithm>
#include <map>
#include <stdexcept>
#include <boost/assert.hpp>

#include "core/utils.h"
#include "contlayer.h"

using namespace std;
using namespace boost;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// ContainerLayer implementation
//
yann::ContainerLayer::ContainerLayer()
{

}

yann::ContainerLayer::~ContainerLayer()
{

}

const Layer * yann::ContainerLayer::get_layer(const size_t & pos) const
{
  YANN_CHECK_LT(pos,_layers.size());
  return _layers[pos].get();
}

Layer * yann::ContainerLayer::get_layer(const size_t & pos)
{
  YANN_CHECK_LT(pos, _layers.size());
  return _layers[pos].get();
}

void yann::ContainerLayer::append_layer(std::unique_ptr<Layer> layer)
{
  YANN_CHECK(layer);
  _layers.push_back(std::move(layer));
}

// Layer overwrites
bool yann::ContainerLayer::is_valid() const
{
  if(!Base::is_valid()) {
    return false;
  }
  if(get_layers_num() <= 0) {
    return false;
  }
  for(auto & layer : _layers) {
    if(!layer->is_valid()) {
      return false;
    }
  }
  return true;
}

bool yann::ContainerLayer::is_equal(const Layer & other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
    return false;
  }
  auto the_other = dynamic_cast<const ContainerLayer*>(&other);
  if(the_other == nullptr) {
    return false;
  }
  if(get_layers_num() != the_other->get_layers_num()) {
    return false;
  }
  for(size_t ii = 0; ii < get_layers_num(); ++ii) {
    if(!_layers[ii]->is_equal(*the_other->_layers[ii], tolerance)) {
      return false;
    }
  }
  return true;
}

void yann::ContainerLayer::init(enum InitMode mode, boost::optional<InitContext> init_context)
{
  YANN_CHECK(is_valid());
  for(auto & layer: _layers) {
    layer->init(mode, init_context ? boost::optional<InitContext>(init_context->next()) : boost::none);
  }
}

// the format is (<layer0>,<layer1>,...
void yann::ContainerLayer::read(std::istream & is)
{
  Base::read(is);

  read_char(is, '(');

  size_t ii = 0;
  for(auto & layer: _layers) {
    if(ii > 0) {
      read_char(is, ',');
    }
    layer->read(is);
    ++ii;
  }
  read_char(is, ')');
}

// the format is (<layer0>,<layer1>,...)
void yann::ContainerLayer::write(std::ostream & os) const
{
  Base::write(os);

  os << '(';
  size_t ii = 0;
  for(auto & layer: _layers) {
    if(ii > 0) {
      os << ",";
    }
    layer->write(os);
    ++ii;
  }
  os << ')';
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// SequentialLayer_Context implementation
//
namespace yann {

class SequentialLayer_Context : public Layer::Context
{
  typedef Layer::Context Base;

public:
  typedef std::vector<std::unique_ptr<Layer::Context>> Contexts;

public:
  SequentialLayer_Context(const MatrixSize & output_size,
                          const MatrixSize & batch_size) :
    Base(output_size, batch_size)
  {
  }

  SequentialLayer_Context(const RefVectorBatch & output):
    Base(output)
  {
  }

  inline  void append_context(std::unique_ptr<Layer::Context> context)
  {
    YANN_CHECK(_contexts.empty() || _contexts.back()->get_batch_size() == context->get_batch_size());
    _contexts.push_back(std::move(context));
  }

  inline size_t get_contexts_num() const
  {
    return _contexts.size();
  }
  inline Layer::Context * get_context(const size_t & pos)
  {
    YANN_CHECK_LT(pos, _contexts.size());
    return _contexts[pos].get();
  }
  inline const Layer::Context * get_context(const size_t & pos) const
  {
    YANN_CHECK_LT(pos, _contexts.size());
    return _contexts[pos].get();
  }

  inline MatrixSize get_batch_size() const
  {
    return !_contexts.empty() ? _contexts.back()->get_batch_size() : 0;
  }

  // Layer::Context overwrites
  virtual void start_epoch()
  {
    Base::start_epoch();

    for(auto & context : _contexts) {
      context->start_epoch();
    }
  }
  virtual void reset_state()
  {
    Base::reset_state();

    for(auto & context : _contexts) {
      context->reset_state();
    }
  }

protected:
  Contexts _contexts;
}; // class SequentialLayer_Context

class SequentialLayer_TrainingContext :
    public SequentialLayer_Context
{
  typedef SequentialLayer_Context Base;

  friend class SequentialLayer;

  typedef std::vector<VectorBatch> Gradients;

public:
  SequentialLayer_TrainingContext(const MatrixSize & output_size,
                                  const MatrixSize & batch_size,
                                  const size_t layers_num) :
    Base(output_size, batch_size),
    _output_gradients(layers_num)
  {
  }

  SequentialLayer_TrainingContext(const RefVectorBatch & output,
                                  const size_t layers_num) :
    Base(output),
    _output_gradients(layers_num)
  {
  }

  inline Gradients & output_gradients()
  {
    return _output_gradients;
  }

  inline void resize_output_gradient(const size_t & pos, const MatrixSize & batch_size,  const MatrixSize & output_size)
  {
    YANN_CHECK_LT(pos, _output_gradients.size());
    resize_batch(_output_gradients[pos], batch_size, output_size);
  }

  inline RefVectorBatch output_gradient(const size_t & pos, const MatrixSize & batch_size)
  {
    YANN_CHECK_LT(pos, _output_gradients.size());
    return _output_gradients[pos].topRows(batch_size); // RowMajor
  }

private:
  Gradients   _output_gradients;
}; // SequentialLayer_TrainingContext



}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::SequentialLayer implementation
//
yann::SequentialLayer::SequentialLayer()
{
}

yann::SequentialLayer::~SequentialLayer()
{
}

// ContainerLayer overwrites
std::string yann::SequentialLayer::get_name() const
{
  return "SequentialLayer";
}

string yann::SequentialLayer::get_info() const
{
  ostringstream oss;
  oss << Base::get_info()
      << " layers[" << get_layers_num() << "]";
  if(get_layers_num() > 0) {
    oss << ": (";
    for(size_t ii = 0; ii < get_layers_num(); ++ii) {
      auto layer = get_layer(ii);
      YANN_CHECK(layer);
      if(ii > 0) {
        oss << "; ";
      }
      oss << layer->get_info();
    }
    oss << ")";
  }
  return oss.str();
}

// 1 input -> 1 output
MatrixSize yann::SequentialLayer::get_input_size() const
{
  return get_layers_num() > 0 ? get_layer(0)->get_input_size() : 0;
}

MatrixSize yann::SequentialLayer::get_output_size() const
{
  return get_layers_num() > 0 ? get_layer(get_layers_num() - 1)->get_output_size() : 0;
}

// prev output is input to the next
void yann::SequentialLayer::append_layer(std::unique_ptr<Layer> layer)
{
  YANN_CHECK(layer);
  YANN_CHECK(get_layers_num() == 0 || get_output_size() == layer->get_input_size());

  Base::append_layer(std::move(layer));
}

unique_ptr<Layer::Context> yann::SequentialLayer::create_context(const MatrixSize & batch_size) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(batch_size, 0);

  auto ctx = make_unique<SequentialLayer_Context>(get_output_size(), batch_size);
  YANN_CHECK(ctx);

  for(size_t ii = 0; ii < get_layers_num(); ++ii) {
    auto layer = get_layer(ii);
    YANN_CHECK(layer);

    // the last context should point to the shared buffer
    if(ii + 1 < get_layers_num()) {
      auto layer_context = layer->create_context(batch_size);
      ctx->append_context(std::move(layer_context));
    } else {
      auto layer_context = layer->create_context(ctx->get_output());
      ctx->append_context(std::move(layer_context));
    }
  }
  return ctx;
}

unique_ptr<Layer::Context> yann::SequentialLayer::create_context(const RefVectorBatch & output) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(output), 0);
  YANN_CHECK_GT(get_batch_item_size(output), 0);

  auto ctx = make_unique<SequentialLayer_Context>(output);
  YANN_CHECK(ctx);

  auto batch_size = get_batch_size(output);
  for(size_t ii = 0; ii < get_layers_num(); ++ii) {
    auto layer = get_layer(ii);
    YANN_CHECK(layer);

    // the last context should point to the shared buffer
    if(ii + 1 < get_layers_num()) {
      auto layer_context = layer->create_context(batch_size);
      ctx->append_context(std::move(layer_context));
    } else {
      auto layer_context = layer->create_context(ctx->get_output());
      ctx->append_context(std::move(layer_context));
    }
  }
  return ctx;
}

unique_ptr<Layer::Context> yann::SequentialLayer::create_training_context(const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(batch_size, 0);

  auto ctx = make_unique<SequentialLayer_TrainingContext>(get_output_size(), batch_size, get_layers_num());
  YANN_CHECK(ctx);
  YANN_CHECK_EQ(ctx->output_gradients().size(), get_layers_num());

  for(size_t ii = 0; ii < get_layers_num(); ++ii) {
    auto layer = get_layer(ii);
    YANN_CHECK(layer);

    // the last context should point to the shared buffer
    if(ii + 1 < get_layers_num()) {
      auto layer_context = layer->create_training_context(batch_size, updater);
      ctx->append_context(std::move(layer_context));
    } else {
      auto layer_context = layer->create_training_context(ctx->get_output(), updater);
      ctx->append_context(std::move(layer_context));
    }

    // setup output gradients buffers
    ctx->resize_output_gradient(ii, batch_size, layer->get_output_size());
  }
  return ctx;
}

unique_ptr<Layer::Context> yann::SequentialLayer::create_training_context(const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(output), 0);
  YANN_CHECK_GT(get_batch_item_size(output), 0);

  auto ctx = make_unique<SequentialLayer_TrainingContext>(output, get_layers_num());
  YANN_CHECK(ctx);
  YANN_CHECK_EQ(ctx->output_gradients().size(), get_layers_num());

  auto batch_size = get_batch_size(output);
  for(size_t ii = 0; ii < get_layers_num(); ++ii) {
    auto layer = get_layer(ii);
    YANN_CHECK(layer);

    // the last context should point to the shared buffer
    if(ii + 1 < get_layers_num()) {
      auto layer_context = layer->create_training_context(batch_size, updater);
      ctx->append_context(std::move(layer_context));
    } else {
      auto layer_context = layer->create_training_context(ctx->get_output(), updater);
      ctx->append_context(std::move(layer_context));
    }

    // setup output gradients buffers
    ctx->resize_output_gradient(ii, batch_size, layer->get_output_size());
  }
  return ctx;
}

template<typename InputType>
void yann::SequentialLayer::feedforward_internal(
    const InputType & input,
    Layer::Context * context,
    enum OperationMode mode) const
{
  auto ctx = dynamic_cast<SequentialLayer_Context *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_EQ(get_layers_num(), ctx->get_contexts_num());

  ////////////////////////////////////////////////////////////////////
  // Forward propagation, save intermediate state
  for(size_t ii = 0; ii < get_layers_num(); ++ii) {
     auto layer = get_layer(ii);
     auto layer_ctx = ctx->get_context(ii);
     auto layer_mode = (ii + 1 < get_layers_num()) ? Operation_Assign : mode; // only the last layer matters
     if(ii > 0) {
       // prev layer output is the input for cur layer
       const auto * prev_layer_ctx = ctx->get_context(ii - 1);
       layer->feedforward(prev_layer_ctx->get_output(), layer_ctx, layer_mode);
     } else {
       // for the first layer, we use external input
       layer->feedforward(input, layer_ctx, layer_mode);
     }
  }
}

void yann::SequentialLayer::feedforward(
    const RefConstVectorBatch & input,
    Layer::Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

void yann::SequentialLayer::feedforward(
    const RefConstSparseVectorBatch & input,
    Layer::Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

template<typename InputType>
void yann::SequentialLayer::backprop_internal(
    const RefConstVectorBatch & gradient_output,
    const InputType & input,
    boost::optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  auto ctx = dynamic_cast<SequentialLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_SLOW_CHECK_EQ(get_layers_num(), ctx->get_contexts_num());
  YANN_SLOW_CHECK_EQ(get_batch_item_size(gradient_output), get_output_size());
  YANN_SLOW_CHECK(!gradient_input || get_batch_item_size(input) == get_batch_item_size(*gradient_input));
  YANN_SLOW_CHECK_EQ(get_layers_num(),ctx->get_contexts_num());
  YANN_SLOW_CHECK_GT(get_layers_num(), 0);

  YANN_SLOW_CHECK_EQ(get_batch_size(input), get_batch_size(gradient_output));
  YANN_SLOW_CHECK(!gradient_input || get_batch_size(input) == get_batch_size(*gradient_input));
  YANN_SLOW_CHECK_LE(get_batch_size(input), context->get_batch_size());
  auto batch_size = get_batch_size(input);

  ////////////////////////////////////////////////////////////////////
  // Back propagation: push "gradient" from outputs to inputs
  for(size_t ii = get_layers_num(); ii > 0; --ii) {
    auto layer = get_layer(ii - 1);
    auto layer_ctx = ctx->get_context(ii - 1);
    auto layer_gradient_out = ctx->output_gradient(ii - 1, batch_size);

    // the last and first layer require special handling
    if(1 < ii && ii < get_layers_num()) {
      // prev output gradient  == current gradient input;
      // prev output = current input
      auto prev_gradient_out= ctx->output_gradient(ii - 2, batch_size);
      const auto * prev_ctx = ctx->get_context(ii - 2);
      layer->backprop(
          layer_gradient_out,
          prev_ctx->get_output(batch_size),
          optional<RefVectorBatch>(prev_gradient_out),
          layer_ctx);
    } else if(1 < ii && ii == get_layers_num()) {
      // this is the last layer, the output gradient comes from outside,
      // otherwise it is the same
      auto prev_gradient_out= ctx->output_gradient(ii - 2, batch_size);
      const auto * prev_ctx = ctx->get_context(ii - 2);
      layer->backprop(
          gradient_output,
          prev_ctx->get_output(batch_size),
          optional<RefVectorBatch>(prev_gradient_out),
          layer_ctx);
    } else if(1 == ii && ii < get_layers_num()) {
      // this is the first layer, the input comes from outside
      layer->backprop(layer_gradient_out, input, gradient_input, layer_ctx);
    } else {
      // this is the case of one layer, both input and output come from outside
      YANN_SLOW_CHECK_EQ(ii, 1);
      YANN_SLOW_CHECK_EQ(get_layers_num(), 1);
      layer->backprop(gradient_output, input, gradient_input, layer_ctx);
    }
  }
}

void yann::SequentialLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Layer::Context * context) const
{
  backprop_internal(gradient_output, input, gradient_input, context);
}

void yann::SequentialLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstSparseVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Layer::Context * context) const
{
  backprop_internal(gradient_output, input, gradient_input, context);
}

void yann::SequentialLayer::update(yann::Layer::Context * context, const size_t & batch_size)
{
  auto ctx = dynamic_cast<SequentialLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_EQ(get_layers_num(), ctx->get_contexts_num());

  size_t ii = 0;
  for(auto & layer: layers()) {
    auto layer_ctx = ctx->get_context(ii);
    layer->update(layer_ctx, batch_size);
    ++ii;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// MappingLayer_Context implementation
//
namespace yann {

class MappingLayer_Context : public Layer::Context
{
  typedef Layer::Context Base;

public:
  typedef map<size_t, unique_ptr<Layer::Context>> LayerContexts; // input frame -> context
  typedef vector<LayerContexts> Contexts; // all output contexts by output frame

public:
  MappingLayer_Context(const MatrixSize & layers_num,
                       const MatrixSize & output_size,
                       const MatrixSize & batch_size) :
    Base(output_size, batch_size),
    _contexts(layers_num)
  {
    YANN_CHECK_GT(layers_num, 0);
  }

  MappingLayer_Context(const MatrixSize & layers_num,
                       const RefVectorBatch & output):
    Base(output),
    _contexts(layers_num)
  {
    YANN_CHECK_GT(layers_num, 0);
  }

  inline void add_context(
      const size_t & layer_num,
      const size_t & input_frame_num,
      std::unique_ptr<Layer::Context> context)
  {
    YANN_CHECK_LT(layer_num, _contexts.size());
    _contexts[layer_num][input_frame_num] = std::move(context);
  }

  inline Context * get_context(
      const size_t & layer_num,
      const size_t & input_frame_num)
  {
    YANN_CHECK_LT(layer_num, _contexts.size());
    auto it = _contexts[layer_num].find(input_frame_num);
    YANN_CHECK_NE(it, _contexts[layer_num].end());
    return (*it).second.get();
  }

  // Layer::Context overwrites
  virtual void start_epoch()
  {
    Base::start_epoch();

    for(auto & layer_contexts : _contexts) {
      for(auto & pair_in_ctx : layer_contexts) {
        pair_in_ctx.second->start_epoch();
      }
    }
  }
  virtual void reset_state()
  {
    Base::reset_state();

    for(auto & layer_contexts : _contexts) {
      for(auto & pair_in_ctx : layer_contexts) {
        pair_in_ctx.second->reset_state();
      }
    }
  }

private:
  Contexts _contexts;
}; // class MappingLayer_Context


class MappingLayer_TrainingContext :
    public MappingLayer_Context
{
  typedef MappingLayer_Context Base;

  friend class MappingLayer;

public:
  MappingLayer_TrainingContext(const MatrixSize & layers_num,
                               const MatrixSize & one_input_size,
                               const MatrixSize & output_size,
                               const MatrixSize & batch_size) :
    Base(layers_num, output_size, batch_size)
  {
    YANN_CHECK_GT(one_input_size, 0);
    resize_batch(_gradient_input, batch_size, one_input_size);
  }

  MappingLayer_TrainingContext(const MatrixSize & layers_num,
                               const MatrixSize & one_input_size,
                               const RefVectorBatch & output) :
    Base(layers_num, output)
  {
    YANN_CHECK_GT(one_input_size, 0);
    YANN_CHECK_GT(yann::get_batch_size(output), 0);
    resize_batch(_gradient_input, yann::get_batch_size(output), one_input_size);
  }

protected:
  VectorBatch _gradient_input;
}; // class MappingLayer_TrainingContext

}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::MappingLayer implementation: send the N inputs to M*N outputs:
//    input size: N * <one layer input>
//    output size is N * M * <one layer output>
//
yann::MappingLayer::MappingLayer(const size_t & input_frames):
    _input_frames(input_frames)
{
  YANN_CHECK_GT(input_frames, 0);
}

yann::MappingLayer::~MappingLayer()
{
}


// ContainerLayer overwrites
std::string yann::MappingLayer::get_name() const
{
  return "MappingLayer";
}

string yann::MappingLayer::get_info() const
{
  ostringstream oss;

  oss << Base::get_info();

  // assume that all layers are the same, print only one
  oss << " input frames: " << _input_frames
      << " layers[" << get_layers_num() << "]";
  if(get_layers_num() > 0) {
    // TODO: print mapping
    oss << ": (" << get_layer(0)->get_info() << ")";
  }

  return oss.str();
}

bool yann::MappingLayer::is_equal(const Layer& other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
    return false;
  }
  auto the_other = dynamic_cast<const MappingLayer*>(&other);
  if(the_other == nullptr) {
    return false;
  }
  if(_input_frames != the_other->_input_frames) {
    return false;
  }
  if(_mappings != the_other->_mappings) {
    return false;
  }
  return true;
}

// note that the number of layers should be a multiply of the number
// of input frames
bool yann::MappingLayer::is_valid() const
{
  if(!Base::is_valid()) {
    return false;
  }
  if(_input_frames <= 0) {
    return false;
  }
  if(_mappings.size() != get_layers_num()) {
    return false;
  }
  return true;
}

// Mapping N input layers into M output layers, assume
// all layers have same input and output sizes
MatrixSize yann::MappingLayer::get_input_size() const
{
  YANN_CHECK(is_valid());
  return _input_frames * get_layer(0)->get_input_size();
}

MatrixSize yann::MappingLayer::get_output_size() const
{
  YANN_CHECK(is_valid());
  return get_layers_num() * get_layer(0)->get_output_size();
}

// all layers have same input and output sizes
void yann::MappingLayer::append_layer(std::unique_ptr<Layer> layer, const InputsMapping & mapping)
{
  // can't check validity here since it depends on the number of layers
  YANN_CHECK(layer);
  YANN_CHECK(!mapping.empty());
  YANN_CHECK_LT(*std::max_element(mapping.begin(), mapping.end()), _input_frames);
  YANN_CHECK(get_layers_num() == 0 || get_layer(0)->get_input_size() == layer->get_input_size());
  YANN_CHECK(get_layers_num() == 0 || get_layer(0)->get_output_size() == layer->get_output_size());

  Base::append_layer(std::move(layer));
  _mappings.push_back(mapping);
}

void yann::MappingLayer::append_layer(std::unique_ptr<Layer> layer)
{
  throw logic_error("can not add layer without mapping");
}

// we assume all layers have same input size
template<typename InputType>
InputType yann::MappingLayer::get_input_block(InputType input, const size_t & input_frame, const MatrixSize & input_size) const
{
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK_LT(input_frame, _input_frames);

  auto batch_size = get_batch_size(input);
  MatrixSize input_pos = input_size * input_frame;
  YANN_CHECK_LE(input_pos + input_size, get_input_size());
  return input.block(0, input_pos, batch_size, input_size); // Assumes RowMajor layout
}

// we assume all layers have same output size
template<typename OutputType>
OutputType yann::MappingLayer::get_output_block(OutputType output, MatrixSize & output_pos, const MatrixSize & output_size) const
{
  YANN_CHECK_EQ(get_batch_item_size(output), get_output_size());

  auto batch_size = get_batch_size(output);
  auto cur_output_pos = output_pos;
  output_pos += output_size;
  YANN_CHECK_LE(output_pos, get_output_size());
  return output.block(0, cur_output_pos, batch_size, output_size); // Assumes RowMajor layout
}

unique_ptr<Layer::Context> yann::MappingLayer::create_context(const MatrixSize & batch_size) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(batch_size, 0);

  auto ctx = make_unique<MappingLayer_Context>(
      get_layers_num(),
      get_output_size(),
      batch_size);
  MatrixSize output_pos = 0;
  size_t layer_num = 0;
  for(auto & layer: layers()) {
    auto out = get_output_block(ctx->get_output(), output_pos, layer->get_output_size());
    const auto & input_mapping = _mappings[layer_num];
    for(const auto & input_frame : input_mapping) {
      auto layer_context = layer->create_context(out); // we will use += operation on the output block
      ctx->add_context(layer_num, input_frame, std::move(layer_context));
    }
    ++layer_num;
  }
  YANN_CHECK_EQ(output_pos, get_output_size());

  return ctx;
}

unique_ptr<Layer::Context> yann::MappingLayer::create_context(const RefVectorBatch & output) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(output), 0);
  YANN_CHECK_GT(get_batch_item_size(output), 0);

  auto ctx = make_unique<MappingLayer_Context>(
      get_layers_num(),
      output);
  MatrixSize output_pos = 0;
  size_t layer_num = 0;
  for(auto & layer: layers()) {
    auto out = get_output_block(ctx->get_output(), output_pos, layer->get_output_size());
    const auto & input_mapping = _mappings[layer_num];
    for(const auto & input_frame : input_mapping) {
      auto layer_context = layer->create_context(out); // we will use += operation on the output block
      ctx->add_context(layer_num, input_frame, std::move(layer_context));
    }
    ++layer_num;
  }
  YANN_CHECK_EQ(output_pos, get_output_size());

  return ctx;
}
unique_ptr<Layer::Context> yann::MappingLayer::create_training_context(const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(batch_size, 0);

  auto one_input_size = get_layer(0)->get_input_size();
  auto ctx = make_unique<MappingLayer_TrainingContext>(
      get_layers_num(),
      one_input_size,
      get_output_size(),
      batch_size);

  MatrixSize output_pos = 0;
  size_t layer_num = 0;
  for(auto & layer: layers()) {
    auto out = get_output_block(ctx->get_output(), output_pos, layer->get_output_size());
    const auto & input_mapping = _mappings[layer_num];
    for(const auto & input_frame : input_mapping) {
      auto layer_context = layer->create_training_context(out, updater); // we will use += operation on the output block
      ctx->add_context(layer_num, input_frame, std::move(layer_context));
    }
    ++layer_num;
  }
  YANN_CHECK_EQ(output_pos, get_output_size());

  return ctx;
}
unique_ptr<Layer::Context> yann::MappingLayer::create_training_context(const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(output), 0);
  YANN_CHECK_GT(get_batch_item_size(output), 0);

  auto one_input_size = get_layer(0)->get_input_size();
  auto ctx = make_unique<MappingLayer_TrainingContext>(
      get_layers_num(),
      one_input_size,
      output);

  MatrixSize output_pos = 0;
  size_t layer_num = 0;
  for(auto & layer: layers()) {
    auto out = get_output_block(ctx->get_output(), output_pos, layer->get_output_size());
    const auto & input_mapping = _mappings[layer_num];
    for(const auto & input_frame : input_mapping) {
      auto layer_context = layer->create_training_context(out, updater); // we will use += operation on the output block
      ctx->add_context(layer_num, input_frame, std::move(layer_context));
    }
    ++layer_num;
  }
  YANN_CHECK_EQ(output_pos, get_output_size());

  return ctx;
}

template<typename InputType>
void yann::MappingLayer::feedforward_internal(
    const InputType & input,
    Layer::Context * context,
    enum OperationMode mode) const
{
  auto ctx = dynamic_cast<MappingLayer_Context *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_EQ(get_batch_size(input), ctx->get_batch_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());

  if(mode == Operation_Assign) {
    ctx->get_output().setZero();
  }
  size_t layer_num = 0;
  for(auto & layer: layers()) {
    const auto & input_mapping = _mappings[layer_num];
    // repeat for each input frame mapped to this layer/output
    for(const auto & input_frame : input_mapping) {
      auto layer_ctx = ctx->get_context(layer_num, input_frame);
      auto input_block = get_input_block(input, input_frame, layer->get_input_size());
      layer->feedforward(input_block, layer_ctx, Operation_PlusEqual); // do not zero output to make it +=
    }
    ++layer_num;
  }
}

void yann::MappingLayer::feedforward(
    const RefConstVectorBatch & input,
    Layer::Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

void yann::MappingLayer::feedforward(
    const RefConstSparseVectorBatch & input,
    Layer::Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

template<typename InputType>
void yann::MappingLayer::backprop_internal(
    const RefConstVectorBatch & gradient_output,
    const InputType & input,
    optional<RefVectorBatch> gradient_input,
    Layer::Context * context) const
{
  auto ctx = dynamic_cast<MappingLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);

  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(gradient_output), 0);
  YANN_CHECK_EQ(get_batch_item_size(gradient_output), get_output_size());
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(gradient_output));
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK(!gradient_input || is_same_size(input, *gradient_input));
  YANN_CHECK_GT(get_layers_num(), 0);

  // we don't need to calculate the gradient(C, a(l)) for the "first" layer (actual inputs)
  if(gradient_input) {
    gradient_input->setZero();
  }

  size_t layer_num = 0;
  MatrixSize gradient_output_pos = 0;
  for(auto & layer: layers()) {
    auto gradient_out = get_output_block(gradient_output, gradient_output_pos, layer->get_output_size());
    const auto & input_mapping = _mappings[layer_num];
    auto inputs_num = input_mapping.size();

    // repeat for each input frame mapped to this layer/output
    for(const auto & input_frame : input_mapping) {
      auto layer_ctx = ctx->get_context(layer_num, input_frame);
      auto input_block = get_input_block(input, input_frame, layer->get_input_size());
      if(gradient_input) {
        auto gradient_input_block = get_input_block((*gradient_input), input_frame, layer->get_input_size());
        YANN_CHECK(is_same_size(gradient_input_block, ctx->_gradient_input));

        layer->backprop(gradient_out, input_block, optional<RefVectorBatch>(ctx->_gradient_input), layer_ctx);
        gradient_input_block.noalias() += (ctx->_gradient_input) / ((Value)inputs_num);
      } else {
        layer->backprop(gradient_out, input_block, optional<RefVectorBatch>(), layer_ctx);
      }
    }
    ++layer_num;
  }
  YANN_CHECK_EQ(gradient_output_pos, get_output_size());
}

void yann::MappingLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Layer::Context * context) const
{
  backprop_internal(gradient_output, input, gradient_input, context);
}

void yann::MappingLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstSparseVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Layer::Context * context) const
{
  backprop_internal(gradient_output, input, gradient_input, context);
}

void yann::MappingLayer::update(yann::Layer::Context * context, const size_t & batch_size)
{
  auto ctx = dynamic_cast<MappingLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());

  size_t layer_num = 0;
  for(auto & layer: layers()) {
    const auto & input_mapping = _mappings[layer_num];
    for(const auto & input_frame : input_mapping) {
      auto layer_ctx = ctx->get_context(layer_num, input_frame);
      layer->update(layer_ctx, batch_size);
    }
    ++layer_num;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::BroadcastLayer implementation: send the 1 input to N outputs:
//
yann::BroadcastLayer::BroadcastLayer():
    Base(1) // always one input
{
}

yann::BroadcastLayer::~BroadcastLayer()
{
}

// ContainerLayer overwrites
std::string yann::BroadcastLayer::get_name() const
{
  return "BroadcastLayer";
}

// map all outputs to the single input frame
void yann::BroadcastLayer::append_layer(std::unique_ptr<Layer> layer)
{
  Base::append_layer(std::move(layer), {0});
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::ParallelLayer implementation: N inputs to N outputs
//
yann::ParallelLayer::ParallelLayer(const size_t & input_frames) :
    Base(input_frames)
{
}

yann::ParallelLayer::~ParallelLayer()
{
}


// Overwrites
std::string yann::ParallelLayer::get_name() const
{
  return "ParallelLayer";
}

// Kth input goes to Kth output
void yann::ParallelLayer::append_layer(std::unique_ptr<Layer> layer)
{
  YANN_CHECK_LT(get_layers_num(), get_input_frames_num());
  Base::append_layer(std::move(layer), {get_layers_num()} );
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::MergeLayer implementation: N inputs to N outputs
//
yann::MergeLayer::MergeLayer(const size_t & input_frames) :
    Base(input_frames)
{
}

yann::MergeLayer::~MergeLayer()
{
}


// Overwrites
std::string yann::MergeLayer::get_name() const
{
  return "MergeLayer";
}

// We have only one output / layer
void yann::MergeLayer::append_layer(std::unique_ptr<Layer> layer)
{
  YANN_CHECK_EQ(get_layers_num(), 0); // only one output/layer is allowed

  // all inputs into one output
  InputsMapping mappings(get_input_frames_num());
  for(size_t ii = 0; ii < get_input_frames_num(); ++ii) {
    mappings[ii] = ii;
  }
  Base::append_layer(std::move(layer), mappings);
}

