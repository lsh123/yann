/*
 * s2slayer.cpp
 *
 * Feed-forward:
 *    z(l) = a(l-1) * w + b // where * (vector, matrix) multiplication
 *    a(l) = activation(z(l))
 *
 * Back propagation:
 *    delta(l) = elem_prod(gradient(C, a(l)), activation_derivative(z(l)))
 *    dC/db(l) = delta(l)
 *    dC/dw(l) = a(l-1) * delta(l)
 *    gradient(C, a(l - 1)) = transp(w) * delta(l)
 */
#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/random.h"
#include "core/functions.h"
#include "lstmlayer.h"
#include "s2slayer.h"

using namespace std;
using namespace boost;
using namespace yann;


namespace yann {

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Seq2SeqLayer_Context implementation
//
class Seq2SeqLayer_Context :
    public Layer::Context
{
  typedef Layer::Context Base;

  friend class Seq2SeqLayer;

public:
  Seq2SeqLayer_Context(
      unique_ptr<Layer::Context> encoder_ctx,
      unique_ptr<Layer::Context> decoder_ctx) :
    Base(decoder_ctx->get_output()) // decoder output is our output
  {
    YANN_CHECK(encoder_ctx);
    YANN_CHECK(decoder_ctx);
    _encoder_ctx = std::move(encoder_ctx);
    _decoder_ctx = std::move(decoder_ctx);
  }

  inline bool is_valid() const
  {
    return _encoder_ctx && _decoder_ctx;
  }

  // Layer::Context overwrites
  virtual void start_epoch()
  {
    YANN_CHECK(is_valid());

    Base::start_epoch();

    _encoder_ctx->start_epoch();
    _decoder_ctx->start_epoch();
  }

  virtual void reset_state()
  {
    YANN_CHECK(is_valid());

    Base::reset_state();

    _encoder_ctx->reset_state();
    _decoder_ctx->reset_state();
  }
protected:
  unique_ptr<Layer::Context> _encoder_ctx;
  unique_ptr<Layer::Context> _decoder_ctx;
}; // class Seq2SeqLayer_Context

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Seq2SeqLayer_TrainingContext implementation
//
class Seq2SeqLayer_TrainingContext :
    public Seq2SeqLayer_Context
{
  typedef Seq2SeqLayer_Context Base;
  friend class Seq2SeqLayer;

public:
  Seq2SeqLayer_TrainingContext(
      unique_ptr<Layer::Context> encoder_ctx,
      unique_ptr<Layer::Context> decoder_ctx) :
    Base(std::move(encoder_ctx), std::move(decoder_ctx))
  {
    _decoder_input_gradient.resize(_decoder_ctx->get_output_size());
    _decoder_output_gradient.resize(_decoder_ctx->get_output_size());
  }

protected:
  Vector _decoder_input_gradient;
  Vector _decoder_output_gradient;
}; // class Seq2SeqLayer_TrainingContext

}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::Seq2SeqLayer implementation
//
std::unique_ptr<Seq2SeqLayer> yann::Seq2SeqLayer::create_lstm(
      const MatrixSize & input_size,
      const MatrixSize & output_size,
      const std::unique_ptr<ActivationFunction> & gate_activation_function,
      const std::unique_ptr<ActivationFunction> & io_activation_function)
{
  YANN_CHECK_GT(input_size, 0);
  YANN_CHECK_GT(output_size, 0);
  YANN_CHECK(gate_activation_function);
  YANN_CHECK(io_activation_function);

  auto encoder = make_unique<LstmLayer>(input_size, output_size);
  YANN_CHECK(encoder);
  encoder->set_activation_functions(gate_activation_function, io_activation_function);

  auto decoder = make_unique<LstmLayer>(output_size, output_size);
  YANN_CHECK(decoder);
  decoder->set_activation_functions(gate_activation_function, io_activation_function);

  return make_unique<Seq2SeqLayer>(std::move(encoder), std::move(decoder));
}

std::unique_ptr<Seq2SeqLayer> yann::Seq2SeqLayer::create_lstm(
    const MatrixSize & input_size,
    const MatrixSize & output_size,
    const std::unique_ptr<ActivationFunction> & activation_function)
{
  YANN_CHECK(activation_function);
  return create_lstm(input_size, output_size, activation_function, activation_function);
}


yann::Seq2SeqLayer::Seq2SeqLayer(std::unique_ptr<Layer> encoder, std::unique_ptr<Layer> decoder) :
    _encoder(std::move(encoder)),
    _decoder(std::move(decoder))
{
  YANN_CHECK_EQ(_encoder->get_output_size(), _decoder->get_input_size());
}

yann::Seq2SeqLayer::~Seq2SeqLayer()
{
}

// Layer overwrites
bool yann::Seq2SeqLayer::is_valid() const
{
  if(!Base::is_valid()) {
    return false;
  }
  if(!_encoder || !_encoder->is_valid()) {
    return false;
  }
  if(!_decoder || !_decoder->is_valid()) {
    return false;
  }
  return true;
}

std::string yann::Seq2SeqLayer::get_name() const
{
  return "Seq2SeqLayer";
}

string yann::Seq2SeqLayer::get_info() const
{
  YANN_CHECK(is_valid());

  ostringstream oss;
  oss << Base::get_info()
      << " encoder: " << _encoder->get_info()
      << ", decoder: " << _decoder->get_info()
  ;
  return oss.str();
}

bool yann::Seq2SeqLayer::is_equal(const Layer & other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
    return false;
  }
  auto the_other = dynamic_cast<const Seq2SeqLayer*>(&other);
  if(the_other == nullptr) {
    return false;
  }
  if(is_valid() != the_other->is_valid()) {
    return false;
  }
  if(_encoder && !_encoder->is_equal(*the_other->_encoder, tolerance)) {
    return false;
  }
  if(_decoder && !_decoder->is_equal(*the_other->_decoder, tolerance)) {
    return false;
  }
  return true;
}

MatrixSize yann::Seq2SeqLayer::get_input_size() const
{
  return _encoder ? _encoder->get_input_size() : 0;
}

MatrixSize yann::Seq2SeqLayer::get_output_size() const
{
  return _decoder ? _decoder->get_output_size() : 0;
}

unique_ptr<Layer::Context> yann::Seq2SeqLayer::create_context(const MatrixSize & batch_size) const
{
  YANN_CHECK(is_valid());

  auto encoder_ctx = _encoder->create_context(batch_size);
  YANN_CHECK(encoder_ctx);
  auto decoder_ctx = _decoder->create_context(batch_size);
  YANN_CHECK(decoder_ctx);

  return make_unique<Seq2SeqLayer_Context>(std::move(encoder_ctx), std::move(decoder_ctx));
}
unique_ptr<Layer::Context> yann::Seq2SeqLayer::create_context(const RefVectorBatch & output) const
{
  YANN_CHECK(is_valid());

  auto encoder_ctx = _encoder->create_context(get_batch_size(output));
  YANN_CHECK(encoder_ctx);
  auto decoder_ctx = _decoder->create_context(output); // decoder output is our output
  YANN_CHECK(decoder_ctx);

  return make_unique<Seq2SeqLayer_Context>(std::move(encoder_ctx), std::move(decoder_ctx));
}
unique_ptr<Layer::Context> yann::Seq2SeqLayer::create_training_context(
    const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(updater);

  auto encoder_ctx = _encoder->create_training_context(batch_size, updater);
  YANN_CHECK(encoder_ctx);
  auto decoder_ctx = _decoder->create_training_context(batch_size, updater);
  YANN_CHECK(decoder_ctx);

  return make_unique<Seq2SeqLayer_TrainingContext>(std::move(encoder_ctx), std::move(decoder_ctx));
}
unique_ptr<Layer::Context> yann::Seq2SeqLayer::create_training_context(
    const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(updater);

  auto encoder_ctx = _encoder->create_training_context(get_batch_size(output), updater);
  YANN_CHECK(encoder_ctx);
  auto decoder_ctx = _decoder->create_training_context(output, updater); // decoder output is our output
  YANN_CHECK(decoder_ctx);

  return make_unique<Seq2SeqLayer_TrainingContext>(std::move(encoder_ctx), std::move(decoder_ctx));
}

template<typename InputType>
void yann::Seq2SeqLayer::feedforward_internal(
    const InputType & input,
    Context * context,
    enum OperationMode mode) const
{
  auto ctx = dynamic_cast<Seq2SeqLayer_Context *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(ctx->is_valid());
  YANN_CHECK(is_valid());

  YANN_CHECK_LT(0, get_batch_size(input));
  YANN_CHECK_LE(get_batch_size(input), get_batch_size(ctx->get_output()));
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());

  //
  // we assume all input is given to us "at once" no incremental feedforward
  //
  auto & encoder_ctx = ctx->_encoder_ctx;
  auto & decoder_ctx = ctx->_decoder_ctx;

  // feed input into encoder, overwrite the output
  _encoder->feedforward(input, encoder_ctx.get(), Operation_Assign);
  auto encoder_output_size = get_batch_size(encoder_ctx->get_output());
  YANN_SLOW_CHECK_GT(encoder_output_size, 0);
  auto state = get_batch(encoder_ctx->get_output(), encoder_output_size - 1); // last output

  // feed state into decoder and then feed prev output
  YANN_CHECK_EQ(mode, Operation_Assign); // we don't support anything else because we re-use output buffer
  for(MatrixSize ii = 0; ii < ctx->get_batch_size(); ++ii) {
    if(ii > 0) {
      auto prev_output = get_batch(decoder_ctx->get_output(), ii - 1);
      _decoder->feedforward(prev_output, decoder_ctx.get(), Operation_Assign);
    } else {
      _decoder->feedforward(state, decoder_ctx.get(), Operation_Assign);
    }
  }
}

void yann::Seq2SeqLayer::feedforward(
    const RefConstVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

void yann::Seq2SeqLayer::feedforward(
    const RefConstSparseVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

template<typename InputType>
void yann::Seq2SeqLayer::backprop_internal(
    const RefConstVectorBatch & gradient_output,
    const InputType & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  auto ctx = dynamic_cast<Seq2SeqLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(ctx->is_valid());
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(gradient_output), 0);
  YANN_CHECK_EQ(get_batch_item_size(gradient_output), get_output_size());
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(gradient_output));
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK(!gradient_input || is_same_size(input, *gradient_input));

  //
  // we assume all output is given to us "at once" no incremental backprop
  //
  auto & encoder_ctx = ctx->_encoder_ctx;
  YANN_SLOW_CHECK(encoder_ctx);
  auto & decoder_ctx = ctx->_decoder_ctx;
  YANN_SLOW_CHECK(decoder_ctx);

  auto encoder_output_size = get_batch_size(encoder_ctx->get_output());
  YANN_SLOW_CHECK_GT(encoder_output_size, 0);
  auto state = get_batch(encoder_ctx->get_output(), encoder_output_size - 1); // last output

  // first we propagate back the output for decoder
  auto & decoder_input_gradient = ctx->_decoder_input_gradient;
  auto & decoder_output_gradient = ctx->_decoder_output_gradient;
  YANN_SLOW_CHECK(is_same_size(decoder_input_gradient, decoder_output_gradient));

  decoder_input_gradient.setZero();
  decoder_output_gradient.setZero();

  const auto batch_size = get_batch_size(gradient_output);
  for(MatrixSize ii = batch_size - 1; ii >= 0; --ii) {
    decoder_output_gradient += get_batch(gradient_output, ii);
    if(ii > 0) {
      auto decoder_prev_output = get_batch(decoder_ctx->get_output(), ii - 1);
      _decoder->backprop(
          decoder_output_gradient, // gradient_output
          decoder_prev_output,     // input
          make_optional<RefVectorBatch>(decoder_input_gradient),  // gradient_input
          decoder_ctx.get());
      swap(decoder_input_gradient, decoder_output_gradient);
    } else {
      _decoder->backprop(
          decoder_output_gradient, // gradient_output
          state,                   // input
          make_optional<RefVectorBatch>(decoder_input_gradient),  // gradient_input
          decoder_ctx.get());
    }
    // decoder_input_gradient contains the state gradient
  }

  // now propagate the gradient for encoder
  decoder_output_gradient.setZero();
  for(MatrixSize ii = batch_size - 1; ii >= 0; --ii) {
    auto gradient_in = gradient_input ? make_optional<RefVectorBatch>(get_batch(*gradient_input, ii)) : boost::none;
    InputType in = input.block(ii, 0, 1, input.cols()); // TODO: RowMajor, switch to get_batch()

    _encoder->backprop(
        decoder_input_gradient, // gradient_output
        in,   // input
        gradient_in,            // gradient_input
        encoder_ctx.get());
    if(ii == batch_size - 1) {
      decoder_input_gradient.setZero(); // we don't care about encoder output except the last vector
    }
  }
}

void yann::Seq2SeqLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
 backprop_internal(gradient_output, input, gradient_input, context);
}

void yann::Seq2SeqLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstSparseVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  backprop_internal<RefConstSparseVectorBatch>(gradient_output, input, gradient_input, context);
}

void yann::Seq2SeqLayer::init(enum InitMode mode, boost::optional<InitContext> init_context)
{
  YANN_CHECK(is_valid());
  _encoder->init(mode, init_context);
  _decoder->init(mode, init_context);
}

void yann::Seq2SeqLayer::update(Context * context, const size_t & tests_num)
{
  auto ctx = dynamic_cast<Seq2SeqLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(ctx->is_valid());
  YANN_CHECK(is_valid());

  _encoder->update(ctx->_encoder_ctx.get(), tests_num);
  _decoder->update(ctx->_decoder_ctx.get(), tests_num);
}

// the format is (e:<encoder>,d:<decoder>)
void yann::Seq2SeqLayer::read(std::istream & is)
{
  YANN_CHECK(is_valid());

  Base::read(is);

  read_char(is, '(');
  read_object(is, "e", *_encoder);
  read_char(is, ',');
  read_object(is, "d", *_decoder);
  read_char(is, ')');
}

// the format is (e:<encoder>,d:<decoder>)
void yann::Seq2SeqLayer::write(std::ostream & os) const
{
  YANN_CHECK(is_valid());

  Base::write(os);

  os << "(";
  write_object(os, "e", *_encoder);
  os << ",";
  write_object(os, "d", *_decoder);
  os << ")";
}

