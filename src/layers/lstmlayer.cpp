/*
 * lstmlayer.cpp
 *
 * Feed-forward:
 *    x(t) -- inputs
 *    h(t-1) -- hidden state, aka outputs on prev timestep
 *    c(t) -- cell state
 *
 *    Phase 1: I/O gates
 *      zz_a(t) = ww_xa*x(t) + ww_ha * h(t-1) + bb_a  -- input
 *      zz_i(t) = ww_xi*x(t) + ww_hi * h(t-1) + bb_i  -- input gate
 *      zz_f(t) = ww_xf*x(t) + ww_hf * h(t-1) + bb_g  -- forget gate
 *      zz_o(t) = ww_xo*x(t) + ww_ho * h(t-1) + bb_o  -- output gate
 *
 *      a(t) = activ1 (zz_a(t))
 *      i(t) = activ2 (zz_i(t))
 *      f(t) = activ2 (zz_f(t))
 *      o(t) = activ2 (zz_o(t))
 *
 *    Phase 2: State
 *      c(t) = elem_prod(i(t), a(t)) + elem_prod(f(t), c(t-1))
 *
 *    Phase 3: Output
 *      h(t) = elem_prod(o(t), activ1(c(t))
 *
 * Back propagation:
 *
 */
#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/random.h"
#include "core/functions.h"
#include "lstmlayer.h"

using namespace std;
using namespace boost;
using namespace yann;


namespace yann {

// prefixes for read/write
static const char g_ltsm_names_suffix[LstmLayer::Gate_Max + 1] = "aifo";

////////////////////////////////////////////////////////////////////////////////////////////////
//
// LstmLayer_Context implementation
//
class LstmLayer_Context :
    public Layer::Context
{
  typedef Layer::Context Base;

  friend class LstmLayer;

public:
  LstmLayer_Context(
      const MatrixSize & output_size,
      const MatrixSize & max_batch_size) :
    Base(output_size, max_batch_size),
    _pos(0)
  {
    init();
  }

  LstmLayer_Context(
      const RefVectorBatch & output) :
    Base(output),
    _pos(0)
  {
    init();
  }

  // Layer::Context overwrites
  virtual void reset_state()
  {
    Base::reset_state();

    _pos = 0;
  }

private:
  void init()
  {
    for(auto & zz_gate: _zz_gate) {
      zz_gate.resize(get_batch_size(), get_output_size());
    }
    for(auto & gate: _gate) {
      gate.resize(get_batch_size(), get_output_size());
    }

    _cc.resize(get_batch_size(), get_output_size());
    _activ_cc.resize(get_batch_size(), get_output_size());

    _hh.resize(get_batch_size(), get_output_size());
  }

protected:
  MatrixSize  _pos;
  VectorBatch _zz_gate[LstmLayer::Gate_Max]; // pre-activation function results for each gate
  VectorBatch _gate[LstmLayer::Gate_Max];    // post-activation function results for each gate
  VectorBatch _cc;                      // cell state (save it for each step)
  VectorBatch _activ_cc;                // activ1(c(t))
  VectorBatch _hh;                      // hidden state aka output (save it for each step)
}; // class LstmLayer_Context

////////////////////////////////////////////////////////////////////////////////////////////////
//
// LstmLayer_TrainingContext implementation
//
class LstmLayer_TrainingContext :
    public LstmLayer_Context
{
  typedef LstmLayer_Context Base;
  friend class LstmLayer;

public:
  LstmLayer_TrainingContext(
      const MatrixSize & input_size,
      const MatrixSize & output_size,
      const MatrixSize & batch_size,
      const unique_ptr<Layer::Updater> & updater) :
    Base(output_size, batch_size)
  {
    init(updater, input_size);
  }

  LstmLayer_TrainingContext(
      const MatrixSize & input_size,
      const RefVectorBatch & output,
      const unique_ptr<Layer::Updater> & updater) :
    Base(output)
  {
    init(updater, input_size);
  }

  // Layer::Context overwrites
  virtual void start_epoch()
  {
    Base::start_epoch();

    for(auto & ww_x_updater: _ww_x_updater) {
      YANN_SLOW_CHECK(ww_x_updater);
      ww_x_updater->start_epoch();
    }
    for(auto & ww_h_updater: _ww_h_updater) {
      YANN_SLOW_CHECK(ww_h_updater);
      ww_h_updater->start_epoch();
    }
    for(auto & bb_updater: _bb_updater) {
      YANN_SLOW_CHECK(bb_updater);
      bb_updater->start_epoch();
    }
  }

  virtual void reset_state()
  {
    Base::reset_state();

    for(auto & delta_ww_x: _delta_ww_x) {
      delta_ww_x.setZero();
    }
    for(auto & delta_ww_h: _delta_ww_h) {
      delta_ww_h.setZero();
    }
    for(auto & delta_bb: _delta_bb) {
      delta_bb.setZero();
    }

    for(auto & ww_x_updater: _ww_x_updater) {
      YANN_SLOW_CHECK(ww_x_updater);
      ww_x_updater->reset();
    }
    for(auto & ww_h_updater: _ww_h_updater) {
      YANN_SLOW_CHECK(ww_h_updater);
      ww_h_updater->reset();
    }
    for(auto & bb_updater: _bb_updater) {
      YANN_SLOW_CHECK(bb_updater);
      bb_updater->reset();
    }

    _gradient_hh.setZero();
    _gradient_cc.setZero();
  }

private:
  void init(
      const unique_ptr<Layer::Updater> & updater,
      const MatrixSize & input_size)
  {
    YANN_CHECK_GT(updater, 0);
    YANN_CHECK_GT(input_size, 0);

    // deltas
    for(auto & delta_ww_x: _delta_ww_x) {
      delta_ww_x.resize(input_size, get_output_size());
    }
    for(auto & delta_ww_h: _delta_ww_h) {
      delta_ww_h.resize(get_output_size(), get_output_size());
    }
    for(auto & delta_bb: _delta_bb) {
      delta_bb.resize(get_output_size());
    }

    // updaters
    for(auto & ww_x_updater: _ww_x_updater) {
      ww_x_updater = updater->copy();
      YANN_SLOW_CHECK(ww_x_updater);
      ww_x_updater->init(input_size, get_output_size());
    }
    for(auto & ww_h_updater: _ww_h_updater) {
      ww_h_updater = updater->copy();
      YANN_SLOW_CHECK(ww_h_updater);
      ww_h_updater->init(get_output_size(), get_output_size());
    }
    for(auto & bb_updater: _bb_updater) {
      bb_updater = updater->copy();
      YANN_SLOW_CHECK(bb_updater);
      bb_updater->init(1, get_output_size()); // RowMajor
    }

    // gate radients
    _gradient_hh.resize(get_output_size());
    _gradient_cc.resize(get_output_size());
    _gradient_gate.resize(get_output_size());
    _gradient_zz_gate.resize(get_output_size());
    _tmp_derivative.resize(get_output_size());
  }

private:
  Matrix _delta_ww_x[LstmLayer::Gate_Max];
  Matrix _delta_ww_h[LstmLayer::Gate_Max];
  Vector _delta_bb[LstmLayer::Gate_Max];

  unique_ptr<Layer::Updater> _ww_x_updater[LstmLayer::Gate_Max];
  unique_ptr<Layer::Updater> _ww_h_updater[LstmLayer::Gate_Max];
  unique_ptr<Layer::Updater> _bb_updater[LstmLayer::Gate_Max];

  Vector _gradient_hh;
  Vector _gradient_cc;
  Vector _gradient_gate;
  Vector _gradient_zz_gate;
  Vector _tmp_derivative;
}; // class LstmLayer_TrainingContext

}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::LstmLayer implementation
//
yann::LstmLayer::LstmLayer(
    const MatrixSize & input_size,
    const MatrixSize & output_size) :
    _gate_activation_function(new SigmoidFunction()),
    _io_activation_function(new TanhFunction())
{
  YANN_CHECK_GT(input_size, 0);

  for(auto & ww_x: _ww_x) {
    ww_x.resize(input_size, output_size);
  }
  for(auto & ww_h: _ww_h) {
    ww_h.resize(output_size, output_size);
  }
  for(auto & bb: _bb) {
    bb.resize(output_size);
  }
}

yann::LstmLayer::~LstmLayer()
{
}

void yann::LstmLayer::set_values(
    const Matrix (& ww_x)[Gate_Max],
    const Matrix (& ww_h)[Gate_Max],
    const Vector (& bb)[Gate_Max])
{
  for(auto ii = 0; ii < Gate_Max; ++ii) {
    YANN_CHECK(is_same_size(ww_x[ii], _ww_x[ii]));
    YANN_CHECK(is_same_size(ww_h[ii], _ww_h[ii]));
    YANN_CHECK(is_same_size(bb[ii], _bb[ii]));

    _ww_x[ii] = ww_x[ii];
    _ww_h[ii] = ww_h[ii];
    _bb[ii] = bb[ii];
  }
}

void yann::LstmLayer::set_activation_functions(
    const std::unique_ptr<ActivationFunction> & gate_activation_function,
    const std::unique_ptr<ActivationFunction> & io_activation_function)
{
  YANN_CHECK(gate_activation_function);
  YANN_CHECK(io_activation_function);
  _gate_activation_function = gate_activation_function->copy();
  _io_activation_function = io_activation_function->copy();
}

// Layer overwrites
bool yann::LstmLayer::is_valid() const
{
  if(!Base::is_valid()) {
    return false;
  }
  if(!_io_activation_function || !_gate_activation_function) {
    return false;
  }
  for(auto ii = 0; ii < Gate_Max; ++ii) {
    if(!is_same_size(_ww_x[0], _ww_x[ii])) {
      return false;
    }
    if(!is_same_size(_ww_h[0], _ww_h[ii])) {
      return false;
    }
    if(!is_same_size(_bb[0], _bb[ii])) {
      return false;
    }
  }

  return true;
}

std::string yann::LstmLayer::get_name() const
{
  return "LstmLayer";
}

string yann::LstmLayer::get_info() const
{
  YANN_CHECK(is_valid());

  ostringstream oss;
  oss << Base::get_info()
      << ", state activation: " << _io_activation_function->get_info()
      << ", output activation: " << _gate_activation_function->get_info()
      ;
  return oss.str();
}

bool yann::LstmLayer::is_equal(const Layer & other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
    return false;
  }
  auto the_other = dynamic_cast<const LstmLayer*>(&other);
  if(the_other == nullptr) {
    return false;
  }
  // TOOD: add deep compare
  if(_io_activation_function->get_info() != the_other->_io_activation_function->get_info()) {
    return false;
  }
  if(_gate_activation_function->get_info() != the_other->_gate_activation_function->get_info()) {
    return false;
  }
  for(auto ii = 0; ii < Gate_Max; ++ii) {
    if(!_ww_x[ii].isApprox(the_other->_ww_x[ii], tolerance)) {
      return false;
    }
    if(!_ww_h[ii].isApprox(the_other->_ww_h[ii], tolerance)) {
      return false;
    }
    if(!_bb[ii].isApprox(the_other->_bb[ii], tolerance)) {
      return false;
    }
  }
  return true;
}

MatrixSize yann::LstmLayer::get_input_size() const
{
  YANN_SLOW_CHECK(is_valid());
  return _ww_x[Gate_A].rows();
}

MatrixSize yann::LstmLayer::get_output_size() const
{
  YANN_SLOW_CHECK(is_valid());
  return _ww_x[Gate_A].cols();
}

unique_ptr<Layer::Context> yann::LstmLayer::create_context(const MatrixSize & batch_size) const
{
  YANN_CHECK(is_valid());
  return make_unique<LstmLayer_Context>(get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::LstmLayer::create_context(const RefVectorBatch & output) const
{
  YANN_CHECK(is_valid());
  return make_unique<LstmLayer_Context>(output);
}
unique_ptr<Layer::Context> yann::LstmLayer::create_training_context(
    const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(updater);
  return make_unique<LstmLayer_TrainingContext>(
      get_input_size(), get_output_size(), batch_size, updater);
}
unique_ptr<Layer::Context> yann::LstmLayer::create_training_context(
    const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(updater);
  return make_unique<LstmLayer_TrainingContext>(
      get_input_size(), output, updater);
}

template<typename InputType>
void yann::LstmLayer::feedforward_internal(
    const InputType & input,
    Context * context,
    enum OperationMode mode) const
{
  auto ctx = dynamic_cast<LstmLayer_Context *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_LE(get_batch_size(input) + ctx->_pos, ctx->get_batch_size());
  YANN_CHECK_LE(get_batch_size(input), get_batch_size(ctx->get_output()));
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());

  for(MatrixSize ii = 0; ii < get_batch_size(input); ++ii, ++ctx->_pos) {
    // Phase 1: I/O gates
    //  zz_a(t) = ww_xa*x(t) + ww_ha * h(t-1) + bb_a  -- input
    //  zz_i(t) = ww_xi*x(t) + ww_hi * h(t-1) + bb_i  -- input gate
    //  zz_f(t) = ww_xf*x(t) + ww_hf * h(t-1) + bb_g  -- forget gate
    //  zz_o(t) = ww_xo*x(t) + ww_ho * h(t-1) + bb_o  -- output gate
    auto in  = get_batch(input, ii);
    auto out = get_batch(ctx->get_output(), ctx->_pos);
    auto hh  = get_batch(ctx->_hh, ctx->_pos);
    if(ctx->_pos > 0) {
      const auto & hh_prev = get_batch(ctx->_hh, ctx->_pos - 1);
      for(auto jj = 0; jj < Gate_Max; ++jj) {
        auto zz_gate = get_batch(ctx->_zz_gate[jj], ctx->_pos);
        zz_gate.noalias() = MatrixFunctions<InputType>::product(in, _ww_x[jj]);
        zz_gate.noalias() += MatrixFunctions<RefConstMatrix>::product(hh_prev, _ww_h[jj]) + _bb[jj];
      }
    } else {
      for(auto jj = 0; jj < Gate_Max; ++jj) {
        auto zz_gate = get_batch(ctx->_zz_gate[jj], ctx->_pos);
        zz_gate.noalias() = MatrixFunctions<InputType>::product(in, _ww_x[jj]);
        zz_gate.noalias() += _bb[jj];
      }
    }

    //  a(t) = activ1 (zz_a(t))
    //  i(t) = activ2 (zz_i(t))
    //  f(t) = activ2 (zz_f(t))
    //  o(t) = activ2 (zz_o(t))
    for(auto jj = 0; jj < Gate_Max; ++jj) {
      const auto zz_gate = get_batch(ctx->_zz_gate[jj], ctx->_pos);
      auto gate = get_batch(ctx->_gate[jj], ctx->_pos);
      if(jj > 0) {
        _gate_activation_function->f(zz_gate, gate, Operation_Assign);
      } else {
        _io_activation_function->f(zz_gate, gate, Operation_Assign);
      }
    }

    // Phase 2: State
    //  c(t) = elem_prod(i(t), a(t)) + elem_prod(f(t), c(t-1))
    const auto gate_aa = get_batch(ctx->_gate[Gate_A], ctx->_pos);
    const auto gate_ii = get_batch(ctx->_gate[Gate_I], ctx->_pos);
    const auto gate_ff = get_batch(ctx->_gate[Gate_F], ctx->_pos);
    auto cc = get_batch(ctx->_cc, ctx->_pos);
    if(ctx->_pos > 0) {
      const auto & cc_prev = get_batch(ctx->_cc, ctx->_pos - 1);
      cc.array() = gate_ii.array() * gate_aa.array() + gate_ff.array() * cc_prev.array();
    } else {
      cc.array() = gate_ii.array() * gate_aa.array();
    }

    // Phase 3: Output
    //  h(t) = elem_prod(o(t), activ1(c(t))
    const auto gate_oo = get_batch(ctx->_gate[Gate_O], ctx->_pos);
    auto activ_cc = get_batch(ctx->_activ_cc, ctx->_pos);
    _io_activation_function->f(cc, activ_cc, Operation_Assign);
    hh.array() = gate_oo.array() * activ_cc.array();

    // done!
    switch(mode) {
    case Operation_Assign:
      out.noalias() = hh;
      break;
    case Operation_PlusEqual:
      out.noalias() += hh;
      break;
    }
  }
}

void yann::LstmLayer::feedforward(
    const RefConstVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

void yann::LstmLayer::feedforward(
    const RefConstSparseVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  feedforward_internal<RefConstSparseMatrix>(input, context, mode);
}

template<
  typename MainInputType,
  typename InputType,
  typename GradientInputType,
  typename ContextType
>
void yann::LstmLayer::backprop_gate(
    enum IOGates gate,
    const std::unique_ptr<ActivationFunction> & activation_function,
    const RefConstVector & gradient_gate,
    const InputType & input,
    boost::optional<GradientInputType> gradient_in,
    ContextType * ctx) const
{
  YANN_SLOW_CHECK(activation_function);
  YANN_SLOW_CHECK(ctx);

  const auto zz_gate = get_batch(ctx->_zz_gate[gate], ctx->_pos);
  const auto & ww_x = _ww_x[gate];
  const auto & ww_h = _ww_h[gate];
  auto & gradient_zz_gate = ctx->_gradient_zz_gate;
  auto & tmp_derivative = ctx->_tmp_derivative;

  // gate(t) = activ(zz_gate(t))
  //
  // grad(zz_gate) = grad(gate) * activ'(zz_gate)
  activation_function->derivative(zz_gate, tmp_derivative);
  gradient_zz_gate.array() = gradient_gate.array() * tmp_derivative.array();

  // zz_gate(t) = ww_x_gate*x(t) + ww_h_gate * h(t-1) + bb_gate
  //
  // grad(ww_x_gate) = grad(zz_gate) * d zz_gate/dww_x_gate = grad(zz_gate) * x(t)
  // grad(ww_h_gate) = grad(zz_gate) * d zz_gate/dww_h_gate = grad(zz_gate) * h(t-1)
  // grad(b) = grad(zz_gate) * d zz_gate/db = grad(zz_gate)
  YANN_SLOW_CHECK(is_same_size(ctx->_delta_ww_x[gate], _ww_x[gate]));
  YANN_SLOW_CHECK(is_same_size(ctx->_delta_ww_h[gate], _ww_h[gate]));
  YANN_SLOW_CHECK(is_same_size(ctx->_delta_bb[gate], _bb[gate]));
  ctx->_delta_ww_x[gate] += MatrixFunctions<MainInputType>::product(input.transpose(), gradient_zz_gate);
  if(ctx->_pos > 0) {
    auto prev_hh = get_batch(ctx->_hh, ctx->_pos - 1);
    ctx->_delta_ww_h[gate] += MatrixFunctions<RefConstMatrix>::product(prev_hh.transpose(), gradient_zz_gate);
  }
  ctx->_delta_bb[gate].noalias() += gradient_zz_gate;

  // zz_gate(t) = ww_x_gate*x(t) + ww_h_gate * h(t-1) + bb_gate
  //
  // grad(h) = grad(zz_gate) * d zz_gate/dh = grad(zz_gate) * transp(ww_h_gate)
  if(ctx->_pos > 0) {
    auto & gradient_hh = ctx->_gradient_hh;
    gradient_hh.noalias() += MatrixFunctions<RefConstMatrix>::product(gradient_zz_gate, ww_h.transpose());
  }

  // zz_gate(t) = ww_x_gate*x(t) + ww_h_gate * h(t-1) + bb_gate
  //
  //  grad(x) = grad(zz_gate) * d zz_gate/dx = grad(zz_gate) * transp(ww_x_gate)
  if(gradient_in) {
    (*gradient_in).noalias() += MatrixFunctions<RefConstMatrix>::product(gradient_zz_gate, ww_x.transpose());
  }
}

template<typename InputType>
void yann::LstmLayer::backprop_internal(
    const RefConstVectorBatch & gradient_output,
    const InputType & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  auto ctx = dynamic_cast<LstmLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(gradient_output), 0);
  YANN_CHECK_LE(get_batch_size(gradient_output), ctx->_pos);
  YANN_CHECK_EQ(get_batch_item_size(gradient_output), get_output_size());
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(gradient_output));
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK(!gradient_input || is_same_size(input, *gradient_input));

  // Zero out since we will be adding to it
  if(gradient_input) {
    gradient_input->setZero();
  }

  for(MatrixSize ii = get_batch_size(gradient_output) - 1; ii >= 0 && (--ctx->_pos) >= 0; --ii) {
    const auto in = get_batch(input, ii);
    auto gradient_in = gradient_input ? make_optional(get_batch(*gradient_input, ii)) : boost::none;
    const auto gate_aa  = get_batch(ctx->_gate[Gate_A], ctx->_pos);
    const auto gate_ii  = get_batch(ctx->_gate[Gate_I], ctx->_pos);
    const auto gate_ff  = get_batch(ctx->_gate[Gate_F], ctx->_pos);
    const auto gate_oo  = get_batch(ctx->_gate[Gate_O], ctx->_pos);
    auto & gradient_gate = ctx->_gradient_gate;

    // note that output gradient also has the internal feedback loop
    auto & gradient_hh = ctx->_gradient_hh;
    gradient_hh += get_batch(gradient_output, ii);

    // State cell
    //
    //  h(t) = elem_prod(o(t), activ1(c(t))
    //  grad(c) = grad(h) * d h/dactiv1(c) * activ1'(c) = elem_prod(grad(h), o, activ1(c)
    const auto cc = get_batch(ctx->_cc, ctx->_pos);
    auto & gradient_cc = ctx->_gradient_cc;
    auto & tmp_derivative = ctx->_tmp_derivative;
    _io_activation_function->derivative(cc, tmp_derivative);
    // add to the gradient from previous step
    gradient_cc.array() += gradient_hh.array() * gate_oo.array() * tmp_derivative.array();


    // Gate O: Output gate:
    //
    //  h(t) = elem_prod(o(t), activ1(c(t))
    //  grad(o) = grad(h) * dh/do = elem_prod(grad(h), activ1(c))
    const auto activ_cc = get_batch(ctx->_activ_cc, ctx->_pos);
    gradient_gate.array() = gradient_hh.array() * activ_cc.array();
    gradient_hh.setZero(); // we will update it in backprop_gate for next step
    backprop_gate<InputType>(Gate_O, _gate_activation_function, gradient_gate, in, gradient_in, ctx);

    // Gate A
    //
    //  c(t) = elem_prod(i(t), a(t)) + elem_prod(f(t), c(t-1))
    //  grad(a) = grad(c) * dc/da = elem_prod(grad(c), i(t))
    gradient_gate.array() = gradient_cc.array() * gate_ii.array();
    backprop_gate<InputType>(Gate_A, _io_activation_function, gradient_gate, in, gradient_in, ctx);

    // Gate I: Input gate
    //
    //  c(t) = elem_prod(i(t), a(t)) + elem_prod(f(t), c(t-1))
    //  grad(i) = grad(c) * dc/di = elem_prod(grad(c), a(t))
    gradient_gate.array() = gradient_cc.array() * gate_aa.array();
    backprop_gate<InputType>(Gate_I, _gate_activation_function, gradient_gate, in, gradient_in, ctx);

    // Gate F: Forget gate
    //
    //  c(t) = elem_prod(i(t), a(t)) + elem_prod(f(t), c(t-1))
    //  grad(f) = grad(c) * dc/df = elem_prod(grad(c), c(t-1))
    if(ctx->_pos > 0) {
      const auto prev_cc = get_batch(ctx->_activ_cc, ctx->_pos - 1);
      gradient_gate.array() = gradient_cc.array() * prev_cc.array();
    } else {
      gradient_gate.setZero();
    }
    backprop_gate<InputType>(Gate_F, _gate_activation_function, gradient_gate, in, gradient_in, ctx);

    // State gradient for prev step
    //
    //  c(t) = elem_prod(i(t), a(t)) + elem_prod(f(t), c(t-1))
    //  grad(c(t-1)) = grad(c) * dc(t)/dc(t-1) = elem_prod(grad(c(t)), f(t))
    if(ctx->_pos > 0) {
      gradient_cc.array() = gradient_cc.array() * gate_ff.array();
    }
  }
}

void yann::LstmLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  backprop_internal(gradient_output, input, gradient_input, context);
}

void yann::LstmLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstSparseVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  // <RefConstSparseMatrix> is required for MatrixFunctions<>::product
  backprop_internal<RefConstSparseMatrix>(gradient_output, input, gradient_input, context);
}

void yann::LstmLayer::init(enum InitMode mode, boost::optional<InitContext> init_context)
{
  switch (mode) {
  case InitMode_Zeros:
    for(auto & ww_x: _ww_x) {
      ww_x.setZero();
    }
    for(auto & ww_h: _ww_h) {
      ww_h.setZero();
    }
    for(auto & bb: _bb) {
      bb.setZero();
    }
    break;
  case InitMode_Random:
    {
      unique_ptr<RandomGenerator> gen01 = RandomGenerator::normal_distribution(0, 1,
          init_context ? optional<Value>(init_context->seed()) : boost::none);
      for(auto & ww_x: _ww_x) {
        gen01->generate(ww_x);
      }
      for(auto & ww_h: _ww_h) {
        gen01->generate(ww_h);
      }
      for(auto & bb: _bb) {
        gen01->generate(bb);
      }
    }
    break;
  }
}

void yann::LstmLayer::update(Context * context, const size_t & tests_num)
{
  auto ctx = dynamic_cast<LstmLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);

  for(auto ii = 0; ii < Gate_Max; ++ii) {
    YANN_SLOW_CHECK(ctx->_ww_x_updater[ii]);
    YANN_SLOW_CHECK(ctx->_ww_h_updater[ii]);
    YANN_SLOW_CHECK(ctx->_bb_updater[ii]);

    ctx->_ww_x_updater[ii]->update(ctx->_delta_ww_x[ii], tests_num, _ww_x[ii]);
    ctx->_ww_h_updater[ii]->update(ctx->_delta_ww_h[ii], tests_num, _ww_h[ii]);
    ctx->_bb_updater[ii]->update(ctx->_delta_bb[ii], tests_num, _bb[ii]);
  }
}

// the format is (no new lines):
// (wxa:<_ww_x[Gate_A]>,wxh:<_ww_h[Gate_A]>,ba:<_bb[Gate_A]>,
//  wxi:<_ww_x[Gate_I]>,wxi:<_ww_h[Gate_I]>,bi:<_bb[Gate_I]>,
//  wxf:<_ww_x[Gate_F]>,wxf:<_ww_h[Gate_F]>,bf:<_bb[Gate_F]>,
//  wxo:<_ww_x[Gate_O]>,wxo:<_ww_h[Gate_O]>,bo:<_bb[Gate_O]>)
void yann::LstmLayer::read(std::istream & is)
{
  Base::read(is);

  read_char(is, '(');

  for(auto ii = 0; ii < Gate_Max; ++ii) {
    const char & suffix =  g_ltsm_names_suffix[ii];
    if(ii > 0) {
      read_char(is, ',');
    }
    read_object(is, string("wx") + suffix, _ww_x[ii]);
    read_char(is, ',');
    read_object(is, string("wh") + suffix, _ww_h[ii]);
    read_char(is, ',');
    read_object(is, string("b") + suffix, _bb[ii]);
  }
  read_char(is, ')');
}

// the format is (no new lines):
// (wxa:<_ww_x[Gate_A]>,wxh:<_ww_h[Gate_A]>,ba:<_bb[Gate_A]>,
//  wxi:<_ww_x[Gate_I]>,wxi:<_ww_h[Gate_I]>,bi:<_bb[Gate_I]>,
//  wxf:<_ww_x[Gate_F]>,wxf:<_ww_h[Gate_F]>,bf:<_bb[Gate_F]>,
//  wxo:<_ww_x[Gate_O]>,wxo:<_ww_h[Gate_O]>,bo:<_bb[Gate_O]>)
void yann::LstmLayer::write(std::ostream & os) const
{
  Base::write(os);

  os << "(";
  for(auto ii = 0; ii < Gate_Max; ++ii) {
    const char & suffix =  g_ltsm_names_suffix[ii];
    if(ii > 0) {
      os << ",";
    }
    write_object(os, string("wx") + suffix, _ww_x[ii]);
    os << ",";
    write_object(os, string("wh") + suffix, _ww_h[ii]);
    os << ",";
    write_object(os, string("b") + suffix, _bb[ii]);
  }
  os << ")";
}
